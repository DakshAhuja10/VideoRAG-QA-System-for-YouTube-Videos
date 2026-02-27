#Here we use a hybrid retrieval logic to retrieve relevant context as per the user query
#for retrieving we use BM25 Retriever,MultiQuery Retriever ,MMR Retriever

#bm25 retriever retrievers on the basis of keywords

#multiquery retrievers helps in the case where the user query is ambigious
#it uses an llm and the user query and ask the llm to generate possible questions from the user #query and for all the questions llm generates it retrieves the context

#here for multi query retriever we have use the gemini-2.5-flash model and for generating
#the final answer we have used openi/gpt-oss-120B model available on Groq

#mmr retriever or Maximal Marginal relevance retriever is used to get results which are relevant 
# to the user query and also diverse in nature this prevents the llm to generate duplicate answers 
#there may be a case when llm might give a same answer twice this prevents us from that 
#(Relevant+Diverse)

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import uuid

# Import configuration
from config import (
    RetrievalConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
    PromptConfig,
)

# Import metrics tracking
# from metrics_tracker import track_latency, track_api_call

load_dotenv()


# Initialize embedder with config
embedder = HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)

# Lightweight but very effective re-ranker
rerank_model = CrossEncoder(RetrievalConfig.RERANK_MODEL)

vector_store = Chroma(
    collection_name=VectorStoreConfig.COLLECTION_NAME,
    persist_directory=VectorStoreConfig.PERSIST_DIRECTORY,
    embedding_function=embedder,
)


# MMR retriever with config values
# fetch_k - fetches the top candidates for MMR to select from
# k - from the candidates, select the best k with diversity
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RetrievalConfig.MMR_K, "fetch_k": RetrievalConfig.MMR_FETCH_K},
)


# MultiQuery LLM with config
mq_llm = ChatGoogleGenerativeAI(model=LLMConfig.MULTIQUERY_MODEL)


# MultiQuery retriever with config
# For all the questions LLM generates from the query, it fetches the top k relevant chunks
# and then finally all the retrieved chunks are combined and duplicated documents are removed
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": RetrievalConfig.MULTIQUERY_K}),
    llm=mq_llm,
)

# BM25 used for keyword search and retrieves the content where there are keyword matches
# Initialize with all documents from vector store
docs = vector_store._collection.get(include=["documents", "metadatas"])

all_docs = []
if docs and docs.get("documents"):
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(docs["documents"], docs["metadatas"])
    ]

if all_docs:
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = RetrievalConfig.BM25_K  # Use config value
else:
    bm25 = None



#retrieved list contains list of all 3 retrievers
#and then we loop over documents retrieved by each retriever
#we use hash to remove any duplicated results
#and finally returns all the list of unique Document objects
# Deduplicate documents based on content hash
def combine_results(*retrieved_lists):
    unique_docs = []
    seen = set()
    for retrieved in retrieved_lists:
        for d in retrieved:
            # use a hash of the content for deduplication
            content_hash = d.metadata.get("text_hash", hash(d.page_content))
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(d)
    return unique_docs


def hybrid_retrieve(query):
    # Phase 1: Broad Retrieval (Greater k to allow re-ranker to find the best)
    r1 = mmr_retriever.invoke(query)  # Fetches based on relevance + diversity
    r2 = multi_retriever.invoke(query) # Fetches based on LLM query expansions

    results = [r1, r2]

    if bm25:
        r3 = bm25.invoke(query) # Keyword matching
        results.append(r3)

    combined = combine_results(*results)

    # Phase 2: Re-ranking (Cross-Encoder)
    if combined:
        # Prepare pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in combined]
        scores = rerank_model.predict(pairs)

        # Attach scores and sort
        for i, doc in enumerate(combined):
            doc.metadata["rerank_score"] = float(scores[i])

        combined.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

    # Return top N most relevant after re-ranking (from config)
    return combined[:RetrievalConfig.RERANK_TOP_N]


#here since the retrieved context is in the form of document object we make this into a structured format in the form of list of dictionary which we can then pass into the llm 
#and remove the unnecessary metadata as it is not needed
def format_docs_structured(docs):
    structured = []
    for i, d in enumerate(docs):
        # Fallback for older documents that might not have citation_url
        citation_url = d.metadata.get("citation_url")
        if not citation_url and "url" in d.metadata and "start" in d.metadata:
            citation_url = f"{d.metadata['url']}&t={int(d.metadata['start'])}s"
            
        structured.append({
            "id": i,
            "text": d.page_content,
            "timestamp": int(d.metadata.get("start", 0)),
            "url": citation_url or d.metadata.get("url", ""),
        })
    return structured

#here we take the structured_docs and convert the timestamps into mm:ss 
#so the prompt is like 
# Chunk 3:
# Text: 
# Timestamp: [12:45]

def docs_to_prompt_context(structured_docs):
    blocks = []
    for d in structured_docs:
        mm = d["timestamp"] // 60
        ss = d["timestamp"] % 60
        ts = f"{mm:02d}:{ss:02d}"

        block = f"--- Source ---\n"
        block += f"Timestamp: [{ts}]({d['url']})\n"
        block += f"Text Content: {d['text']}"
        blocks.append(block)
    return "\n\n".join(blocks)




#here we use the prompt template class so insert the chunks, questions dynamically into the prompt
#we strictly force the llm to answer only from the context and not from its own knowledge and not hallucinate 
#we also tell the model to generate clickable timestamps for every answer it generates 
rank_prompt = PromptTemplate.from_template(
f"""You must answer ONLY using the context chunks below.

If none of the chunks answer the question, reply EXACTLY:
"I don't know. The transcripts do not contain the answer."

CRITICAL RULES (MANDATORY):
1. Use ONLY the provided timestamps and URLs.
2. Do NOT invent or modify timestamps.
3. Do NOT merge multiple chunks into one answer.
4. Select EXACTLY the TOP 3 most relevant chunks.
5. EACH answer must contain **at least TWO complete, grammatically correct sentences**.
6. EACH answer must be **explanatory**, not a phrase, heading, or fragment.
7. EACH answer must be **3–4 full sentences total**.
8. EACH answer must end with **ONE clickable timestamp**.
9. DO NOT output sentence fragments, titles, or transcript headings.

STRICT OUTPUT FORMAT (NO DEVIATION):
1. <Answer written in 3–4 complete sentences.> [[MM:SS]](URL)
2. <Answer written in 3–4 complete sentences.> [[MM:SS]](URL)
3. <Answer written in 3–4 complete sentences.> [[MM:SS]](URL)

----------------------
CONTEXT CHUNKS:
{{context}}
----------------------

QUESTION:
{{question}}

IMPORTANT:
If an answer is shorter than two full sentences, REWRITE it to meet the rules.
"""
)


# LLM for answer generation - using config values
llm = ChatGroq(
    model=LLMConfig.ANSWER_MODEL,
    temperature=LLMConfig.ANSWER_TEMPERATURE
)


#this function is finally used to run the complete retrieval pipeline
# and it returns the final answer and retrieved context which will be needed while evaluating the RAG System


def ask(question: str):
    """
    Non-streaming version - returns complete answer with metrics tracking
    """
    query_id = str(uuid.uuid4())
    
    # Track retrieval latency
    # with track_latency("retrieval", query_id=query_id):
    docs = hybrid_retrieve(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    # Track LLM generation latency
    # with track_latency("llm_generation", query_id=query_id):
    final_prompt = rank_prompt.format(context=prompt_context, question=question)
    response = llm.invoke(final_prompt).content
        
    # Estimate token usage (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = (len(final_prompt) + len(response)) // 4
    # track_api_call("groq", estimated_tokens, query_id=query_id)

    return {
        "answer": response,
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
    }


def ask_stream(question: str):
    """
    Streaming version - yields tokens as they are generated with metrics tracking
    Returns a generator that yields:
    - {"type": "token", "content": <token>} for each token
    - {"type": "done", "retrieved_contexts": [...], "query_id": ...} when complete
    """
    query_id = str(uuid.uuid4())
    import time
    
    # Track retrieval latency
    # with track_latency("retrieval", query_id=query_id):
    docs = hybrid_retrieve(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)
    final_prompt = rank_prompt.format(context=prompt_context, question=question)
    
    # Track streaming LLM generation
    start_time = time.time()
    token_count = 0
    response_text = ""
    
    # Stream tokens from the LLM
    for chunk in llm.stream(final_prompt):
        if chunk.content:
            token_count += 1
            response_text += chunk.content
            yield {"type": "token", "content": chunk.content}
    
    # Log streaming latency and token usage
    duration_ms = (time.time() - start_time) * 1000
    estimated_tokens = (len(final_prompt) + len(response_text)) // 4
    
    # with track_latency("llm_streaming", query_id=query_id):
    #     # This will log 0ms since we already measured, but records the event
    #     pass
    
    # track_api_call("groq", estimated_tokens, query_id=query_id)
    
    # Send the retrieved contexts at the end
    yield {
        "type": "done",
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
    }