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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="lexi_transcripts",
    persist_directory="chroma_db", # on cloud
    # persist_directory="15.LexiChat/chroma_db", #locally at this path
    embedding_function=embedder,
)


#fetch k - fetches the top20 most similar chunks
#k- from the chunks fetched above select the best 6
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)


mq_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")


#for all the questions llm generates from the query it fetches the top 6 relevant chunks 
# and then finally all the retrieved chunks are combined and duplicated documents are removed 
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
    llm=mq_llm,
)

#bm25 used for keyword search and retrieves the content where there are keyword matches
#here we have kept k=6 
docs = vector_store._collection.get(include=["documents", "metadatas"])

all_docs = []
if docs and docs.get("documents"):
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(docs["documents"], docs["metadatas"])
    ]

if all_docs:
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 6
else:
    bm25 = None



#retrieved list contains list of all 3 retrievers
#and then we loop over documents retrieved by each retriever
#we use hash to remove any duplicated results
#and finally returns all the list of unique Document objects
def combine_results(*retrieved_lists):
    seen = set()
    unique_docs = []

    for retrieved in retrieved_lists:
        for d in retrieved:
            key = d.metadata.get("text_hash", d.page_content[:80])
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

    return unique_docs


def hybrid_retrieve(query):
    r1 = mmr_retriever.invoke(query)
    r2 = multi_retriever.invoke(query)

    results = [r1, r2]

    if bm25:
        r3 = bm25.invoke(query)
        results.append(r3)

    combined = combine_results(*results)
    return combined[:20]
  #we finally return the top 20 documents 


#here since the retrieved context is in the form of document object we make this into a structured format in the form of list of dictionary which we can then pass into the llm 
#and remove the unnecessary metadata as it is not needed
def format_docs_structured(docs):
    structured = []
    for i, d in enumerate(docs):
        structured.append({
            "id": i,
            "text": d.page_content,
            "timestamp": int(d.metadata["start"]),
            "url": d.metadata["citation_url"],
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

        blocks.append(f"""Chunk {d['id']}:Text: {d['text']}Timestamp: [{ts}]({d['url']})""")
    return "\n".join(blocks)




#here we use the prompt template class so insert the chunks, questions dynamically into the prompt
#we strictly force the llm to answer only from the context and not from its own knowledge and not hallucinate 
#we also tell the model to generate clickable timestamps for every answer it generates 
rank_prompt = PromptTemplate.from_template(
"""
You must answer ONLY using the context chunks below.

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
1. <Answer written in 3–4 complete sentences.> [MM:SS](URL)
2. <Answer written in 3–4 complete sentences.> [MM:SS](URL)
3. <Answer written in 3–4 complete sentences.> [MM:SS](URL)

----------------------
CONTEXT CHUNKS:
{context}
----------------------

QUESTION:
{question}

IMPORTANT:
If an answer is shorter than two full sentences, REWRITE it to meet the rules.
"""
)


# temperature is kept to zero so has to have reproducibilty in the answers
llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)


#this function is finally used to run the complete retrieval pipeline
# and it returns the final answer and retrieved context which will be needed while evaluating the RAG System


def ask(question: str):
    
    docs = hybrid_retrieve(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    final_prompt=rank_prompt.format(context=prompt_context,question=question)
    response = llm.invoke(final_prompt).content

    return {
        "answer": response,
        "retrieved_contexts": [d["text"] for d in structured_docs],
    }