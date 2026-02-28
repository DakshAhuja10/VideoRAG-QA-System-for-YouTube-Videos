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

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
import streamlit as st

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


# ── Cached resource loaders ──────────────────────────────────────────────────
# @st.cache_resource ensures these heavy objects are initialised once per
# process and reused across all sessions/reruns, including cold starts on
# Streamlit Cloud.  The underscore prefix on helper names keeps them private.

@st.cache_resource(show_spinner=False)
def _load_embedder():
    return HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)


@st.cache_resource(show_spinner=False)
def _load_rerank_model():
    # CrossEncoder model weights are downloaded once and cached in memory
    return CrossEncoder(RetrievalConfig.RERANK_MODEL)


@st.cache_resource(show_spinner=False)
def _load_vector_store():
    return Chroma(
        collection_name=VectorStoreConfig.COLLECTION_NAME,
        persist_directory=VectorStoreConfig.PERSIST_DIRECTORY,
        embedding_function=_load_embedder(),
    )


@st.cache_resource(show_spinner=False)
def _load_retrievers():
    vs = _load_vector_store()
    mq_llm = ChatGoogleGenerativeAI(model=LLMConfig.MULTIQUERY_MODEL)
    # MMR retriever — relevance + diversity
    mmr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RetrievalConfig.MMR_K, "fetch_k": RetrievalConfig.MMR_FETCH_K},
    )
    # MultiQuery retriever — handles ambiguous queries via LLM expansion
    mq = MultiQueryRetriever.from_llm(
        retriever=vs.as_retriever(search_kwargs={"k": RetrievalConfig.MULTIQUERY_K}),
        llm=mq_llm,
    )
    return mmr, mq


@st.cache_resource(show_spinner=False)
def _load_bm25():
    # BM25 requires loading all documents from ChromaDB to build its index.
    # Caching this avoids the expensive full-collection scan on every rerun.
    vs = _load_vector_store()
    raw = vs._collection.get(include=["documents", "metadatas"])
    all_docs = []
    if raw and raw.get("documents"):
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
    if all_docs:
        b = BM25Retriever.from_documents(all_docs)
        b.k = RetrievalConfig.BM25_K
        return b
    return None


@st.cache_resource(show_spinner=False)
def _load_answer_llm():
    return ChatGroq(
        model=LLMConfig.ANSWER_MODEL,
        temperature=LLMConfig.ANSWER_TEMPERATURE,
    )


# Resolve all resources at module level.
# After the first call these are instant lookups from the cache.
embedder        = _load_embedder()
rerank_model    = _load_rerank_model()
vector_store    = _load_vector_store()
mmr_retriever, multi_retriever = _load_retrievers()
bm25            = _load_bm25()


# BM25 is initialised via _load_bm25() above and assigned to `bm25` at module level.



# ── Retrieval helpers ────────────────────────────────────────────────────────

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
    """Standard retrieval used for first-pass answers."""
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


def hybrid_retrieve_broad(query):
    """
    Wider retrieval used on retry when the first-pass confidence is low.

    Differences vs hybrid_retrieve:
    - MMR: fetches 40 candidates (vs 20) and returns top 12 (vs 6).
      A larger candidate pool forces MMR to explore more of the vector space,
      surfacing chunks that a shallow search misses.
    - MultiQuery: reuses the same retriever; because the underlying LLM is
      non-deterministic it will generate different query expansions, hitting
      different parts of the index.
    - BM25: returns 12 docs (vs 6), broadening keyword coverage.
    - Reranker: keeps top 15 after scoring (vs 10), giving the LLM a richer
      context window to find an explicit answer in.
    """
    # Create a wider MMR retriever on-the-fly using the cached vector_store.
    # This is cheap — it's just a config wrapper over the already-loaded Chroma
    # connection; no model weights are reloaded.
    mmr_broad = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40},
    )
    r1 = mmr_broad.invoke(query)

    # MultiQuery with a wider per-sub-query k
    mq_broad = vector_store.as_retriever(search_kwargs={"k": 10})
    mq_llm = ChatGoogleGenerativeAI(model=LLMConfig.MULTIQUERY_MODEL)
    multi_broad = MultiQueryRetriever.from_llm(retriever=mq_broad, llm=mq_llm)
    r2 = multi_broad.invoke(query)

    results = [r1, r2]

    if bm25:
        original_k = bm25.k
        bm25.k = 12          # temporarily widen BM25
        r3 = bm25.invoke(query)
        bm25.k = original_k  # restore so normal retrieval is unaffected
        results.append(r3)

    combined = combine_results(*results)

    # Re-rank and keep a larger top-N
    if combined:
        pairs = [[query, doc.page_content] for doc in combined]
        scores = rerank_model.predict(pairs)
        for i, doc in enumerate(combined):
            doc.metadata["rerank_score"] = float(scores[i])
        combined.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

    # Keep top 15 (vs default 10) to give the LLM more grounding material
    return combined[:15]


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
"""CONTEXT CHUNKS:
{context}

QUESTION: {question}

---
STRICT RULES — violating any rule is not allowed under any circumstances:

RULE 1: Use ONLY information that is EXPLICITLY stated in the context chunks above. Never use your own knowledge.

RULE 2: These words and phrases are FORBIDDEN in your response: "could be related", "might be", "possibly", "more research needed", "I cannot confirm", "without more information", "the context mentions", "the chunk mentions", "the discussion is", "the speaker mentions", "as mentioned", "discussion about", "it is mentioned that", "the transcripts mention", "suggests a connection".

RULE 3: Before writing any answer, ask yourself: "Is there at least one chunk that is PRIMARILY and SUBSTANTIALLY about this exact topic — not just a passing word match?" 
         - If NO → output this single line and stop: I don't know. The transcripts do not contain information about this topic.
         - If you find yourself writing "the context does not provide a clear explanation" or similar — that means NO. Stop and output the refusal line.
         - A tangential or incidental mention of a word is NOT enough. The chunk must be substantially about the topic.

RULE 4: If the context IS substantially about the topic, write exactly 3 numbered answers:
         - Each answer must directly explain the actual content — facts, concepts, examples, processes, names, numbers — as if teaching the user.
         - Each answer must be 3-4 sentences of real substance.
         - Each answer must end with this EXACT clickable format: [[MM:SS]](URL) — copy URL and timestamp verbatim from the context.
         - No preamble before answer 1.

RULE 5: Each answer must come from a DIFFERENT chunk with distinct, non-overlapping information.
"""
)


# Answer-generation LLM — cached via _load_answer_llm() above
llm = _load_answer_llm()

_DONT_KNOW = "I don't know. The transcripts do not contain information about this topic."


def _check_relevance(docs: list) -> bool:
    """
    Score-based relevance gate using CrossEncoder rerank scores.
    Returns False (= refuse to answer) when the best-matching chunk scored
    below RERANK_MIN_SCORE, indicating no chunk is truly relevant to the query.
    """
    if not docs:
        return False
    top_score = docs[0].metadata.get("rerank_score", 0.0)
    import logging
    logging.getLogger(__name__).info(f"Top rerank score: {top_score:.3f} (threshold: {RetrievalConfig.RERANK_MIN_SCORE})")
    return top_score >= RetrievalConfig.RERANK_MIN_SCORE


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

    if not _check_relevance(docs):
        return {
            "answer": _DONT_KNOW,
            "retrieved_contexts": [d["text"] for d in structured_docs],
            "query_id": query_id,
        }

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )
    response = llm.invoke(final_prompt).content

    return {
        "answer": response,
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
    }


def ask_stream(question: str):
    """
    Streaming version — standard first-pass retrieval.
    Yields:
    - {"type": "token", "content": <token>} for each token
    - {"type": "done", "retrieved_contexts": [...], "query_id": ...} when complete
    """
    query_id = str(uuid.uuid4())

    docs = hybrid_retrieve(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    if not _check_relevance(docs):
        yield {"type": "token", "content": _DONT_KNOW}
        yield {"type": "done", "retrieved_contexts": [d["text"] for d in structured_docs], "query_id": query_id}
        return

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )

    for chunk in llm.stream(final_prompt):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    yield {
        "type": "done",
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
    }


def ask_stream_broad(question: str):
    """
    Streaming version using hybrid_retrieve_broad — used on retry.
    Fetches from a larger candidate pool (MMR k=12/fetch_k=40, BM25 k=12,
    MultiQuery k=10 per sub-query, reranker top-15) so the LLM receives
    genuinely different and more extensive context than the first pass.
    Same yield contract as ask_stream.
    """
    query_id = str(uuid.uuid4())

    docs = hybrid_retrieve_broad(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    if not _check_relevance(docs):
        yield {"type": "token", "content": _DONT_KNOW}
        yield {"type": "done", "retrieved_contexts": [d["text"] for d in structured_docs], "query_id": query_id}
        return

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )

    for chunk in llm.stream(final_prompt):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    yield {
        "type": "done",
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
    }