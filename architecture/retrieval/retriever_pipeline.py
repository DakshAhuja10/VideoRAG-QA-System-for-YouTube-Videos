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

# Observability
from architecture.observability.langfuse_tracing import (
    create_trace, start_span, end_span, log_generation,
    score_trace, timed_span, flush as langfuse_flush, estimate_cost,
)

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


def hybrid_retrieve(query, trace=None):
    """Standard retrieval used for first-pass answers."""
    retrieval_span = start_span(trace, name="retrieval", input={"query": query, "mode": "standard"})

    # Phase 1: Broad Retrieval (Greater k to allow re-ranker to find the best)
    with timed_span(retrieval_span, name="mmr_retrieve", input={"k": RetrievalConfig.MMR_K, "fetch_k": RetrievalConfig.MMR_FETCH_K}) as (s1, t1):
        r1 = mmr_retriever.invoke(query)
        end_span(s1, output={"doc_count": len(r1)})

    with timed_span(retrieval_span, name="multiquery_retrieve", input={"k": RetrievalConfig.MULTIQUERY_K}) as (s2, t2):
        r2 = multi_retriever.invoke(query)
        end_span(s2, output={"doc_count": len(r2)})

    results = [r1, r2]

    if bm25:
        with timed_span(retrieval_span, name="bm25_retrieve", input={"k": RetrievalConfig.BM25_K}) as (s3, t3):
            r3 = bm25.invoke(query)
            end_span(s3, output={"doc_count": len(r3)})
        results.append(r3)

    with timed_span(retrieval_span, name="combine_dedup") as (s4, t4):
        combined = combine_results(*results)
        end_span(s4, output={"before_dedup": sum(len(r) for r in results), "after_dedup": len(combined)})

    # Phase 2: Re-ranking (Cross-Encoder)
    if combined:
        with timed_span(retrieval_span, name="cross_encoder_rerank", input={"num_docs": len(combined), "model": RetrievalConfig.RERANK_MODEL}) as (s5, t5):
            pairs = [[query, doc.page_content] for doc in combined]
            scores = rerank_model.predict(pairs)

            for i, doc in enumerate(combined):
                doc.metadata["rerank_score"] = float(scores[i])

            combined.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

            rerank_output = [
                {"rank": i+1, "score": round(float(scores[idx]), 4), "preview": combined[i].page_content[:120]}
                for i, idx in enumerate(range(min(RetrievalConfig.RERANK_TOP_N, len(combined))))
            ]
            end_span(s5, output={"top_n": RetrievalConfig.RERANK_TOP_N, "top_score": round(float(combined[0].metadata["rerank_score"]), 4), "ranking": rerank_output})

    end_span(retrieval_span, output={"final_doc_count": min(RetrievalConfig.RERANK_TOP_N, len(combined))})
    return combined[:RetrievalConfig.RERANK_TOP_N]


def hybrid_retrieve_broad(query, trace=None):
    """
    Wider retrieval used on retry when the first-pass confidence is low.

    Differences vs hybrid_retrieve:
    - MMR: fetches 40 candidates (vs 20) and returns top 12 (vs 6).
    - BM25: returns 12 docs (vs 6), broadening keyword coverage.
    - Reranker: keeps top 15 after scoring (vs 10).
    """
    retrieval_span = start_span(trace, name="retrieval_broad", input={"query": query, "mode": "broad_retry"})

    with timed_span(retrieval_span, name="mmr_retrieve_broad", input={"k": 12, "fetch_k": 40}) as (s1, _):
        mmr_broad = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 40},
        )
        r1 = mmr_broad.invoke(query)
        end_span(s1, output={"doc_count": len(r1)})

    with timed_span(retrieval_span, name="multiquery_retrieve_broad", input={"k": 10}) as (s2, _):
        mq_broad = vector_store.as_retriever(search_kwargs={"k": 10})
        mq_llm = ChatGoogleGenerativeAI(model=LLMConfig.MULTIQUERY_MODEL)
        multi_broad = MultiQueryRetriever.from_llm(retriever=mq_broad, llm=mq_llm)
        r2 = multi_broad.invoke(query)
        end_span(s2, output={"doc_count": len(r2)})

    results = [r1, r2]

    if bm25:
        with timed_span(retrieval_span, name="bm25_retrieve_broad", input={"k": 12}) as (s3, _):
            original_k = bm25.k
            bm25.k = 12
            r3 = bm25.invoke(query)
            bm25.k = original_k
            end_span(s3, output={"doc_count": len(r3)})
        results.append(r3)

    with timed_span(retrieval_span, name="combine_dedup") as (s4, _):
        combined = combine_results(*results)
        end_span(s4, output={"before_dedup": sum(len(r) for r in results), "after_dedup": len(combined)})

    if combined:
        with timed_span(retrieval_span, name="cross_encoder_rerank", input={"num_docs": len(combined)}) as (s5, _):
            pairs = [[query, doc.page_content] for doc in combined]
            scores = rerank_model.predict(pairs)
            for i, doc in enumerate(combined):
                doc.metadata["rerank_score"] = float(scores[i])
            combined.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

            rerank_output = [
                {"rank": i+1, "score": round(float(combined[i].metadata["rerank_score"]), 4), "preview": combined[i].page_content[:120]}
                for i in range(min(15, len(combined)))
            ]
            end_span(s5, output={"top_n": 15, "top_score": round(float(combined[0].metadata["rerank_score"]), 4), "ranking": rerank_output})

    end_span(retrieval_span, output={"final_doc_count": min(15, len(combined))})
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
    trace = create_trace(name="rag_ask", metadata={"query_id": query_id, "question": question, "mode": "non_streaming"})

    docs = hybrid_retrieve(question, trace=trace)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    if not _check_relevance(docs):
        if trace:
            score_trace(trace, name="relevance_gate", value=0.0, comment="Rejected: top rerank score below threshold")
            langfuse_flush()
        return {
            "answer": _DONT_KNOW,
            "retrieved_contexts": [d["text"] for d in structured_docs],
            "query_id": query_id,
            "trace": trace,
        }

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )
    response = llm.invoke(final_prompt)
    answer_text = response.content

    # Log the LLM generation
    usage_meta = response.response_metadata.get("usage", {}) if hasattr(response, "response_metadata") else {}
    input_tokens = usage_meta.get("prompt_tokens", 0)
    output_tokens = usage_meta.get("completion_tokens", 0)
    log_generation(
        trace, name="answer_llm", model=LLMConfig.ANSWER_MODEL,
        input=final_prompt, output=answer_text,
        usage={"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
        metadata={"cost_usd": estimate_cost(LLMConfig.ANSWER_MODEL, input_tokens, output_tokens)},
    )

    langfuse_flush()
    return {
        "answer": answer_text,
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
        "trace": trace,
    }


def ask_stream(question: str):
    """
    Streaming version — standard first-pass retrieval.
    Yields:
    - {"type": "token", "content": <token>} for each token
    - {"type": "done", "retrieved_contexts": [...], "query_id": ..., "trace": ...} when complete
    """
    import time as _time
    query_id = str(uuid.uuid4())
    trace = create_trace(name="rag_ask_stream", metadata={"query_id": query_id, "question": question, "mode": "streaming"})

    docs = hybrid_retrieve(question, trace=trace)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    if not _check_relevance(docs):
        score_trace(trace, name="relevance_gate", value=0.0, comment="Rejected: top rerank score below threshold")
        langfuse_flush()
        yield {"type": "token", "content": _DONT_KNOW}
        yield {"type": "done", "retrieved_contexts": [d["text"] for d in structured_docs], "query_id": query_id, "trace": trace}
        return

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )

    full_response = ""
    t0 = _time.perf_counter()
    last_chunk_meta = {}
    for chunk in llm.stream(final_prompt):
        if chunk.content:
            full_response += chunk.content
            yield {"type": "token", "content": chunk.content}
        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            last_chunk_meta = chunk.response_metadata

    gen_ms = (_time.perf_counter() - t0) * 1000
    usage_meta = last_chunk_meta.get("usage", {}) or {}
    input_tokens = usage_meta.get("prompt_tokens", 0)
    output_tokens = usage_meta.get("completion_tokens", 0)

    log_generation(
        trace, name="answer_llm", model=LLMConfig.ANSWER_MODEL,
        input=final_prompt, output=full_response,
        usage={"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
        metadata={"duration_ms": round(gen_ms, 1), "cost_usd": estimate_cost(LLMConfig.ANSWER_MODEL, input_tokens, output_tokens)},
    )

    langfuse_flush()
    yield {
        "type": "done",
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
        "trace": trace,
    }


def ask_stream_broad(question: str, trace=None):
    """
    Streaming version using hybrid_retrieve_broad — used on retry.
    Same yield contract as ask_stream.
    """
    import time as _time
    query_id = str(uuid.uuid4())
    # Reuse the trace from the initial ask_stream if provided (retry scenario)
    if trace is None:
        trace = create_trace(name="rag_ask_stream_broad", metadata={"query_id": query_id, "question": question, "mode": "streaming_broad_retry"})

    docs = hybrid_retrieve_broad(question, trace=trace)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    if not _check_relevance(docs):
        score_trace(trace, name="relevance_gate", value=0.0, comment="Rejected on broad retry")
        langfuse_flush()
        yield {"type": "token", "content": _DONT_KNOW}
        yield {"type": "done", "retrieved_contexts": [d["text"] for d in structured_docs], "query_id": query_id, "trace": trace}
        return

    final_prompt = rank_prompt.format(
        context=prompt_context,
        question=question,
    )

    full_response = ""
    t0 = _time.perf_counter()
    last_chunk_meta = {}
    for chunk in llm.stream(final_prompt):
        if chunk.content:
            full_response += chunk.content
            yield {"type": "token", "content": chunk.content}
        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            last_chunk_meta = chunk.response_metadata

    gen_ms = (_time.perf_counter() - t0) * 1000
    usage_meta = last_chunk_meta.get("usage", {}) or {}
    input_tokens = usage_meta.get("prompt_tokens", 0)
    output_tokens = usage_meta.get("completion_tokens", 0)

    log_generation(
        trace, name="answer_llm_retry", model=LLMConfig.ANSWER_MODEL,
        input=final_prompt, output=full_response,
        usage={"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
        metadata={"duration_ms": round(gen_ms, 1), "cost_usd": estimate_cost(LLMConfig.ANSWER_MODEL, input_tokens, output_tokens)},
    )

    langfuse_flush()
    yield {
        "type": "done",
        "retrieved_contexts": [d["text"] for d in structured_docs],
        "query_id": query_id,
        "trace": trace,
    }