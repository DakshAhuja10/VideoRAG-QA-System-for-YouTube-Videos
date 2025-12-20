from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# EMBEDDINGS + VECTOR STORE
# ============================================================

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = Chroma(
    collection_name="lexi_transcripts",
    persist_directory="15.LexiChat/chroma_db",
    embedding_function=embedder,
)

# ============================================================
# RETRIEVERS
# ============================================================

# --- MMR Retriever ---
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)

# --- Multi-Query Retriever (Gemini) ---
mq_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
    llm=mq_llm,
)

# --- BM25 Retriever ---
docs = vector_store._collection.get(include=["documents", "metadatas"])
all_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(docs["documents"], docs["metadatas"])
]

bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 6

# ============================================================
# HYBRID RETRIEVAL
# ============================================================

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
    r3 = bm25.invoke(query)

    combined = combine_results(r1, r2, r3)
    return combined[:10]  # slightly higher recall for ranking

# ============================================================
# STRUCTURED CONTEXT
# ============================================================

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


def docs_to_prompt_context(structured_docs):
    blocks = []
    for d in structured_docs:
        mm = d["timestamp"] // 60
        ss = d["timestamp"] % 60
        ts = f"{mm:02d}:{ss:02d}"

        blocks.append(
            f"""
Chunk {d['id']}:
Text: {d['text']}
Timestamp: [{ts}]({d['url']})
"""
        )
    return "\n".join(blocks)

# ============================================================
# PROMPT (TOP-3 ANSWERS)
# ============================================================

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


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# ============================================================
# PUBLIC API (STREAMLIT + EVALUATION)
# ============================================================

def ask(question: str):
    """
    Returns:
    {
        "answer": str,
        "retrieved_contexts": list[str]
    }
    """

    docs = hybrid_retrieve(question)
    structured_docs = format_docs_structured(docs)
    prompt_context = docs_to_prompt_context(structured_docs)

    response = llm.invoke(
        rank_prompt.format(
            context=prompt_context,
            question=question
        )
    ).content

    return {
        "answer": response,
        "retrieved_contexts": [d["text"] for d in structured_docs],
    }
print(ask("what is geopolitics")['answer'])