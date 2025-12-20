from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


# --------------------------------------
# 1. Load Embedding Model
# --------------------------------------
embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --------------------------------------
# 2. Load Chroma Vector Store
# --------------------------------------
persist_path = "15.LexiChat/chroma_db"

vector_store = Chroma(
    collection_name="lexi_transcripts",
    persist_directory=persist_path,
    embedding_function=embedder,
)


# --------------------------------------
# 3. Create an MMR Retriever (Top-k = 2)
# --------------------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10},    # fetch 10, rerank to top-2
)


# --------------------------------------
# 4. LLM Model
# --------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)


# --------------------------------------
# 5. Prompt Template (STRICT CONTEXT ENFORCEMENT)
# --------------------------------------
prompt = PromptTemplate.from_template("""

You are a Question Answering system over Lex Fridman's YouTube transcripts.

STRICT RULES YOU MUST FOLLOW:
1. You MUST answer ONLY using the provided context.  
2. If the answer is NOT present in the context, respond:
   "I don’t know. The provided transcripts do not contain information to answer this."
3. ALWAYS provide clickable timestamps for every referenced snippet.
4. Format timestamps as:  
   [MM:SS](URL)

--------------------------------------
CONTEXT:
{context}
--------------------------------------

QUESTION:
{question}

Provide the final answer below (with clickable timestamps):
""")



# --------------------------------------
# 6. Build the RAG pipeline
# --------------------------------------
def format_docs(docs):
    """Format retrieved docs into readable text + clickable timestamps."""
    formatted = ""
    for d in docs:
        ts = int(d.metadata["start"])
        url = d.metadata["citation_url"]
        text = d.page_content
        formatted += f"- ({ts}s) [{url}] → {text}\n"
    return formatted


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# --------------------------------------
# 7. Query Function
# --------------------------------------
def ask(question: str):
    print("\nQUESTION:", question)
    answer = rag_chain.invoke(question)
    print("\nANSWER:\n", answer.content)
    return answer.content


# --------------------------------------
# TEST
# --------------------------------------
if __name__ == "__main__":
    ask("What is deep learning??")
