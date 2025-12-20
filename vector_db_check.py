# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# load_dotenv()
# embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# vs = Chroma(
#     persist_directory="15.LexiChat/chroma_db",
#     collection_name="lexi_transcripts",
#     embedding_function=embedder,
# )

# print(vs._collection.count())  # should print 5008

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

persist_path = "15.LexiChat/chroma_db"

vs = Chroma(
    persist_directory=persist_path,
    collection_name="lexi_transcripts",
    embedding_function=embedder,
)

# Access raw Chroma collection
collection = vs._collection

# Fetch all IDs
results = collection.get(include=["metadatas", "documents"])

all_ids = results["ids"]

print(len(all_ids))
print(all_ids[:10])  # show first 10 IDs

