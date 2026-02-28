# Check whether collections are present in ChromaDB
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from config import VectorStoreConfig, EmbeddingConfig
load_dotenv()


persist_path = str(VectorStoreConfig.PERSIST_DIRECTORY)

embedder = HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)

vs = Chroma(
    persist_directory=persist_path,
    collection_name=VectorStoreConfig.COLLECTION_NAME,
    embedding_function=embedder,
)


collection = vs._collection

# Fetch all IDs
results = collection.get(include=["metadatas", "documents"])

all_ids = results["ids"]

print(len(all_ids))  #total no. of ids present
print(all_ids[:10])  

