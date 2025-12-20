#this file is just to check whether the collections are present in chromadb 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


persist_path = "15.LexiChat/chroma_db" #your chromadb path

embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vs = Chroma(
    persist_directory=persist_path,
    collection_name="lexi_transcripts",
    embedding_function=embedder,
)


collection = vs._collection

# Fetch all IDs
results = collection.get(include=["metadatas", "documents"])

all_ids = results["ids"]

print(len(all_ids))  #total no. of ids present
print(all_ids[:10])  

