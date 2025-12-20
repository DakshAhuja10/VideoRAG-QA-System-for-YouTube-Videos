#use chromadb or any other vector store  which has the ability to persist
#we cannot use FAISS because it does not have the ability of persistence
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm  #to get the status bar while we are creating embedding
import uuid

from doc_loader import Csv_Loader
from dotenv import load_dotenv
load_dotenv()

PERSIST_PATH = "15.LexiChat/chroma_db" # where chroma db is stored
CSV_PATH = "15.LexiChat/video_with_meta_data_and_transcript.csv" #meta data along with transcripts


def build_vector_store():
    loader = Csv_Loader(CSV_PATH)
    docs = loader.load()   #use lazy_load if size of transcripts become greater than 2MB
    print(f"Loaded {len(docs)} transcript documents.")
    print("Loading completed.")

    
    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = Chroma(
        embedding_function=embedder,
        persist_directory=PERSIST_PATH,
        collection_name="lexi_transcripts"
    )

    collection = vector_store._collection

    #a function to get the details if a hash is already present in the collection
    # and returns a dictionary with 4 keys and value pair namely ids,metadata,document and embeddings
    def find_existing_by_hash(text_hash):
        return collection.get(
            where={"text_hash": text_hash},
            include=["metadatas", "documents", "embeddings"]
    )

    #here everytime we create new embedding we see how many new embeddings do we need to had
    #how many we need to skip and how many can be resued.
    added = 0
    skipped = 0
    reused = 0

    print("Processing documents...\n")

    for doc in tqdm(docs):
        text = doc.page_content
        metadata = doc.metadata
        text_hash = metadata["text_hash"]

        existing = find_existing_by_hash(text_hash)

        #Case 1:Exact duplicate - if the hash is already present in the db then it will return id
        #then we compare metadata 
        #if metadata also matches with our current documents metadata then it is a duplicate 
        #so no need to compute embedding and store in db
        if existing["ids"]:
            existing_meta = existing["metadatas"][0]

            if metadata == existing_meta:
                skipped += 1
                continue

            #Case 2 hash is present but metadata is not same so maybe text is same but start and end
            #duration (or any other meta data) is not same so we use the already created embeddings # from the existing hash and 
            #create a new id to store the collection in chroma db and then add it to the Chroma DB
            #while reusing the precomputed embeddings
            existing_embedding = existing["embeddings"][0]
            new_id = str(uuid.uuid4())

            collection.add(
                ids=[new_id],
                documents=[text],
                embeddings=[existing_embedding],
                metadatas=[metadata]
            )
            reused += 1
            continue

        #case 3: if the hash does not exist so add the hash into chroma db
        vector_store.add_documents([doc])
        added += 1

    print("\n------------------ SUMMARY ------------------")
    print(f"New embeddings created (Case C): {added}")
    print(f"Reused embeddings (Case A):       {reused}")
    print(f"Duplicates skipped (Case B):      {skipped}")
    print("----------------------------------------------")
    print(f"Chroma DB updated at: {PERSIST_PATH}")

if __name__ == "__main__":
    build_vector_store()
