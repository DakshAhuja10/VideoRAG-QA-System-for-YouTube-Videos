from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm
from doc_loader import Csv_Loader
import uuid
from dotenv import load_dotenv

load_dotenv()

PERSIST_PATH = "15.LexiChat/chroma_db"
CSV_PATH = "15.LexiChat/video_with_meta_data_and_transcript.csv"


def build_vector_store():
    # Step 1: Load documents
    loader = Csv_Loader(CSV_PATH)
    docs = loader.load()

    print(f"Loaded {len(docs)} transcript documents.")
    print("Loading completed.")

    # Step 2: Google Embeddings
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    # Step 3: Chroma DB
    vector_store = Chroma(
        embedding_function=embedder,
        persist_directory=PERSIST_PATH,
        collection_name="lexi_transcripts"
    )

    collection = vector_store._collection

    def find_existing_by_hash(text_hash):
        return collection.get(
            where={"text_hash": text_hash},
            include=["metadatas", "documents", "embeddings"]
        )

    added = 0
    skipped = 0
    reused = 0

    print("Processing documents...\n")

    for doc in tqdm(docs):
        text = doc.page_content
        metadata = doc.metadata
        text_hash = metadata["text_hash"]

        existing = find_existing_by_hash(text_hash)

        # CASE B: Exact duplicate
        if existing["ids"]:
            existing_meta = existing["metadatas"][0]

            if metadata == existing_meta:
                skipped += 1
                continue

            # CASE A: Same text, different metadata
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

        # CASE C: New text
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
