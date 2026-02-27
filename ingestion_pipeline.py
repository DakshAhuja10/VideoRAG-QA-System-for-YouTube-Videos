import os
import pandas as pd
from youtube_meta_data import get_video_info, append_data_to_csv
from transcript_generate import ytt_api
from config import VIDEOS_CSV, TRANSCRIPT_CSV, VectorStoreConfig, EmbeddingConfig

def ingest_video(video_url, progress_callback=None):
    """
    Unified pipeline to ingest a single video URL:
    1. Metadata extraction
    2. Transcript generation
    3. Vector store indexing
    """
    # 1. Fetch Video Info
    video_info = get_video_info(video_url)
    if not video_info:
        return False, "Failed to fetch video metadata. Please check the URL."
    
    video_id = video_info["video_id"]
    
    # Check if already processed (exists in transcript CSV)
    already_indexed = False
    if os.path.exists(TRANSCRIPT_CSV):
        df_t = pd.read_csv(TRANSCRIPT_CSV)
        if video_id in df_t["video_id"].astype(str).values:
            already_indexed = True
    
    if already_indexed:
        if progress_callback: progress_callback(100, "Video is already in the knowledge base.")
        return True, f"'{video_info['title']}' is already processed and ready for questions."

    if progress_callback: progress_callback(10, "Extracting metadata...")
    append_data_to_csv(video_info, VIDEOS_CSV)

    if progress_callback: progress_callback(20, "Fetching transcript...")

    # 2. Fetch Transcript
    try:
        transcript = ytt_api.fetch(video_id, languages=["en"])
    except Exception as e:
        return False, f"Error fetching transcript: {str(e)}"
    
    # Check if transcript already exists
    processed = False
    if os.path.exists(TRANSCRIPT_CSV):
        df_t = pd.read_csv(TRANSCRIPT_CSV)
        if video_id in df_t["video_id"].astype(str).values:
            processed = True
            if progress_callback: progress_callback(40, "Transcript already exists.")
    
    if not processed:
        rows = []
        for snippet in transcript:
            rows.append({
                "video_id": video_id,
                "title": video_info["title"],
                "length": video_info["length"],
                "publish_date": video_info["publish_date"],
                "views": video_info["views"],
                "url": video_url,
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            })
        
        df_new = pd.DataFrame(rows)
        df_new.to_csv(TRANSCRIPT_CSV, mode='a', header=not os.path.exists(TRANSCRIPT_CSV), index=False)
        if progress_callback: progress_callback(60, f"Saved {len(df_new)} transcript segments.")

    if progress_callback: progress_callback(70, "Initializing vector store...")

    # 3. Vector Store Indexing
    from vector_store_chroma import build_vector_store
    
    if progress_callback: progress_callback(80, "Generating embeddings and updating vector store...")
    
    try:
        build_vector_store(progress_callback=progress_callback)
        if progress_callback: progress_callback(100, "Ingestion and embedding complete!")
        return True, f"Successfully ingested video: {video_info['title']} and updated vector store."
    except Exception as e:
        return False, f"Error during embedding generation: {str(e)}"

if __name__ == "__main__":
    # Test
    url = "https://www.youtube.com/watch?v=EUowNpYL120"
    success, msg = ingest_video(url, progress_callback=lambda p, m: print(f"[{p}%] {m}"))
    print(msg)
