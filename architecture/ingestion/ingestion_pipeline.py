import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os
import re
import tempfile
from types import SimpleNamespace

import pandas as pd
from architecture.ingestion.youtube_meta_data import get_video_info, append_data_to_csv
from architecture.ingestion.transcript_generate import ytt_api
from config import VIDEOS_CSV, TRANSCRIPT_CSV, URLS_FILE, VectorStoreConfig, EmbeddingConfig


# ── yt-dlp transcript fallback ────────────────────────────────────────────────
# youtube-transcript-api is blocked by YouTube on most cloud provider IPs
# (Streamlit Cloud, AWS, GCP, Azure, etc.).  yt-dlp downloads the subtitle
# file directly using a browser-like request, which avoids this block.

def _parse_vtt(content: str) -> list:
    """
    Parse a WebVTT subtitle file downloaded by yt-dlp into a list of
    SimpleNamespace(text, start, duration) objects – the same contract as
    youtube-transcript-api snippets.

    Handles yt-dlp auto-caption quirks:
    - Strips all HTML tags  (<c>, </c>, timestamp tags, etc.)
    - Deduplicates repeated context lines (auto-captions echo the previous
      line on each new cue)
    - Skips empty cues
    """
    # Remove header block
    content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    content = re.sub(r'NOTE\s.*?\n\n', '', content, flags=re.DOTALL | re.MULTILINE)

    blocks = re.split(r'\n{2,}', content.strip())
    segments = []
    seen_texts: set = set()

    for block in blocks:
        lines = block.strip().splitlines()
        timing_line = None
        text_lines = []

        for line in lines:
            if '-->' in line:
                timing_line = line
            elif timing_line and line.strip():
                text_lines.append(line)

        if not timing_line or not text_lines:
            continue

        m = re.match(
            r'(\d+):(\d+):(\d+\.\d+)\s*-->\s*(\d+):(\d+):(\d+\.\d+)',
            timing_line,
        )
        if not m:
            continue

        h1, m1, s1, h2, m2, s2 = m.groups()
        start    = int(h1) * 3600 + int(m1) * 60 + float(s1)
        end      = int(h2) * 3600 + int(m2) * 60 + float(s2)
        duration = max(end - start, 0.0)

        raw = " ".join(text_lines)
        text = re.sub(r'<[^>]+>', '', raw)      # strip all HTML/XML tags
        text = re.sub(r'\s+', ' ', text).strip()

        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        segments.append(SimpleNamespace(text=text, start=start, duration=duration))

    return segments


def _fetch_transcript_ytdlp(video_id: str, video_url: str) -> list:
    """
    Download auto-generated or manual English subtitles using yt-dlp and
    return them in the same format as youtube-transcript-api.

    yt-dlp spoofs a real browser User-Agent and uses different request
    patterns, so it succeeds on cloud IPs where youtube-transcript-api fails.
    """
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,       # manual captions
            "writeautomaticsub": True,    # auto-generated captions
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        vtt_file = next(
            (os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".vtt")),
            None,
        )

        if not vtt_file:
            raise FileNotFoundError(
                "yt-dlp could not download subtitles. "
                "The video may have no English captions, or yt-dlp is also being blocked."
            )

        with open(vtt_file, "r", encoding="utf-8") as fh:
            content = fh.read()

    segments = _parse_vtt(content)
    if not segments:
        raise ValueError("Subtitle file was downloaded but contained no parseable text.")
    return segments


def ingest_video(video_url, progress_callback=None, **kwargs):
    """
    Unified pipeline to ingest a single video URL:
    1. Metadata extraction
    2. Transcript generation
    3. Vector store indexing

    Returns: (success: bool, message: str, video_id: str | None, title: str | None)

    Optional kwargs:
        vector_store: An existing Chroma instance to reuse (avoids dual-connection
                      "Error finding id" when called from Streamlit).
    """
    # 1. Fetch Video Info
    video_info = get_video_info(video_url)
    if not video_info:
        return False, "Failed to fetch video metadata. Please check the URL.", None, None
    
    video_id = video_info["video_id"]
    
    # Check if already processed (exists in transcript CSV)
    already_indexed = False
    if os.path.exists(TRANSCRIPT_CSV):
        df_t = pd.read_csv(TRANSCRIPT_CSV)
        if video_id in df_t["video_id"].astype(str).values:
            already_indexed = True
    
    if already_indexed:
        # Ensure URL is tracked even if video was ingested before urls.txt tracking existed
        if os.path.exists(URLS_FILE):
            with open(URLS_FILE, "r") as f:
                existing_urls = f.read()
        else:
            existing_urls = ""
        if video_url.strip() not in existing_urls:
            with open(URLS_FILE, "a") as f:
                f.write(video_url.strip() + "\n")
        if progress_callback: progress_callback(100, "Video is already in the knowledge base.")
        return True, f"'{video_info['title']}' is already processed and ready for questions.", video_id, video_info['title']

    if progress_callback: progress_callback(10, "Extracting metadata...")
    append_data_to_csv(video_info, VIDEOS_CSV)

    if progress_callback: progress_callback(20, "Fetching transcript...")

    # 2. Fetch Transcript
    # Try youtube-transcript-api first (fast, clean output).
    # On Streamlit Cloud and other cloud IPs, YouTube blocks its requests.
    # yt-dlp uses browser-like requests and succeeds in most cases where
    # youtube-transcript-api is blocked.
    transcript = None
    try:
        transcript = ytt_api.fetch(video_id, languages=["en"])
        if progress_callback: progress_callback(35, "Transcript fetched via youtube-transcript-api.")
    except Exception as primary_err:
        if progress_callback:
            progress_callback(25, "youtube-transcript-api blocked — trying yt-dlp fallback...")
        try:
            transcript = _fetch_transcript_ytdlp(video_id, video_url)
            if progress_callback:
                progress_callback(35, f"Transcript fetched via yt-dlp ({len(transcript)} segments).")
        except Exception as fallback_err:
            return False, (
                f"Could not fetch transcript using either method.\n\n"
                f"**Primary (youtube-transcript-api):** {primary_err}\n\n"
                f"**Fallback (yt-dlp):** {fallback_err}\n\n"
                "YouTube blocks most cloud provider IPs from fetching transcripts. "
                "This feature works reliably when running the app locally."
            ), video_id, video_info['title']
    
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
            text = snippet.text if isinstance(snippet.text, str) else ""
            if not text.strip():
                continue
            rows.append({
                "video_id": video_id,
                "title": video_info["title"],
                "length": video_info["length"],
                "publish_date": video_info["publish_date"],
                "views": video_info["views"],
                "url": video_url,
                "text": text,
                "start": snippet.start,
                "duration": snippet.duration
            })
        
        df_new = pd.DataFrame(rows)
        df_new.to_csv(TRANSCRIPT_CSV, mode='a', header=not os.path.exists(TRANSCRIPT_CSV), index=False)
        if progress_callback: progress_callback(60, f"Saved {len(df_new)} transcript segments.")
    else:
        rows = []

    if not rows:
        # Transcript was already saved and no new rows to embed — skip vector store
        with open(URLS_FILE, "a") as f:
            f.write(video_url.strip() + "\n")
        if progress_callback: progress_callback(100, "Ingestion complete (transcript already embedded).")
        return True, f"Successfully ingested video: {video_info['title']}.", video_id, video_info['title']

    if progress_callback: progress_callback(70, "Initializing vector store...")

    # 3. Vector Store Indexing — add only the new video's documents.
    # Build Document objects from the rows we just wrote to CSV so we don't
    # reload all 11k+ docs.  When an existing_vector_store is provided (from
    # Streamlit's cached Chroma instance) we reuse it to avoid opening a
    # second SQLite connection (which causes "Error finding id").
    from hashlib import sha256
    from langchain_core.documents import Document
    from architecture.ingestion.vector_store_chroma import add_new_docs
    
    if progress_callback: progress_callback(80, "Generating embeddings and updating vector store...")

    new_docs = []
    for r in rows:
        text_hash = sha256(r["text"].encode("utf-8")).hexdigest()
        new_docs.append(Document(
            page_content=r["text"],
            metadata={
                "video_id": r["video_id"],
                "title": r["title"],
                "start": r["start"],
                "duration": r["duration"],
                "url": r["url"],
                "publish_date": r.get("publish_date"),
                "views": r.get("views"),
                "length": r.get("length"),
                "citation_url": f"{r['url']}&t={int(r['start'])}s",
                "text_hash": text_hash,
            },
        ))
    
    try:
        add_new_docs(new_docs, existing_vector_store=kwargs.get("vector_store"), progress_callback=progress_callback)
        # Record the URL in urls.txt
        with open(URLS_FILE, "a") as f:
            f.write(video_url.strip() + "\n")
        if progress_callback: progress_callback(100, "Ingestion and embedding complete!")
        return True, f"Successfully ingested video: {video_info['title']} and updated vector store.", video_id, video_info['title']
    except Exception as e:
        return False, f"Error during embedding generation: {str(e)}", video_id, video_info['title']

if __name__ == "__main__":
    # Test
    url = "https://www.youtube.com/watch?v=EUowNpYL120"
    success, msg, vid, title = ingest_video(url, progress_callback=lambda p, m: print(f"[{p}%] {m}"))
    print(msg)
