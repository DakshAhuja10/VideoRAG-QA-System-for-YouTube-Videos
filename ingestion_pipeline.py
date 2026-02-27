import os
import re
import tempfile
from types import SimpleNamespace

import pandas as pd
from youtube_meta_data import get_video_info, append_data_to_csv
from transcript_generate import ytt_api
from config import VIDEOS_CSV, TRANSCRIPT_CSV, VectorStoreConfig, EmbeddingConfig


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
            )
    
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
