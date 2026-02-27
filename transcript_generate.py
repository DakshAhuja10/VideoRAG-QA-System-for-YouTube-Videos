import pandas as pd
from pathlib import Path
from config import VIDEOS_CSV, TRANSCRIPT_CSV
from youtube_transcript_api import YouTubeTranscriptApi

INPUT_CSV = VIDEOS_CSV
OUTPUT_CSV = TRANSCRIPT_CSV

ytt_api = YouTubeTranscriptApi()

#here i am reading the csv file at the path and then dropping the rows where video id is null or duplicated rows 
df_videos = pd.read_csv(INPUT_CSV)
df_videos = df_videos.dropna(subset=["video_id"])
df_videos["video_id"] = df_videos["video_id"].astype(str)
df_videos = df_videos.drop_duplicates(subset=["video_id"])

#the videos which we have already processed so no need to extract their transcripts again
if OUTPUT_CSV.exists():
    df_existing = pd.read_csv(OUTPUT_CSV)
    processed_video_ids = set(df_existing["video_id"].astype(str).unique())
    print(f"Found {len(processed_video_ids)} videos already processed.")
else:
    df_existing = None
    processed_video_ids = set()
    print("No existing transcript file found. Processing all videos.")

rows = []

for _, row in df_videos.iterrows():
    video_id = row["video_id"]

    if not isinstance(video_id, str) or not video_id.strip():
        print("Skipping row with invalid video_id")
        continue

    if video_id in processed_video_ids:
        print(f"Skipping {video_id} (already processed)")
        continue

    title = row.get("Title")
    length = row.get("Length")
    publish_date = row.get("publish_date")
    views = row.get("views")
    url = row.get("url")

    print(f"Fetching transcript for {video_id} ...")

    try:
        transcript = ytt_api.fetch(video_id, languages=["en"])
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        continue
# transcript we fetch is of the form transcript line ,start time , end time 
# so here we merge the meta data we already have with the transcript,start time and end time
    for snippet in transcript:
        rows.append({
            "video_id": video_id,
            "title": title,
            "length": length,
            "publish_date": publish_date,
            "views": views,
            "url": url,
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration
        })

if rows:
    df_new = pd.DataFrame(rows)
    try:
        df_new.to_csv(OUTPUT_CSV, mode='a', header=not OUTPUT_CSV.exists(), index=False)
        print(f"Saved {len(df_new)} new transcript rows.")
    except PermissionError:
        fallback_file = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_fallback.csv")
        print(f"Warning: {OUTPUT_CSV} is locked or read-only. Permission denied.")
        df_new.to_csv(fallback_file, mode='a', header=not fallback_file.exists(), index=False)
        print(f"Saved {len(df_new)} new transcript rows to fallback file: {fallback_file}")
        print("Please resolve the lock on the original file and merge the contents manually.")
else:
    print("No new transcripts to add.")

