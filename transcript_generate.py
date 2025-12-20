import pandas as pd
from pathlib import Path
#this library is used to get the transcript of a video 
from youtube_transcript_api import YouTubeTranscriptApi

INPUT_CSV = Path("15.LexiChat/videos.csv")
OUTPUT_CSV = Path("15.LexiChat/video_with_meta_data_and_transcript.csv")

ytt_api = YouTubeTranscriptApi()
df_videos = pd.read_csv(INPUT_CSV)


df_videos = df_videos.dropna(subset=["video_id"])
df_videos["video_id"] = df_videos["video_id"].astype(str)
df_videos = df_videos.drop_duplicates(subset=["video_id"])

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

    if df_existing is not None:
        final_df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        final_df = df_new

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_new)} new transcript rows.")
else:
    print("No new transcripts to add.")

print("Done!")
