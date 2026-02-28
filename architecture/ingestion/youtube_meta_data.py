#We use python library for youtube videos yt_dlp to extract meta deta about a youtube video 
from yt_dlp import YoutubeDL
import csv


#this functions takes a playlist url and file path then writes all the urls inside the playlist into the file at the given path
def dump_urls_to_file(playlist_url: str, file_path):
    ydl_opts = {
        "quiet": True,#to update the logs only when there is an error
        "skip_download": True,#to not download the video files
        "extract_flat": True,#returns a dictionary containing a top level meta data
        "extractor_args": {"youtube": {"player_client": "default"}},
        "no_warnings": True,#ignore warning 
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        
    entries = info.get("entries") or []
    
    # Read existing URLs to prevent duplicates and overwriting
    import os
    existing_urls = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_urls.add(line.strip())

    #opens the file and appends new urls to it
    with open(file_path, "a", encoding="utf-8") as f:
        for e in entries:
            vid = e.get("id") or e.get("url")
            if vid:
                url = f"https://www.youtube.com/watch?v={vid}"
                if url not in existing_urls:
                    f.write(f"{url}\n")
                    existing_urls.add(url)


#reads the csv file and returns all the unique urls
def read_csv_file(file_path):
    set_urls = set()
    import os
    if not os.path.exists(file_path):
        return set_urls
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'url' in row:
                set_urls.add(row['url'])
    return set_urls

#this function is used to get all the metadeta related to a youtube video using its video id
def get_video_info(video_url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extractor_args": {"youtube": {"player_client": "default"}},
        "no_warnings": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        video_id = info.get("id")
        return {
            "title": info.get("title"),
            "length": info.get("duration"),
            "video_id": video_id,
            "publish_date": info.get("upload_date"),
            "views": info.get("view_count"),
            "url": "https://www.youtube.com/watch?v=" + video_id if video_id else video_url
        }
    except Exception as e:
        print(f"An error occured with url:{video_url},error:{str(e)}")
        return None


#this function appends the metadata to the csv file
def append_data_to_csv(video_info, file_path):
    import os
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "length", "video_id", "publish_date", "views", "url"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(video_info)


if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/playlist?list=PLmWa9ZZLlCjvTdH_rk9NwZtsd2h9ORZYt"
    urls_file_path = "15.LexiChat/architecture/data/urls.txt"
    csv_file_path = "15.LexiChat/architecture/data/videos.csv"

    # Step 1: Dump URLs from playlist to a file
    dump_urls_to_file(playlist_url, urls_file_path)

    set_urls = read_csv_file(csv_file_path)

    with open(urls_file_path, "r") as f:
        urls = f.readlines()

    for url in urls:
        url = url.strip()
        if url not in set_urls:
            print(f"Processing Urls {url}")
            video_info = get_video_info(url)
            if video_info:
                append_data_to_csv(video_info, csv_file_path)
