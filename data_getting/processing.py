# process_trending.py
import json, csv, sys, os
from typing import Dict, Any, List

# base directories
CSV_DIR = "csvs"
JSON_DIR = "jsons"

# ensure directories exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

RAW_IN    = os.environ.get("RAW_IN", "raw_trending.jsonl")         # input from scraper
CSV_OUT   = os.path.join(CSV_DIR, "trending.csv")                  # tidy table
JSONL_OUT = os.path.join(JSON_DIR, "trending_clean.jsonl")         # flattened jsonl

def pick_best_play_url(vdict: Dict[str, Any]) -> str:
    """Pick a decent playable URL with reasonable fallbacks."""
    video = vdict.get("video") or {}

    play_addr = video.get("playAddr")
    if isinstance(play_addr, str) and play_addr:
        return play_addr

    best = None
    for bi in (video.get("bitrateInfo") or []):
        try:
            br = int(bi.get("Bitrate") or 0)
            urls = (((bi.get("PlayAddr") or {}).get("UrlList")) or [])
            if urls:
                if best is None or br > best[0]:
                    best = (br, urls[0])
        except Exception:
            pass
    if best:
        return best[1]

    return vdict.get("downloadAddr") or vdict.get("playAddr") or ""

def flatten_video(vdict: Dict[str, Any]) -> Dict[str, Any]:
    author = vdict.get("author") or {}
    stats  = vdict.get("stats")  or {}
    video  = vdict.get("video")  or {}
    contents = vdict.get("contents") or []

    desc = vdict.get("desc") or ""
    if not desc and contents and isinstance(contents[0], dict):
        desc = contents[0].get("desc") or ""

    return {
        "id": vdict.get("id") or video.get("id") or "",
        "createTime": vdict.get("createTime", ""),
        "author_id": author.get("id", ""),
        "author_uniqueId": author.get("uniqueId", ""),
        "author_nickname": author.get("nickname", ""),
        "desc": desc,
        "duration": video.get("duration", vdict.get("duration", "")),
        "playCount": stats.get("playCount", ""),
        "diggCount": stats.get("diggCount", ""),
        "shareCount": stats.get("shareCount", ""),
        "commentCount": stats.get("commentCount", ""),
        "cover": video.get("cover", vdict.get("cover", "")),
        "playUrl": pick_best_play_url(vdict),
        "downloadAddr": vdict.get("downloadAddr", ""),
        "music_title": (vdict.get("music") or {}).get("title", ""),
        "music_author": (vdict.get("music") or {}).get("authorName", ""),
        "isAd": vdict.get("isAd", False),
        "language": vdict.get("textLanguage") or "",
    }

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    if not os.path.exists(RAW_IN):
        raise FileNotFoundError(f"Input file not found: {RAW_IN}. Run scrape_trending.py first.")
    raw = load_jsonl(RAW_IN)
    flat = [flatten_video(v) for v in raw]
    write_jsonl(JSONL_OUT, flat)
    write_csv(CSV_OUT, flat)
    print(f"Processed {len(flat)} records → {CSV_OUT}, {JSONL_OUT}")

if __name__ == "__main__":
    main()
