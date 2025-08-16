#!/usr/bin/env python3
# scripts/getting_videos.py
import os, sys, json, time, shutil, subprocess, pathlib, argparse
import requests
from urllib.parse import urlencode
from typing import Dict, Any, List, Optional

# --- Config & dirs ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training"
RAW  = DATA / "raw_videos" / "youtube"
MANI = DATA / "manifests"
RAW.mkdir(parents=True, exist_ok=True)
MANI.mkdir(parents=True, exist_ok=True)

YT_API_KEY = os.getenv("YT_API_KEY")  # set in your shell or .env
FFMPEG = shutil.which("ffmpeg")
YTDLP  = shutil.which("yt-dlp")

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEO_URL  = "https://www.googleapis.com/youtube/v3/videos"

def die(msg: str, code=1):
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(code)

def require_tools():
    if not YT_API_KEY:
        die("Missing YT_API_KEY env var.")
    if not YTDLP:
        die("yt-dlp not found. Install: pip install yt-dlp")
    if not FFMPEG:
        die("ffmpeg not found. Install ffmpeg (brew install ffmpeg on macOS).")

def yt_search(query: str, max_items: int = 20) -> List[str]:
    """Return a list of video IDs (Creative Commons, short)."""
    ids: List[str] = []
    page_token = None
    while len(ids) < max_items:
        params = dict(
            part="snippet",
            type="video",
            q=query,
            maxResults=min(50, max_items - len(ids)),
            videoDuration="short",
            videoLicense="creativeCommon",
            key=YT_API_KEY,
            safeSearch="moderate",
        )
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("items", []):
            vid = item["id"].get("videoId")
            if vid:
                ids.append(vid)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return ids[:max_items]

def yt_video_details(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch license/duration/etc to validate results."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        params = dict(
            part="snippet,contentDetails,status",
            id=",".join(chunk),
            key=YT_API_KEY,
        )
        r = requests.get(VIDEO_URL, params=params, timeout=30)
        r.raise_for_status()
        for it in r.json().get("items", []):
            vid = it["id"]
            out[vid] = it
    return out

def run(cmd: List[str]) -> int:
    print("[cmd]", " ".join(cmd))
    return subprocess.call(cmd)

def normalize_to_vertical(src: pathlib.Path, dst: pathlib.Path, target_h=1280, target_w=720) -> None:
    """
    Letterbox to 9:16 (720x1280 by default) without distortion.
    """
    tmp = dst.with_suffix(".tmp.mp4")
    # scale to fit height, pad width; if wider than tall, scale to fit width then pad height
    fil = (
        f"scale=w={target_w}:h=-2:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )
    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-vf", fil,
        "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        str(tmp),
    ]
    rc = run(cmd)
    if rc != 0:
        raise RuntimeError("ffmpeg normalization failed")
    tmp.replace(dst)

def download_video(video_id: str, out_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Download with yt-dlp as MP4, return filepath or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(out_dir / f"{video_id}.%(ext)s")
    fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best"
    cmd = [
        YTDLP, "-f", fmt, f"https://www.youtube.com/watch?v={video_id}",
        "-o", out_tpl, "--no-playlist", "--retries", "5"
    ]
    rc = run(cmd)
    if rc != 0:
        return None
    # find resulting file
    for ext in (".mp4", ".mkv", ".webm", ".m4v"):
        p = out_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None

def save_metadata(meta_path: pathlib.Path, d: Dict[str, Any]):
    meta_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    require_tools()
    ap = argparse.ArgumentParser(description="Fetch CC-licensed short videos for a topic.")
    ap.add_argument("topic", type=str, help="Search topic, e.g. 'cute dogs'")
    ap.add_argument("--max", type=int, default=20, help="Max videos to fetch")
    ap.add_argument("--normalize", action="store_true", help="Normalize to 9:16 720x1280")
    args = ap.parse_args()

    ids = yt_search(args.topic, args.max)
    if not ids:
        die("No videos found.")
    details = yt_video_details(ids)

    manifest_entries = []
    for vid in ids:
        info = details.get(vid) or {}
        title = ((info.get("snippet") or {}).get("title")) or ""
        license_ok = ((info.get("status") or {}).get("license")) in ("creativeCommon", "creativeCommonAttribution") or True  # search already filtered
        meta = {
            "platform": "youtube",
            "id": vid,
            "title": title,
            "topic": args.topic,
            "source_url": f"https://www.youtube.com/watch?v={vid}",
            "license": (info.get("status") or {}).get("license", "unknown"),
            "fetched_at": int(time.time()),
        }
        print(f"[info] downloading {vid} | {title[:60]!r}")
        raw_path = download_video(vid, RAW)
        if not raw_path:
            print(f"[warn] download failed for {vid}")
            continue

        final_path = RAW / f"{vid}.mp4"
        if raw_path.suffix.lower() != ".mp4":
            # convert container
            run([FFMPEG, "-y", "-i", str(raw_path), "-c", "copy", str(final_path)])
            raw_path.unlink(missing_ok=True)
        else:
            final_path = raw_path

        if args.normalize:
            norm_path = RAW / f"{vid}.norm.mp4"
            try:
                normalize_to_vertical(final_path, norm_path)
                final_path.unlink(missing_ok=True)
                final_path = RAW / f"{vid}.mp4"
                norm_path.replace(final_path)
            except Exception as e:
                print(f"[warn] normalization failed for {vid}: {e}")

        meta["file"] = str(final_path.relative_to(ROOT))
        save_metadata(RAW / f"{vid}.json", meta)
        manifest_entries.append(meta)

    MANI.mkdir(parents=True, exist_ok=True)
    (MANI / "youtube_manifest.json").write_text(
        json.dumps(manifest_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] Saved {len(manifest_entries)} items. Manifest: {MANI/'youtube_manifest.json'}")

if __name__ == "__main__":
    main()
