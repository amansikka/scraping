# scraping.py
import asyncio, os, json, random
from TikTokApi import TikTokApi
from TikTokApi.exceptions import EmptyResponseException

MS_TOKEN = os.environ.get("ms_token")
if not MS_TOKEN:
    raise RuntimeError("ms_token env var is missing. Export it first.")

BROWSER  = os.getenv("TIKTOK_BROWSER", "webkit")   # webkit|chromium|firefox
HEADLESS = os.getenv("HEADLESS", "false").lower() in ("1","true","yes")
RAW_OUT  = os.getenv("RAW_OUT", "jsons/raw_trending.jsonl")

async def trending_videos(count: int = 25):
    os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)

    attempts = 0
    while True:
        try:
            async with TikTokApi() as api:
                await api.create_sessions(
                    ms_tokens=[MS_TOKEN],
                    num_sessions=1,
                    sleep_after=3,
                    browser=BROWSER,
                    headless=HEADLESS,
                )

                wrote = 0
                with open(RAW_OUT, "a", encoding="utf-8") as f:
                    async for video in api.trending.videos(count=count):
                        f.write(json.dumps(video.as_dict, ensure_ascii=False) + "\n")
                        wrote += 1
                        await asyncio.sleep(0.5 + random.random()*0.5)

                print(f"[ok] saved {wrote} items to {RAW_OUT}")
                return

        except EmptyResponseException as e:
            attempts += 1
            print(f"[warn] Empty response (attempt {attempts}): {e}")
            if attempts >= 3:
                raise
            await asyncio.sleep(3 * attempts)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    asyncio.run(trending_videos(count=n))
