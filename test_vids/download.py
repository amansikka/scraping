from huggingface_hub import snapshot_download
from pathlib import Path
import os

MODELS_DIR = Path("/workspace/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Disable xet backend to avoid quota errors
os.environ["HF_HUB_ENABLE_XET"] = "0"

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SVD_XT_REPO = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

print(f"Downloading to {MODELS_DIR.resolve()} ...")

sdxl_path = snapshot_download(
    repo_id=SDXL_REPO,
    local_dir=MODELS_DIR / "sdxl-base-1.0",
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
)
print("SDXL ->", sdxl_path)

svd_xt_path = snapshot_download(
    repo_id=SVD_XT_REPO,
    local_dir=MODELS_DIR / "svd-img2vid-xt-1-1",
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
)
print("SVD-XT ->", svd_xt_path)
