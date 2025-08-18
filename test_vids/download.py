from huggingface_hub import snapshot_download
from pathlib import Path
import os

# ---------- knobs ----------
MODELS_DIR = Path("/workspace/models")
SDXL_REPO  = "stabilityai/stable-diffusion-xl-base-1.0"
SVD_REPO   = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
HF_TOKEN   = os.environ.get("HF_TOKEN")   # or login via huggingface-cli
# ---------------------------

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Avoid the xet CAS backend (prevents the bogus “Disk quota exceeded”)
os.environ["HF_HUB_ENABLE_XET"] = "0"

# 2) Keep concurrency modest (avoids 429s)
os.environ.setdefault("HF_HUB_DOWNLOAD_MAX_THREADS", "4")

# 3) Only pull what PyTorch diffusers needs (skip ONNX/OpenVINO/FLAX, images)
ALLOW_COMMON = [
    "*.safetensors", "*.json", "*.txt", "*.py", "model_index.json", "LICENSE*",
    "README.md",
]
IGNORE_HEAVY = [
    "*.onnx", "*openvino*", "*flax*", "*.msgpack", "*.bin", "*.png", "*.jpg", "*.jpeg"
]

print(f"Downloading to {MODELS_DIR.resolve()} ...")

sdxl_path = snapshot_download(
    repo_id=SDXL_REPO,
    local_dir=(MODELS_DIR / "sdxl-base-1.0"),
    token=HF_TOKEN,
    local_dir_use_symlinks=False,  # harmless even if deprecated
    allow_patterns=ALLOW_COMMON + [
        # SDXL subfolders we actually need
        "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
        "unet/*", "vae/*", "vae_decoder/*", "vae_encoder/*", "scheduler/*",
    ],
    ignore_patterns=IGNORE_HEAVY,
)
print("SDXL ->", sdxl_path)

svd_path = snapshot_download(
    repo_id=SVD_REPO,
    local_dir=(MODELS_DIR / "svd-img2vid-xt-1-1"),
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
    allow_patterns=ALLOW_COMMON + [
        # Typical SVD structure (keep core weights/configs)
        "image_encoder/*", "text_encoder/*", "tokenizer/*",
        "unet/*", "vae/*", "scheduler/*",
    ],
    ignore_patterns=IGNORE_HEAVY,
)
print("SVD-XT ->", svd_path)

print("\nDone. Paths are:")
print("  SDXL:", (MODELS_DIR / "sdxl-base-1.0").resolve())
print("  SVD :", (MODELS_DIR / "svd-img2vid-xt-1-1").resolve())
