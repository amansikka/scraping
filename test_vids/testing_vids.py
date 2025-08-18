import os, math, subprocess, tempfile, random
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import StableVideoDiffusionPipeline  # base
from diffusers import StableVideoDiffusionImg2VidPipeline  # img2vid-xt
from PIL import Image
import numpy as np
import imageio.v3 as iio
from tqdm import trange

# -------------------------
# Config
# -------------------------
PROMPT = (
    "cute dogs frolicking in the grass, golden hour, soft sunlight, "
    "playful motion, shallow depth of field, cinematic, 4k, highly detailed"
)
NEGATIVE_PROMPT = "blurry, low quality, artifacts, watermark, text, logo"

# Resolution for keyframe (SVD likes landscape-ish; keep within VRAM)
WIDTH, HEIGHT = 1024, 576

# Each SVD-XT clip: number of frames and fps. (49@12fps ≈ 4.1s per clip)
# If you hit a shape assertion error, try FRAMES_PER_CLIP=25 instead.
FRAMES_PER_CLIP = 49
FPS = 12

# Target total duration (seconds)
TARGET_SECONDS = 30

# Motion / style controls for SVD (tweak-friendly defaults)
MOTION_BUCKET_ID = 127  # 1..255-ish; higher = more motion
CONDITIONING_SCALE = 1.8  # 1.0–2.5 sensible
AUGMENT_PIPELINE = True   # random slight variations across segments

OUT_DIR = Path("out_dogs_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Models (downloaded automatically from HF). You can swap in local paths.
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SVD_XT_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

# Seeds (set to None for random)
SEED = None

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int | None):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    return seed

def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

def write_mp4(frames: List[Image.Image], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = [np.array(pil_to_rgb(f)) for f in frames]
    iio.imwrite(path.as_posix(), arr, fps=fps, codec="h264", quality=8)

def concat_videos_ffmpeg(video_paths: List[Path], out_path: Path):
    # Use ffmpeg concat demuxer
    list_file = out_path.with_suffix(".txt")
    with open(list_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file.as_posix(),
        "-c", "copy",
        out_path.as_posix(),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)

# -------------------------
# Load models
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("Loading SDXL (text->image)...")
sdxl = StableDiffusionXLPipeline.from_pretrained(
    SDXL_MODEL, torch_dtype=dtype, use_safetensors=True
)
sdxl.to(device)
sdxl.enable_xformers_memory_efficient_attention()
sdxl.enable_model_cpu_offload() if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 28e9 else None

print("Loading SVD-XT (img->video)...")
svd = StableVideoDiffusionImg2VidPipeline.from_pretrained(
    SVD_XT_MODEL, torch_dtype=dtype, use_safetensors=True, variant="fp16"
)
svd.to(device)
svd.enable_xformers_memory_efficient_attention()
svd.vae.enable_tiling()  # helps with larger frames

# -------------------------
# Create keyframe with SDXL
# -------------------------
seed_used = set_seed(SEED)
print(f"Seed: {seed_used}")

print("Generating keyframe...")
keyframe = sdxl(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    guidance_scale=6.0,
    num_inference_steps=30,
    width=WIDTH,
    height=HEIGHT,
).images[0]
keyframe_path = OUT_DIR / "keyframe.png"
keyframe.save(keyframe_path)
print(f"Saved keyframe -> {keyframe_path}")

# -------------------------
# Generate multiple clips to reach ~30s
# -------------------------
seconds_per_clip = FRAMES_PER_CLIP / FPS
num_clips = math.ceil(TARGET_SECONDS / seconds_per_clip)

print(f"Target {TARGET_SECONDS}s; making {num_clips} clips "
      f"({FRAMES_PER_CLIP} frames @ {FPS} fps ≈ {seconds_per_clip:.2f}s/clip).")

segment_paths = []
base_motion = MOTION_BUCKET_ID
base_scale = CONDITIONING_SCALE

for i in trange(num_clips, desc="Rendering clips"):
    # slight variation to keep motion lively
    motion = base_motion
    scale = base_scale
    strength = 0.6  # how strongly we adhere to the input image (0..1)

    if AUGMENT_PIPELINE:
        motion = int(np.clip(base_motion + np.random.randint(-10, 11), 1, 255))
        scale = float(np.clip(base_scale + np.random.uniform(-0.15, 0.15), 0.8, 3.0))
        strength = float(np.clip(0.55 + np.random.uniform(-0.05, 0.05), 0.4, 0.8))

    generator = torch.manual_seed(random.randint(0, 2**31 - 1))
    frames = svd(
        image=keyframe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,             # 20–40 usually fine
        num_frames=FRAMES_PER_CLIP,         # try 25 if you see a shape error
        motion_bucket_id=motion,            # motion intensity
        fps=FPS,                            # baked meta; we’ll also write mp4 at this FPS
        conditioning_scale=scale,           # prompt adherence
        noise_aug_strength=0.02,            # tiny noise to fight drift/artifacts
        decode_chunk_size=8,
        generator=generator,
        strength=strength,                  # adherence to keyframe content
    ).frames  # List[PIL.Image]

    seg_path = OUT_DIR / f"segment_{i:02d}.mp4"
    write_mp4(frames, seg_path, fps=FPS)
    segment_paths.append(seg_path)

# -------------------------
# Concatenate to final video
# -------------------------
final_path = OUT_DIR / "dogs_frolicking_30s.mp4"
concat_videos_ffmpeg(segment_paths, final_path)

print(f"\nDone! Final video -> {final_path.resolve()}")
print("Tip: If motion is too wild, lower MOTION_BUCKET_ID or raise strength slightly.")
