import math, random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
import imageio.v3 as iio
from tqdm import trange
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline

# -------------------------
# PATHS (edit these two)
# -------------------------
MODELS_DIR = Path("/workspace/models")              # where Script A saved weights
OUTPUT_DIR = Path("/workspace/outputs/dogs_30s")   # where to save results
SDXL_PATH = MODELS_DIR / "sdxl-base-1.0"
SVD_PATH  = MODELS_DIR / "svd-img2vid-xt-1-1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# PROMPT / SETTINGS
# -------------------------
PROMPT = (
    "cute dogs frolicking in the grass, golden hour, soft sunlight, "
    "playful motion, shallow depth of field, cinematic, 4k, highly detailed"
)
NEGATIVE_PROMPT = "blurry, low quality, artifacts, watermark, text, logo"

# Keyframe size (multiples of 8). 1024x576 works great for SVD-XT.
WIDTH, HEIGHT = 1024, 576

# Each SVD clip length settings
FRAMES_PER_CLIP = 49   # try 25 if you see shape issues
FPS = 12               # output fps

# Target total duration
TARGET_SECONDS = 30

# Motion/style knobs
MOTION_BUCKET_ID = 127
CONDITIONING_SCALE = 1.8
STRENGTH = 0.60        # adherence to keyframe content (0..1). Higher = less motion
VARY_BETWEEN_CLIPS = True  # make tiny variations per segment

# Reproducibility
SEED = 12345  # set to None for random

# -------------------------
# Helpers
# -------------------------
def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    return seed

def pil_to_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")

def write_mp4(frames: List[Image.Image], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = [np.array(pil_to_rgb(f)) for f in frames]
    iio.imwrite(str(path), arr, fps=fps, codec="h264", quality=8)

# -------------------------
# Load models (from local dirs)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("Loading SDXL from", SDXL_PATH)
sdxl = StableDiffusionXLPipeline.from_pretrained(
    SDXL_PATH.as_posix(), torch_dtype=dtype, use_safetensors=True
).to(device)
sdxl.enable_xformers_memory_efficient_attention()

print("Loading SVD-XT from", SVD_PATH)
svd = StableVideoDiffusionPipeline.from_pretrained(
    SVD_PATH.as_posix(), torch_dtype=dtype, use_safetensors=True, variant="fp16"
).to(device)
svd.enable_xformers_memory_efficient_attention()
svd.vae.enable_tiling()

# -------------------------
# 1) Make a keyframe with SDXL
# -------------------------
seed_used = set_seed(SEED)
print("Seed:", seed_used)

print("Generating keyframe...")
keyframe = sdxl(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    guidance_scale=6.0,
    num_inference_steps=30,
    width=WIDTH,
    height=HEIGHT,
).images[0]
keyframe_path = OUTPUT_DIR / "keyframe.png"
keyframe.save(keyframe_path)
print("Saved keyframe ->", keyframe_path)

# -------------------------
# 2) Make enough clips to reach ~30s, then concatenate (pure Python)
# -------------------------
seconds_per_clip = FRAMES_PER_CLIP / FPS
num_clips = math.ceil(TARGET_SECONDS / seconds_per_clip)
print(f"Target {TARGET_SECONDS}s -> {num_clips} clip(s) ({FRAMES_PER_CLIP}f @ {FPS}fps ≈ {seconds_per_clip:.2f}s/clip).")

all_frames: List[Image.Image] = []

for i in trange(num_clips, desc="Rendering SVD clips"):
    # small variation to keep it lively
    motion = MOTION_BUCKET_ID
    scale = CONDITIONING_SCALE
    strength = STRENGTH
    if VARY_BETWEEN_CLIPS:
        motion = int(np.clip(MOTION_BUCKET_ID + np.random.randint(-10, 11), 1, 255))
        scale = float(np.clip(CONDITIONING_SCALE + np.random.uniform(-0.15, 0.15), 0.8, 3.0))
        strength = float(np.clip(STRENGTH + np.random.uniform(-0.04, 0.04), 0.4, 0.85))

    generator = torch.manual_seed(random.randint(0, 2**31 - 1))

    out = svd(
        image=keyframe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        num_frames=FRAMES_PER_CLIP,
        motion_bucket_id=motion,
        fps=FPS,
        conditioning_scale=scale,
        noise_aug_strength=0.02,
        decode_chunk_size=8,
        strength=strength,
        generator=generator,
    )
    frames = out.frames  # List[PIL.Image]

    # optionally save per-clip mp4s for debugging
    clip_path = OUTPUT_DIR / f"segment_{i:02d}.mp4"
    write_mp4(frames, clip_path, FPS)

    all_frames.extend(frames)

# cut to exact ~30s if we over-shot a little
target_frames = TARGET_SECONDS * FPS
if len(all_frames) > target_frames:
    all_frames = all_frames[:target_frames]

final_path = OUTPUT_DIR / "dogs_frolicking_30s.mp4"
write_mp4(all_frames, final_path, FPS)

print("\nDone!")
print("Final video ->", final_path.resolve())
print("Keyframe     ->", keyframe_path.resolve())
