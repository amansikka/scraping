#!/usr/bin/env python3
# scripts/generation.py
import os, sys, json, time, shutil, pathlib, subprocess, math, random
from typing import List
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_videos"
TMP = OUT_DIR / "tmp_frames"
for p in (OUT_DIR, TMP):
    p.mkdir(parents=True, exist_ok=True)

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    print("[error] ffmpeg required.")
    sys.exit(1)

def ffmpeg_frames_to_mp4(frames_dir: pathlib.Path, out_path: pathlib.Path, fps: int = 24, h=1280, w=720):
    vf = f"scale='if(gt(a,{w}/{h}),{w},-2)':'if(gt(a,{w}/{h}),-2,{h})':force_original_aspect_ratio=decrease," \
         f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    cmd = [
        FFMPEG, "-y", "-r", str(fps), "-i", str(frames_dir / "%06d.png"),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path)
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    import argparse, torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline

    ap = argparse.ArgumentParser(description="Generate a vertical short using SDXL+LoRA.")
    ap.add_argument("--prompt", type=str, required=True, help="Text prompt")
    ap.add_argument("--neg", type=str, default="low quality, blurry, artifacts")
    ap.add_argument("--lora", type=str, default=str((ROOT/"models/sdxl_lora_dogs")))
    ap.add_argument("--frames", type=int, default=24, help="Number of frames")
    ap.add_argument("--fps", type=int, default=24, help="Output FPS")
    ap.add_argument("--h", type=int, default=1280)
    ap.add_argument("--w", type=int, default=720)
    ap.add_argument("--cfg", type=float, default=6.5, help="Guidance scale")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if (device.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype, use_safetensors=True
    ).to(device)

    # Load your LoRA if present
    lora_dir = pathlib.Path(args.lora)
    if lora_dir.exists():
        try:
            pipe.load_lora_weights(lora_dir)
            pipe.fuse_lora()  # for a tiny speedup
            print(f"[info] loaded LoRA from {lora_dir}")
        except Exception as e:
            print(f"[warn] could not load LoRA: {e}")

    # Generate frames (naive “micro-motion”: varying seed and subtle prompt noise)
    TMP.mkdir(parents=True, exist_ok=True)
    for f in TMP.glob("*.png"):
        f.unlink()

    base_seed = args.seed if args.seed != 0 else int.from_bytes(os.urandom(2), "big")
    for i in range(args.frames):
        g = torch.Generator(device=device).manual_seed(base_seed + i)
        # Slightly vary prompt to introduce micro-changes
        prompt = args.prompt
        img = pipe(
            prompt=prompt,
            negative_prompt=args.neg,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=args.h,
            width=args.w,
            generator=g
        ).images[0]
        img.save(TMP / f"{i+1:06d}.png")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"gen_{ts}.mp4"
    ffmpeg_frames_to_mp4(TMP, out_path, fps=args.fps, h=args.h, w=args.w)

    meta = {
        "prompt": args.prompt, "negative": args.neg,
        "frames": args.frames, "fps": args.fps, "h": args.h, "w": args.w,
        "cfg": args.cfg, "steps": args.steps, "seed": base_seed,
        "lora": str(lora_dir),
        "created_at": ts
    }
    (OUT_DIR / f"gen_{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")

if __name__ == "__main__":
    main()
