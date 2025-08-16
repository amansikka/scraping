#!/usr/bin/env python3
# scripts/finetuning.py
import os, sys, json, math, random, shutil, pathlib, subprocess
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training"
RAW  = DATA / "raw_videos" / "youtube"
FRM  = DATA / "frames"
CAPS = DATA / "captions"
MODELS = ROOT / "models" / "sdxl_lora_dogs"
MANI = DATA / "manifests"
for p in (FRM, CAPS, MODELS, MANI):
    p.mkdir(parents=True, exist_ok=True)

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    print("[error] ffmpeg required for frame extraction.")
    sys.exit(1)

# ---------- Step 1: Extract frames ----------
def extract_frames(mp4_path: pathlib.Path, out_dir: pathlib.Path, fps: int = 4, size=1024):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Letterbox to square 1024x1024 for SDXL
    vf = f"scale='if(gt(a,1),{size},-2)':'if(gt(a,1),-2,{size})':force_original_aspect_ratio=decrease," \
         f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    out_tpl = str(out_dir / "%06d.jpg")
    cmd = [FFMPEG, "-y", "-i", str(mp4_path), "-vf", f"{vf},fps={fps}", "-q:v", "2", out_tpl]
    print("[extract]", " ".join(cmd))
    subprocess.check_call(cmd)

def find_videos() -> List[pathlib.Path]:
    return sorted([p for p in RAW.glob("*.mp4") if p.is_file()])

# ---------- Step 2: Caption frames (simple) ----------
def simple_caption(file: pathlib.Path) -> str:
    # Super basic placeholder; upgrade to BLIP/CLIP later.
    # You can also parse surrounding video metadata to enrich this.
    return "a cute dog, playful, soft light, high quality photo"

def caption_frames(frame_dir: pathlib.Path, topic: str) -> List[Dict[str, Any]]:
    rows = []
    for img in sorted(frame_dir.glob("*.jpg")):
        cap = f"{topic}, {simple_caption(img)}"
        rows.append({"image": str(img.relative_to(ROOT)), "prompt": cap})
    return rows

# ---------- Step 3: Optional LoRA training on SDXL ----------
TRAIN_NOTE = """
If you lack a GPU with sufficient VRAM (>=16GB recommended), you can skip training for now.
Otherwise, ensure torch + diffusers + accelerate + transformers + peft are installed.
"""

def train_lora_sdxl(
    dataset_jsonl: pathlib.Path,
    output_dir: pathlib.Path,
    max_steps: int = 800,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    seed: int = 42,
):
    """
    Minimal SDXL LoRA training loop (UNet only) using diffusers.
    Trains on square 1024x1024 frames with simple text prompts.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    from diffusers import (
        StableDiffusionXLPipeline,
        DDPMScheduler,
    )
    from diffusers.loaders import AttnProcsLayers
    from diffusers.utils import USE_PEFT_BACKEND
    from peft import LoraConfig

    # --- Dataset ---
    class FrameSet(Dataset):
        def __init__(self, items):
            self.items = items
            self.tx = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # expect -1..1
            ])
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            row = self.items[idx]
            img = Image.open(ROOT/row["image"]).convert("RGB")
            t = self.tx(img)
            return {"pixel_values": t, "prompt": row["prompt"]}

    data = [json.loads(l) for l in dataset_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(seed); random.shuffle(data)
    train_items = data
    ds = FrameSet(train_items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # --- Base pipeline & components ---
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=dtype, use_safetensors=True
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
    pipe.set_progress_bar_config(disable=True)

    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze VAE + text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(True)

    # Attach LoRA to UNet attention blocks
    lora_config = LoraConfig(
        r=8, lora_alpha=16, init_lora_weights="gaussian", target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    unet.add_adapter(lora_config)
    attn_procs = AttnProcsLayers(unet.attn_processors)
    params_to_opt = list(attn_procs.parameters())

    opt = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=1e-2)
    global_step = 0

    # Helper to build SDXL extra cond (time ids)
    def make_time_ids(bsz: int, H=1024, W=1024, device=None, dtype=torch.float32):
        # (orig_h, orig_w, crop_h, crop_w, target_h, target_w) pattern used in SDXL
        # We keep square, no crop
        ti = torch.tensor([H, W, 0, 0, H, W], device=device, dtype=dtype)
        return ti.unsqueeze(0).repeat(bsz, 1)

    pipe.unet.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16 and device.type=="cuda"))

    loss_avg = 0.0
    progress = tqdm(total=max_steps, desc="LoRA SDXL training")
    while global_step < max_steps:
        for batch in dl:
            if global_step >= max_steps:
                break
            with torch.no_grad():
                # encode text with both encoders
                prompts = batch["prompt"]
                prompt_embeds, pooled = pipe.encode_prompt(
                    prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                # encode images to latents
                imgs = batch["pixel_values"].to(device=device, dtype=dtype)
                latents = vae.encode(imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # sample timestep/noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # added cond kwargs for SDXL
            added_kwargs = {
                "text_embeds": pooled,
                "time_ids": make_time_ids(latents.shape[0], device=device, dtype=dtype),
            }

            with torch.cuda.amp.autocast(enabled=(dtype!=torch.float32)):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_kwargs,
                ).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()

            if (global_step + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            global_step += 1
            loss_avg = loss_avg * 0.98 + float(loss.detach().cpu()) * 0.02
            progress.set_postfix(loss=f"{loss_avg:.4f}")
            progress.update(1)

    progress.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save LoRA weights
    pipe.save_lora_weights(output_dir)
    (output_dir / "training_args.json").write_text(json.dumps({
        "max_steps": max_steps, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum, "seed": seed
    }, indent=2), encoding="utf-8")
    print(f"[done] LoRA saved to {output_dir}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Prep frames/captions and (optionally) train SDXL LoRA.")
    ap.add_argument("--topic", type=str, default="cute dogs")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--train", action="store_true", help="Run LoRA training after prep")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    videos = find_videos()
    if not videos:
        print("[error] No videos found in", RAW)
        sys.exit(1)

    # 1) Extract frames
    all_rows: List[Dict[str, Any]] = []
    for v in videos:
        out_dir = FRM / v.stem
        if any(out_dir.glob("*.jpg")):
            print(f"[skip] frames exist for {v.stem}")
        else:
            extract_frames(v, out_dir, fps=args.fps, size=1024)

        rows = caption_frames(out_dir, args.topic)
        all_rows.extend(rows)

    # 2) Save captions jsonl
    caps_path = CAPS / "frames_captions.jsonl"
    with caps_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3) Save manifest
    (MANI / "frame_manifest.json").write_text(json.dumps(all_rows[:20], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Wrote {len(all_rows)} frame rows → {caps_path}")

    # 4) Optional training
    if args.train:
        print(TRAIN_NOTE)
        train_lora_sdxl(caps_path, MODELS, max_steps=args.steps)

if __name__ == "__main__":
    main()
