# caption_folder.py
import os, json, sys, argparse
from pathlib import Path

import torch
from PIL import Image
import open_clip

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def decode_caption(tokens):
    # Clean CoCa special tokens
    txt = open_clip.decode(tokens).split("<end_of_text>")[0]
    txt = txt.replace("<start_of_text>", "").strip()
    return txt

def load_images(folder):
    folder = Path(folder)
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing images")
    ap.add_argument("--out", default="captions.json", help="Output JSON path")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CoCa (image-to-text) model + transform
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model = model.to(device).eval()

    # Gather image paths
    paths = list(load_images(args.folder))
    if not paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    results = {}
    batch, batch_names = [], []

    @torch.no_grad()
    def flush_batch():
        nonlocal batch, batch_names
        if not batch:
            return
        ims = torch.stack(batch, dim=0).to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            generated = model.generate(ims)  # [B, seq]
        for name, toks in zip(batch_names, generated):
            results[name] = decode_caption(toks)
        batch.clear()
        batch_names.clear()

    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            batch.append(transform(im))
            batch_names.append(p.name)
            if len(batch) >= args.batch:
                flush_batch()
        except Exception as e:
            # store the error as the "caption" so you can spot failures
            results[p.name] = f"[ERROR] {e}"

    flush_batch()

    # Save JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} captions to {args.out}")

if __name__ == "__main__":
    main()
