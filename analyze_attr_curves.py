#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Helpers: parsing
# ----------------------------
def parse_name(fname: str):
    """
    Split '{base_name}_{number}.png' into (base_name, float(number)).
    Works even if base_name itself contains underscores.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in stem:
        return None
    base, num_str = stem.rsplit("_", 1)
    try:
        val = float(num_str)
        return base, val
    except ValueError:
        return None


def base_from_example(example_file: str):
    parsed = parse_name(os.path.basename(example_file))
    if parsed is None:
        raise ValueError(
            f"--example_file must look like '{{base}}_{{number}}.png', got: {example_file}"
        )
    base, _ = parsed
    return base


# ----------------------------
# CLIP utilities
# ----------------------------
def build_clip(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, preprocess


@torch.no_grad()
def encode_prompts(prompts, tokenizer, clip_model, device):
    """
    Encode a list of prompts -> (D,) torch vector (normalized mean).
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    tokens = tokenizer(prompts).to(device)
    text_feats = clip_model.encode_text(tokens)
    text_feats = F.normalize(text_feats, dim=-1)
    mean_feat = F.normalize(text_feats.mean(dim=0, keepdim=True), dim=-1)
    return mean_feat.squeeze(0)  # (D,)


@torch.no_grad()
def encode_images_batch(pil_images, preprocess, clip_model, device):
    """
    Encode a batch of PIL images with CLIP preprocess.
    Returns (B, D) normalized features.
    """
    batch = torch.stack([preprocess(img) for img in pil_images], dim=0).to(device)
    feats = clip_model.encode_image(batch)
    feats = F.normalize(feats, dim=-1)
    return feats


@torch.no_grad()
def cosine_similarity(image_feat: torch.Tensor, prompt_feat: torch.Tensor) -> float:
    """
    Cosine similarity for normalized features (scalar).
    """
    return (image_feat * prompt_feat).sum().item()


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "For ONE image series (chosen by --example_file), plot CLIP Ratio (depth / flat) vs. Strength.\n"
            "No averaging across bases; uses only the selected base."
        )
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Folder containing {base}_{number}.png")
    parser.add_argument("--example_file", type=str, required=True,
                        help="One filename from the desired series (e.g., wood_0.0.png)")
    parser.add_argument("--prompts_json", type=str, required=True,
                        help="Path to prompts.json with 'depth' and 'flat' prompt sets")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="Prefix for output PNG; default auto: '{base_name}'")
    parser.add_argument("--model_name", type=str, default="ViT-B-16",
                        help="open_clip model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k",
                        help="open_clip pretrained tag")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for CLIP image encoding")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Small number to stabilize ratio depth/flat")
    parser.add_argument("--log_ratio", action="store_true",
                        help="If set, plot log((depth+eps)/(flat+eps)) instead of raw ratio")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve base name to analyze
    base_name = base_from_example(args.example_file)
    out_prefix = args.out_prefix or base_name

    # Ensure output folder exists (matches prior pattern)
    os.makedirs("plot", exist_ok=True)

    # 1) Load prompts and fixed keys
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    depth_key, flat_key = "depth", "flat"
    if depth_key not in prompts or flat_key not in prompts:
        raise KeyError(
            f"prompts.json must contain '{depth_key}' and '{flat_key}' keys."
        )

    # 2) Build CLIP and text features
    clip_model, tokenizer, preprocess = build_clip(args.model_name, args.pretrained, device)
    depth_feat = encode_prompts(prompts[depth_key], tokenizer, clip_model, device)  # (D,)
    flat_feat  = encode_prompts(prompts[flat_key],  tokenizer, clip_model, device)  # (D,)

    # 3) Collect this series only: {base_name}_{alpha}.png
    expected_alphas = [float(i) for i in range(-5, 6)]  # -5..5
    records = []  # list of (alpha, path, PIL)
    files_in_dir = os.listdir(args.images_dir)

    def snap_alpha(a):
        for v in expected_alphas:
            if abs(a - v) < 1e-3:
                return v
        return a

    for fname in files_in_dir:
        parsed = parse_name(fname)
        if parsed is None:
            continue
        b, a = parsed
        if b != base_name:
            continue
        path = os.path.join(args.images_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        records.append((snap_alpha(a), path, img))

    if not records:
        raise RuntimeError(
            f"No files for base '{base_name}' found in {args.images_dir}. "
            f"Expected names like {base_name}_-5.0.png … {base_name}_5.0.png"
        )

    # Sort by alpha
    records.sort(key=lambda x: x[0])

    # 4) CLIP encode in batches
    image_feats = {}
    with torch.no_grad():
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i+args.batch_size]
            pil_batch = [r[2] for r in batch]
            feats = encode_images_batch(pil_batch, preprocess, clip_model, device)  # (B, D)
            for (alpha, path, _), feat in zip(batch, feats):
                image_feats[path] = feat.cpu()

    # 5) Compute CLIP depth/flat ratio for this series
    ratio_curve = OrderedDict()
    for (alpha, path, _) in records:
        feat = image_feats[path]
        sim_depth = cosine_similarity(feat, depth_feat.cpu())
        sim_flat  = cosine_similarity(feat, flat_feat.cpu())
        if args.log_ratio:
            ratio = float(np.log((sim_depth + args.eps) / (sim_flat + args.eps)))
        else:
            ratio = sim_depth / (sim_flat + args.eps)
            ratio = sim_depth / 1.0
        ratio_curve[alpha] = ratio

    # Prepare arrays
    alphas = np.array(list(ratio_curve.keys()), dtype=np.float64)
    ratio_vals = np.array(list(ratio_curve.values()), dtype=np.float64)

    # 6) Plot: CLIP depth/flat ratio vs. Strength
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.plot(alphas, ratio_vals, marker="s")
    ax.set_xlabel("Editing Strength (alpha)")
    ylabel = f"CLIP {'log ' if args.log_ratio else ''}Ratio: {depth_key} / {flat_key}"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{base_name}: {ylabel} vs. Strength")
    ax.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    suffix = "logratio" if args.log_ratio else "ratio"
    out_png = f"plot/{out_prefix}_clip_{suffix}_{depth_key}_over_{flat_key}.png"
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[OK] Saved CLIP {suffix} plot to {out_png}")

    # 7) Console table
    print(f"\nalpha\tclip_{suffix}({depth_key}/{flat_key})")
    for a in alphas:
        print(f"{a:+.1f}\t{ratio_curve[a]:+.6f}")


if __name__ == "__main__":
    main()
