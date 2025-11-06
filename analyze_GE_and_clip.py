#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
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
# Helpers: parsing & luminance
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


def to_luminance_np(pil_img: Image.Image) -> np.ndarray:
    """
    Compute luminance using Rec.709 on sRGB (approx):
      Y = 0.2126 R + 0.7152 G + 0.0722 B
    Returns float64 array (H, W).
    """
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float64)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ----------------------------
# Roughness: Gradient Energy
# ----------------------------
def sobel_gradients(Y: np.ndarray):
    """
    Compute Sobel gradients with reflect padding (NumPy-only).
    Returns gx, gy (same shape as Y).
    """
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    Ky = Kx.T

    Yp = np.pad(Y, pad_width=1, mode='reflect')

    # Vectorized 3x3 convolution
    def conv3(Ypad, K):
        return (
            K[0,0]*Ypad[:-2, :-2] + K[0,1]*Ypad[:-2, 1:-1] + K[0,2]*Ypad[:-2, 2:] +
            K[1,0]*Ypad[1:-1, :-2] + K[1,1]*Ypad[1:-1, 1:-1] + K[1,2]*Ypad[1:-1, 2:] +
            K[2,0]*Ypad[2:, :-2] + K[2,1]*Ypad[2:, 1:-1] + K[2,2]*Ypad[2:, 2:]
        )

    gx = conv3(Yp, Kx)
    gy = conv3(Yp, Ky)
    return gx, gy


def gradient_energy(Y: np.ndarray) -> float:
    """
    Mean gradient magnitude (Sobel). Higher => rougher.
    """
    gx, gy = sobel_gradients(Y)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


# ----------------------------
# CLIP: encode prompts & images
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
            "For ONE image series (chosen by --example_file), plot:\n"
            "  1) Gradient Energy (roughness proxy) vs. Strength\n"
            "  2) CLIP Ratio (glossiness / matteness) vs. Strength\n"
            "No averaging across bases; uses only the selected base."
        )
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Folder containing {base}_{number}.png")
    parser.add_argument("--example_file", type=str, required=True,
                        help="One filename from the desired series (e.g., wood_0.0.png)")
    parser.add_argument("--prompts_json", type=str, required=True,
                        help="Path to prompts.json with glossiness + matteness prompt sets")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="Prefix for output PNGs; default auto: '{base_name}'")
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
                        help="Small number to stabilize ratio glossiness/matteness")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve base name to analyze
    base_name = base_from_example(args.example_file)
    out_prefix = args.out_prefix or base_name

    # Ensure output folder exists (keeps your original 'plot/' paths)
    os.makedirs("plot", exist_ok=True)

    # 1) Load prompts and fixed keys
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Fixed key names per your request
    gloss_key, matte_key = "roughness", "smoothness"
    if gloss_key not in prompts or matte_key not in prompts:
        raise KeyError(
            f"prompts.json must contain '{gloss_key}' and '{matte_key}' keys."
        )

    # 2) Build CLIP and text features
    clip_model, tokenizer, preprocess = build_clip(args.model_name, args.pretrained, device)
    gloss_feat = encode_prompts(prompts[gloss_key], tokenizer, clip_model, device)  # (D,)
    matte_feat = encode_prompts(prompts[matte_key], tokenizer, clip_model, device)  # (D,)

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

    # 5) Compute per-image metrics for this series
    # Curves: alpha -> energy, alpha -> ratio
    energy_curve = OrderedDict()
    ratio_curve = OrderedDict()

    for (alpha, path, pil_img) in records:
        # Gradient energy (roughness proxy)
        Y = to_luminance_np(pil_img)
        energy = gradient_energy(Y)

        # CLIP glossiness/matteness ratio
        feat = image_feats[path]
        sim_gloss = cosine_similarity(feat, gloss_feat.cpu())
        sim_matte = cosine_similarity(feat, matte_feat.cpu())
        ratio = sim_gloss / (sim_matte + args.eps)

        energy_curve[alpha] = energy
        ratio_curve[alpha] = ratio

    # Prepare arrays
    alphas = np.array(list(energy_curve.keys()), dtype=np.float64)
    energy_vals = np.array(list(energy_curve.values()), dtype=np.float64)
    ratio_vals = np.array(list(ratio_curve.values()), dtype=np.float64)

    # 6) Plot 1: Energy vs. Strength
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax1.plot(alphas, energy_vals, marker="o")
    ax1.set_xlabel("Editing Strength (alpha)")
    ax1.set_ylabel("Gradient Energy (Sobel mean magnitude)")
    ax1.set_title(f"{base_name}: Energy (roughness) vs. Strength")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    out_energy = f"plot/{out_prefix}_energy.png"
    plt.savefig(out_energy, bbox_inches="tight")
    print(f"[OK] Saved energy plot to {out_energy}")

    # 7) Plot 2: CLIP glossiness/matteness ratio vs. Strength
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax2.plot(alphas, ratio_vals, marker="s")
    ax2.set_xlabel("Editing Strength (alpha)")
    ax2.set_ylabel(f"CLIP Ratio: {gloss_key} / {matte_key}")
    ax2.set_title(f"{base_name}: CLIP Ratio ({gloss_key}/{matte_key}) vs. Strength")
    ax2.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    out_ratio = f"plot/{out_prefix}_clip_ratio_{gloss_key}_over_{matte_key}.png"
    plt.savefig(out_ratio, bbox_inches="tight")
    print(f"[OK] Saved CLIP ratio plot to {out_ratio}")

    # 8) Console table
    print("\nalpha\tenergy\t\tclip_ratio(glossiness/matteness)")
    for a in alphas:
        print(f"{a:+.1f}\t{energy_curve[a]:+.6f}\t{ratio_curve[a]:+.6f}")


if __name__ == "__main__":
    main()
