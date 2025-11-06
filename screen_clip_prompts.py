#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find best 5-prompt combinations that maximize CLIP accuracy
for distinguishing natural vs. unnatural textures.
"""

import os, json, argparse, itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import open_clip
import numpy as np
import random

# -----------------------------
# 1. Fixed 100 Prompts
# -----------------------------
def get_prompt_sets():
    natural_prompts = [
        "a realistic photograph of a natural texture",
        "a detailed close-up of a real surface",
        "an authentic texture captured by camera",
        "a lifelike material surface under soft light",
        "a macro shot of natural details",
        "a genuine surface pattern from nature",
        "a real-world photo of material texture",
        "a faithful image of natural appearance",
        "a realistic texture photograph without artifacts",
        "a high-quality capture of natural surface structure",
        "a naturally patterned surface in daylight",
        "a clean, high-resolution image of texture",
        "an unretouched photo of real material",
        "a true-to-life texture under neutral lighting",
        "a natural surface with organic variation",
        "a realistic image of fine surface detail",
        "a genuine texture showing random irregularities",
        "a naturally lit close-up of texture",
        "a crisp, artifact-free surface photograph",
        "a macro image of realistic texture",
        "an authentic, undistorted texture image",
        "a texture photograph with natural color tones",
        "a photo of natural material without enhancement",
        "a realistic photo with smooth tonal transitions",
        "a natural-looking photo of surface detail",
        "a high-fidelity capture of real texture",
        "a realistic depiction of surface microstructure",
        "a true photograph of real material",
        "a close-up of a natural-looking surface pattern",
        "an evenly illuminated photo of real texture",
        "a photo of natural material with balanced contrast",
        "a faithful macro photo of real surface",
        "a natural texture captured in ambient light",
        "a realistic photograph of a smooth texture",
        "a genuine close-up of surface roughness",
        "a natural texture with subtle shadows",
        "a texture photo with continuous tone and color",
        "a clean macro photograph of surface details",
        "an authentic depiction of physical texture",
        "a texture photograph showing organic irregularities",
        "a natural close-up photo with realistic lighting",
        "a genuine image of material surface under soft light",
        "a realistic photo showing fine surface grain",
        "a natural texture image without post-processing",
        "a photograph of surface texture with depth and realism",
        "a faithful representation of real-world texture",
        "a macro photo of natural material surface",
        "a clear, realistic photo of natural texture",
        "an authentic surface photo captured under daylight",
        "a true photo of physical texture structure",
    ]

    unnatural_prompts = [
        "a synthetic texture with visible artifacts",
        "an artificial texture with unrealistic highlights",
        "a computer-generated texture lacking realism",
        "a digitally produced texture with banding",
        "an overprocessed texture image",
        "a low-quality texture with compression artifacts",
        "a fake texture with harsh lighting",
        "a noisy digital texture",
        "an oversharpened image of a texture",
        "a rendered texture with unnatural edges",
        "a distorted texture with visual artifacts",
        "an image with pixelation and noise",
        "a fake surface pattern lacking realism",
        "a digitally manipulated texture",
        "a synthetic material with unrealistic shading",
        "a fake-looking photo of a surface",
        "an artificial texture with exaggerated contrast",
        "a distorted, low-quality surface image",
        "a computer-rendered texture with tiling",
        "a digital texture with visible grid artifacts",
        "a noisy, grainy artificial texture",
        "a fake surface with blown highlights",
        "an oversaturated artificial texture",
        "a rendered material with plastic appearance",
        "an image of texture with color banding",
        "a low-fidelity digital surface pattern",
        "a fake texture exhibiting halos and overshoot",
        "a synthetic texture with checkerboard patterns",
        "a texture with moiré or aliasing effects",
        "a distorted digital rendering of texture",
        "an artificial texture with harsh reflections",
        "a poorly rendered texture with visible noise",
        "a fake material texture with tiling seams",
        "a computer-generated surface lacking depth",
        "a digital texture with flat, uniform lighting",
        "an unrealistic artificial texture with clipping",
        "an image with overexposed highlights and artifacts",
        "a computer-synthesized texture with strange patterns",
        "a fake glossy texture with noise artifacts",
        "a low-resolution digital texture image",
        "a synthetic texture exhibiting edge halos",
        "a fake surface image with pixel noise",
        "a distorted texture due to excessive sharpening",
        "a digitally enhanced texture with unnatural contrast",
        "an artificial texture with reflective glare artifacts",
        "an unrealistic surface texture from 3D rendering",
        "a photo with visible JPEG compression",
        "a digitally generated texture lacking fine detail",
        "a fake texture showing posterization artifacts",
        "an unnatural, computer-rendered texture image",
    ]

    return natural_prompts, unnatural_prompts

def setup_clip(model_name="ViT-L-14", pretrained="openai"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess, device


def load_images(folder, preprocess, device, max_n=None):
    imgs = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = preprocess(Image.open(os.path.join(folder, fn)).convert("RGB")).unsqueeze(0)
        imgs.append(img)
        if max_n and len(imgs) >= max_n:
            break
    return torch.cat(imgs, 0).to(device)


@torch.no_grad()
def encode_images(imgs, model):
    return F.normalize(model.encode_image(imgs), dim=-1)


@torch.no_grad()
def encode_texts(prompts, model, tokenizer, device):
    toks = tokenizer(prompts).to(device)
    return F.normalize(model.encode_text(toks), dim=-1)


# ---------------------------
# Dual comparison accuracy
# ---------------------------
@torch.no_grad()
def dual_accuracy(img_nat, img_unn, feat_nat, feat_unn):
    sim_nat_nat = (img_nat @ feat_nat.T).mean(dim=1)
    sim_nat_unn = (img_nat @ feat_unn.T).mean(dim=1)
    sim_unn_nat = (img_unn @ feat_nat.T).mean(dim=1)
    sim_unn_unn = (img_unn @ feat_unn.T).mean(dim=1)

    # correct when each sample prefers its own combo
    acc_nat = (sim_nat_nat > sim_nat_unn).float().mean().item()
    acc_unn = (sim_unn_unn > sim_unn_nat).float().mean().item()
    acc_total = (acc_nat + acc_unn) / 2.0
    return acc_total, acc_nat, acc_unn


# ---------------------------
# Search best combo
# ---------------------------
def search_best_combo(feats_nat, feats_unn, img_nat, img_unn, n_try=20000, k=5):
    n_nat = feats_nat.shape[0]
    n_unn = feats_unn.shape[0]
    combos_nat = random.sample(list(itertools.combinations(range(n_nat), k)), min(n_try, len(list(itertools.combinations(range(n_nat), k)))))
    combos_unn = random.sample(list(itertools.combinations(range(n_unn), k)), min(n_try, len(list(itertools.combinations(range(n_unn), k)))))

    best_acc, best_pair = 0.0, None
    for combo_nat in tqdm(combos_nat, desc="Nat combos"):
        f_nat_combo = F.normalize(feats_nat[list(combo_nat)].mean(dim=0, keepdim=True), dim=-1)
        for combo_unn in random.sample(combos_unn, min(200, len(combos_unn))):  # pair partial search
            f_unn_combo = F.normalize(feats_unn[list(combo_unn)].mean(dim=0, keepdim=True), dim=-1)
            acc, acc_n, acc_u = dual_accuracy(img_nat, img_unn, f_nat_combo, f_unn_combo)
            if acc > best_acc:
                best_acc = acc
                best_pair = (combo_nat, combo_unn, acc_n, acc_u)
    return best_pair, best_acc


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Dual-comparison CLIP prompt search (5 natural vs 5 unnatural)")
    ap.add_argument("--natural_dir", required=True)
    ap.add_argument("--unnatural_dir", required=True)
    ap.add_argument("--max_imgs", type=int, default=None)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--out_json", default="prompt_dual_best.json")
    args = ap.parse_args()

    natural_prompts, unnatural_prompts = get_prompt_sets()
    model, tokenizer, preprocess, device = setup_clip()

    imgs_nat = load_images(args.natural_dir, preprocess, device, args.max_imgs)
    imgs_unn = load_images(args.unnatural_dir, preprocess, device, args.max_imgs)
    f_nat, f_unn = encode_images(imgs_nat, model), encode_images(imgs_unn, model)

    t_nat = encode_texts(natural_prompts, model, tokenizer, device)
    t_unn = encode_texts(unnatural_prompts, model, tokenizer, device)

    (best_nat_idx, best_unn_idx, acc_nat, acc_unn), best_acc = search_best_combo(
        t_nat, t_unn, f_nat, f_unn, n_try=args.samples, k=5
    )

    print(f"\n✅ Best Overall Accuracy: {best_acc*100:.2f}%")
    print(f"   Natural folder accuracy:   {acc_nat*100:.2f}%")
    print(f"   Unnatural folder accuracy: {acc_unn*100:.2f}%\n")

    print("Best 5 Natural Prompts:")
    for i in best_nat_idx:
        print(" -", natural_prompts[i])

    print("\nBest 5 Unnatural Prompts:")
    for i in best_unn_idx:
        print(" -", unnatural_prompts[i])

    json.dump(
        {
            "accuracy_total": best_acc,
            "accuracy_nat": acc_nat,
            "accuracy_unn": acc_unn,
            "best_natural_prompts": [natural_prompts[i] for i in best_nat_idx],
            "best_unnatural_prompts": [unnatural_prompts[i] for i in best_unn_idx],
        },
        open(args.out_json, "w"),
        indent=2,
    )
    print(f"\nResults saved → {args.out_json}")


if __name__ == "__main__":
    main()