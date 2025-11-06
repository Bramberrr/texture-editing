#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run inference on a re-merged image from saved tiles, with optional video rendering.

New:
- --make_video to export an mp4 over a strength sweep (default -10..10, 41 steps)
- Uses cv2 VideoWriter; frames stored in a tmp folder (configurable), optional cleanup
"""

import os, json, math, argparse, glob
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import dnnlib
from PIL import Image
import numpy as np
import cv2  # <-- for video

# ------------------------------
# Optional deps you already use
# ------------------------------
from model.band_sifting import band_sifting_editing
from gloss_editor.predict import TextCondDelta, TextCondDeltaW
import open_clip

# ------------------------------
# Constants
# ------------------------------
OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD  = [0.26862954, 0.26130258, 0.27577711]

# ------------------------------
# Generator helpers
# ------------------------------
def get_generator_kwargs(device):
    from model.trainer import get_generator_kwargs as _gk
    return _gk(device=device)

def load_generator(g_path: str, device: torch.device):
    G_kwargs = get_generator_kwargs(device=device)
    G_ema = dnnlib.util.construct_class_by_name(**G_kwargs).eval().to(device)
    state = torch.load(g_path, map_location="cpu")
    G_ema.load_state_dict(state, strict=False)
    for p in G_ema.parameters(): p.requires_grad_(False)
    return G_ema

def to_01(x):           # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) * 0.5) + 0.5

# ------------------------------
# Hann blending for seamless merge
# ------------------------------
def hann2d(h: int, w: int, device) -> torch.Tensor:
    win_h = torch.hann_window(h, periodic=False, device=device).view(h, 1)
    win_w = torch.hann_window(w, periodic=False, device=device).view(1, w)
    win2d = (win_h @ win_w).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    return win2d.clamp_min(1e-3)

def blend_tiles_to_canvas(tiles_gen01: List[torch.Tensor],
                          coords: List[Tuple[int,int]],
                          canvas_hw: Tuple[int,int],
                          pad_tuple: Tuple[int,int,int,int],
                          patch: int,
                          device: torch.device) -> torch.Tensor:
    Hp, Wp = canvas_hw
    pl, pr, pt, pb = pad_tuple
    acc  = torch.zeros(1, 3, Hp, Wp, device=device)
    wsum = torch.zeros(1, 1, Hp, Wp, device=device)
    win  = hann2d(patch, patch, device=device)
    for (tile, (y, x)) in zip(tiles_gen01, coords):
        acc[..., y:y+patch, x:x+patch]  += tile * win
        wsum[..., y:y+patch, x:x+patch] += win
    merged = (acc / wsum.clamp_min(1e-6)).clamp(0, 1)
    H = Hp - (pt + pb); W = Wp - (pl + pr)
    return merged[..., pt:pt+H, pl:pl+W]

# ------------------------------
# Helper: staged band-sifting
# ------------------------------
def apply_bs_gradual(img01: torch.Tensor, effect: str, strength: float, device: torch.device):
    # s = float(strength)
    # if s == 0.0:
    #     return img01
    # s_abs = abs(s)
    # edited = img01
    # if s_abs < 2.0:
        # return band_sifting_editing(edited.squeeze(0), effect=effect, strength=s_abs).to(device).unsqueeze(0)
    # n = max(1, int(math.floor(math.log2(s_abs))))
    # for _ in range(n):
    #     edited = band_sifting_editing(edited.squeeze(0), effect=effect, strength=2.0).to(device).unsqueeze(0)
    # residual = s_abs / (2.0 ** n)
    # if residual > 0:
    #     edited = band_sifting_editing(edited.squeeze(0), effect=effect, strength=residual).to(device).unsqueeze(0)
    # return edited
    return band_sifting_editing(img01.squeeze(0), effect, strength).to(device).unsqueeze(0)

def bs_effect_for_attr(attr: str, strength: float) -> str:
    """
    Map attribute + sign(strength) to a band-sifting effect.
    Tweak as you like.
    """
    if attr in ("glossy", "matte"):
        return "shine"
    if attr in ("rough", "smooth"):
        return "rough"
    if attr in ("coarse", "fine"):
        return "rough"
    if attr == "metal":
        return "metal"  # ignore sign
    return "shine"

# ------------------------------
# CLIP text helpers
# ------------------------------
def build_clip(device):
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return clip_model, tokenizer

def encode_prompts(prompt_list: List[str], tokenizer, clip_model, device) -> torch.Tensor:
    with torch.no_grad():
        toks = tokenizer(prompt_list).to(device)
        feats = clip_model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        feat = feats.mean(dim=0, keepdim=True)
        feat = F.normalize(feat, dim=-1)  # [1, D]
    return feat

# ------------------------------
# Attr editors (your TextCond* modules)
# ------------------------------
class AttrEditors(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._load_modules()

    def _load_modules(self):
        g = torch.load("trained_dirs/textcond_delta_step02000.pt", map_location="cpu")
        self.textcond_gloss = TextCondDelta(
            d_s=g["D_s"], text_dim=768, hidden=1024,
            edit_mask=g["edit_mask_bool"].to(self.device), weight_scale=0.25
        ).to(self.device)
        self.textcond_gloss.load_state_dict(g["state_dict"])

        r = torch.load("trained_dirs/textcond_delta_step02000rough.pt", map_location="cpu")
        self.textcond_rough = TextCondDelta(
            d_s=r["D_s"], text_dim=768, hidden=1024,
            edit_mask=r["edit_mask_bool"].to(self.device), weight_scale=0.25
        ).to(self.device)
        self.textcond_rough.load_state_dict(r["state_dict"])

        d = torch.load("trained_dirs/textcond_delta_step02000depth.pt", map_location="cpu")
        self.textcond_depth = TextCondDelta(
            d_s=d["D_s"], text_dim=768, hidden=1024,
            edit_mask=d["edit_mask_bool"].to(self.device), weight_scale=0.10
        ).to(self.device)
        self.textcond_depth.load_state_dict(d["state_dict"])

        rw = torch.load("trained_dirs/textcond_delta_rust.pt", map_location="cpu")
        self.textcond_random_w = TextCondDeltaW(
            d_s=512, text_dim=768, hidden=1024,
            edit_mask=rw["edit_mask_bool"].to(self.device), weight_scale=0.01
        ).to(self.device)
        self.textcond_random_w.load_state_dict(rw["state_dict"])

        self.rough_weights = torch.load("trained_dirs/rough_weights.pt", map_location="cpu").to(self.device)
        self.rough_full    = torch.load("trained_dirs/rough_full.pt",   map_location="cpu").to(self.device)
        self.depth_vec     = torch.load("trained_dirs/depth.pt",        map_location="cpu").to(self.device)  # [D_s]
        self.coarse_full   = torch.load("trained_dirs/coarse_full.pt",  map_location="cpu").to(self.device)  # [D_s]

    def forward(self, attr: str, generator, ws: torch.Tensor, s_code: torch.Tensor,
                weights_deltas, alpha: float, text_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if attr == "glossy":
            s_dir = self.textcond_gloss(s_code, text_feat) * s_code * 0.02
            if s_dir.shape[-1] > 12287:
                s_dir[:, 12287] = s_code[:, 12287] * 0.2
            img = generator.synthesis(
                ss=s_code + s_dir * alpha, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

        elif attr == "rough":
            s_dir = self.rough_weights * 0.5 * s_code
            s_dir += self.textcond_rough(s_code, text_feat) * s_code * 0.03
            s_dir += self.rough_full * 0.5
            img = generator.synthesis(
                ss=s_code + s_dir * alpha, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

        elif attr == "depth":
            s_dir = self.depth_vec.unsqueeze(0) * 0.5
            s_dir += self.textcond_depth(s_code, text_feat) * s_code * 0.03
            img = generator.synthesis(
                ss=s_code + s_dir * alpha, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

        elif attr == "coarse":
            s_dir = self.coarse_full * 0.2
            if s_dir.ndim == 1: s_dir = s_dir.unsqueeze(0)
            s_dir[:, 0:4] = 0
            img = generator.synthesis(
                ss=s_code + s_dir * alpha, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

        elif attr == "random":
            w0 = ws.mean(dim=1)
            delta = self.textcond_random_w(w0, text_feat)
            w1 = w0 + delta * (alpha * 0.1)
            num_ws = ws.shape[1]
            ws_new = w1.unsqueeze(1).repeat(1, num_ws, 1)
            img = generator.synthesis(
                ws=ws_new, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

        else:
            img = generator.synthesis(
                ss=s_code, weights_deltas=weights_deltas,
                noise_mode='const', update_emas=False
            ).clamp(-1, 1)
            return to_01(img)

# ------------------------------
# Tile render (no edit)
# ------------------------------
@torch.no_grad()
def render_tile(generator, ws_cpu, deltas_cpu, device) -> torch.Tensor:
    deltas = [d.to(device) for d in deltas_cpu]
    ws     = ws_cpu.to(device)
    img = generator.synthesis(
        ws=ws, weights_deltas=deltas,
        noise_mode='const', update_emas=False
    ).clamp(-1, 1)
    return to_01(img)  # [1,3,256,256]

def resolve_out_path(out_path: str, tiles_dir: str, method: str, attr: str, strength: float) -> str:
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    base_dir = os.path.dirname(out_path)
    root, ext = os.path.splitext(out_path)
    if ext.lower() in valid_exts:
        os.makedirs(base_dir or ".", exist_ok=True)
        return out_path
    tiles_base = os.path.basename(os.path.normpath(tiles_dir))
    stem = tiles_base[:-6] if tiles_base.endswith("_tiles") else tiles_base
    meth = method.replace("clip-styleGAN", "clip")
    attr_part = f"_{attr}" if method == "clip-styleGAN" else ""
    strength_part = str(strength).replace(".", "p")
    fname = f"{stem}_{meth}{attr_part}_{strength_part}.png"
    out_dir = out_path if out_path else "."
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fname)

# ------------------------------
# Video helpers
# ------------------------------
def strengths_linspace(vmin: float, vmax: float, steps: int) -> List[float]:
    steps = max(2, int(steps))
    return np.linspace(vmin, vmax, steps, dtype=np.float32).tolist()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tensor01_to_bgr8(img01: torch.Tensor) -> np.ndarray:
    """img01: [1,3,H,W] in [0,1] -> HxWx3 uint8 (BGR)"""
    x = img01.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()  # HWC RGB float
    x = (x * 255.0 + 0.5).astype(np.uint8)
    bgr = x[..., ::-1]
    return bgr

def write_video_from_frames(frames_dir: str, out_mp4: str, fps: int = 24, fourcc: str = "mp4v"):
    imgs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not imgs:
        raise RuntimeError(f"No frames found in {frames_dir}")
    # Probe size
    sample = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    H, W = sample.shape[:2]
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_mp4} with fourcc={fourcc}")
    for fp in imgs:
        frame = cv2.imread(fp, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()

# ------------------------------
# Caption generator (CoCa) for per-tile text_feat
# ------------------------------
def generate_caption(img, model, device):
    img_tensor = F.interpolate(img.unsqueeze(0).to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img16 = norm_tensor.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        generated = model.generate(img16)
        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        caption = caption.split('.')[0]
    return caption

# ------------------------------
# Main driver
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True, type=str)
    ap.add_argument("--g_path",    required=True, type=str)
    ap.add_argument("--out_path",  required=True, type=str)
    ap.add_argument("--method",    required=True, choices=["bs", "clip-styleGAN", "none"])
    ap.add_argument("--attr",      type=str, default="glossy", help="Used for clip-styleGAN")
    ap.add_argument("--strength",  type=float, default=1.0)
    ap.add_argument("--device",    type=str, default=None)

    # Video options
    ap.add_argument("--make_video", action="store_true", help="Sweep strengths and export an mp4")
    ap.add_argument("--video_min", type=float, default=-10.0)
    ap.add_argument("--video_max", type=float, default=10.0)
    ap.add_argument("--video_steps", type=int, default=41)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--fourcc", type=str, default="mp4v")
    ap.add_argument("--tmp_frames_dir", type=str, default=None)
    ap.add_argument("--keep_frames", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load meta
    meta_path = os.path.join(args.tiles_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.json not found in {args.tiles_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    H, W   = meta["H"], meta["W"]
    Hp, Wp = meta["Hp"], meta["Wp"]
    pad    = meta["pad"]
    patch  = int(meta["patch"])
    coords = [tuple(c) for c in meta["coords"]]

    # Load generator
    G = load_generator(args.g_path, device)

    # Preload all tile pt paths mapped by coord
    pt_files = [fn for fn in os.listdir(args.tiles_dir) if fn.startswith("tile_") and fn.endswith(".pt")]
    pt_map: Dict[Tuple[int,int], str] = {}
    for fn in pt_files:
        path = os.path.join(args.tiles_dir, fn)
        d = torch.load(path, map_location="cpu")
        y, x = int(d["y"]), int(d["x"])
        pt_map[(y, x)] = path

    # -------------------------
    # Non-video single output
    # -------------------------
    if not args.make_video:
        if args.method in ("bs", "none"):
            # Reconstruct -> merge
            gen_tiles = []
            for (y, x) in coords:
                d = torch.load(pt_map[(y, x)], map_location="cpu")
                ws = d["ws"]; deltas_cpu = d["weight_deltas"]
                tile01 = render_tile(G, ws, deltas_cpu, device)
                gen_tiles.append(tile01)

            merged01 = blend_tiles_to_canvas(
                tiles_gen01=gen_tiles,
                coords=coords,
                canvas_hw=(Hp, Wp),
                pad_tuple=(pad["pl"], pad["pr"], pad["pt"], pad["pb"]),
                patch=patch, device=device
            )
            if args.method == "bs":
                effect = bs_effect_for_attr(args.attr, args.strength)
                merged01 = apply_bs_gradual(merged01, effect=effect, strength=args.strength, device=device)

            save_path = resolve_out_path(args.out_path, args.tiles_dir, args.method, args.attr, args.strength)
            save_image(merged01, save_path)
            print(f"[SAVED] {save_path}")
            return

        # method == clip-styleGAN
        clip_model, tokenizer = build_clip(device)
        coca_clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        coca_clip_model.half().to(device)
        editors = AttrEditors(device)

        edited_tiles = []
        for (y, x) in coords:
            d = torch.load(pt_map[(y, x)], map_location="cpu")
            ws = d["ws"]; deltas_cpu = d["weight_deltas"]
            deltas = [t.to(device) for t in deltas_cpu]
            ws_dev = ws.to(device)
            with torch.no_grad():
                s_code = G.synthesis.get_s_codes(ws_dev).to(device)
                # per-tile caption → text feat
                base_tile = G.synthesis(ws=ws_dev, weights_deltas=deltas, noise_mode="const").clamp(-1, 1).squeeze(0)
                base_tile01 = (base_tile + 1) / 2
                text = generate_caption(base_tile01, coca_clip_model, device)
                text_feat = encode_prompts([text], tokenizer, clip_model, device)

            tile01 = editors(
                attr=args.attr, generator=G, ws=ws_dev, s_code=s_code,
                weights_deltas=deltas, alpha=args.strength, text_feat=text_feat
            )
            edited_tiles.append(tile01)

        merged01 = blend_tiles_to_canvas(
            tiles_gen01=edited_tiles,
            coords=coords,
            canvas_hw=(Hp, Wp),
            pad_tuple=(pad["pl"], pad["pr"], pad["pt"], pad["pb"]),
            patch=patch, device=device
        )
        save_path = resolve_out_path(args.out_path, args.tiles_dir, args.method, args.attr, args.strength)
        save_image(merged01, save_path)
        print(f"[SAVED] {save_path}")
        return

    # -------------------------
    # Video export path
    # -------------------------
    # Where to save frames
    tiles_base = os.path.basename(os.path.normpath(args.tiles_dir))
    stem = tiles_base[:-6] if tiles_base.endswith("_tiles") else tiles_base
    frames_dir = args.tmp_frames_dir or os.path.join(os.path.dirname(args.out_path) or ".", f"{stem}_frames_tmp")
    ensure_dir(frames_dir)

    # Prepare strength schedule
    strengths = strengths_linspace(args.video_min, args.video_max, args.video_steps)

    # ====== BS / NONE (video) ======
    if args.method in ("bs", "none"):
        # reconstruct and merge once
        gen_tiles = []
        for (y, x) in coords:
            d = torch.load(pt_map[(y, x)], map_location="cpu")
            ws = d["ws"]; deltas_cpu = d["weight_deltas"]
            tile01 = render_tile(G, ws, deltas_cpu, device)
            gen_tiles.append(tile01)

        merged01_base = blend_tiles_to_canvas(
            tiles_gen01=gen_tiles,
            coords=coords,
            canvas_hw=(Hp, Wp),
            pad_tuple=(pad["pl"], pad["pr"], pad["pt"], pad["pb"]),
            patch=patch, device=device
        )

        for i, s in enumerate(strengths):
            if args.method == "bs":
                effect = bs_effect_for_attr(args.attr, s)
                frame01 = apply_bs_gradual(merged01_base, effect=effect, strength=s, device=device)
            else:
                frame01 = merged01_base
            fpath = os.path.join(frames_dir, f"frame_{i:05d}.png")
            save_image(frame01, fpath)

    # ====== CLIP-styleGAN (video) ======
    else:
        clip_model, tokenizer = build_clip(device)
        coca_clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        coca_clip_model.half().to(device)
        editors = AttrEditors(device)

        # Cache per-tile ws/deltas/s_code/text_feat once
        tile_cache = []
        for (y, x) in coords:
            d = torch.load(pt_map[(y, x)], map_location="cpu")
            ws = d["ws"]; deltas_cpu = d["weight_deltas"]
            deltas = [t.to(device) for t in deltas_cpu]
            ws_dev = ws.to(device)
            with torch.no_grad():
                s_code = G.synthesis.get_s_codes(ws_dev).to(device)
                base_tile = G.synthesis(ws=ws_dev, weights_deltas=deltas, noise_mode="const").clamp(-1, 1).squeeze(0)
                base_tile01 = (base_tile + 1) / 2
                caption = generate_caption(base_tile01, coca_clip_model, device)
                text_feat = encode_prompts([caption], tokenizer, clip_model, device)
            tile_cache.append((ws_dev, deltas, s_code, text_feat))

        for i, s in enumerate(strengths):
            edited_tiles = []
            for (ws_dev, deltas, s_code, text_feat) in tile_cache:
                tile01 = editors(
                    attr=args.attr, generator=G, ws=ws_dev, s_code=s_code,
                    weights_deltas=deltas, alpha=float(s), text_feat=text_feat
                )
                edited_tiles.append(tile01)

            merged01 = blend_tiles_to_canvas(
                tiles_gen01=edited_tiles,
                coords=coords,
                canvas_hw=(Hp, Wp),
                pad_tuple=(pad["pl"], pad["pr"], pad["pt"], pad["pb"]),
                patch=patch, device=device
            )
            fpath = os.path.join(frames_dir, f"frame_{i:05d}.png")
            save_image(merged01, fpath)

    # Encode video
    out_dir = os.path.dirname(args.out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # If out_path has an extension, switch it to .mp4; else, build a name
    root, ext = os.path.splitext(args.out_path)
    if ext.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        out_mp4 = args.out_path
    else:
        meth = args.method.replace("clip-styleGAN", "clip")
        out_mp4 = os.path.join(out_dir, f"{stem}_{meth}_{args.attr}_vmin{args.video_min}_vmax{args.video_max}_n{len(strengths)}.mp4")

    write_video_from_frames(frames_dir, out_mp4, fps=args.fps, fourcc=args.fourcc)
    print(f"[VIDEO] {out_mp4}")

    if not args.keep_frames:
        # Cleanup frame pngs
        for fp in glob.glob(os.path.join(frames_dir, "*.png")):
            try: os.remove(fp)
            except Exception: pass
        try: os.rmdir(frames_dir)
        except Exception: pass

if __name__ == "__main__":
    main()
