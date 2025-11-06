import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import os
from model.trainer import get_generator_kwargs
import dnnlib
import torch.nn as nn
from model.s_transformation import S_Transformation
from model.band_sifting import band_sifting_editing
import transformers
import open_clip
import json
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.serialization as ts
from scipy.stats import skew
from .loss_and_scores import VGG19, STSIM_VGG, slicing_loss
from .predict import FullDirectionInterpolator, prompts_to_centroid, TextCondDelta, TextCondDeltaW
from typing import Optional
import matplotlib
matplotlib.use("Agg")   # use headless backend
import matplotlib.pyplot as plt
import math
from typing import Callable, Dict, Optional, Tuple, List
import torch.fft as fft
from scipy.stats import kurtosis
import logging
logger = logging.getLogger(__name__)
# ------------------------------
# Settings and Initialization
# ------------------------------

OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
GENERATOR_CKPT = 'G_ema_weights.pt'
# GENERATOR_CKPT = "stylegan3-bs-augment.pt"
CFG = 'stylegan3-r'
RES = 256
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_S_ANALYSIS = {
    "latent_s": [],   # stores latent_s
    "s_dir": []       # stores predicted direction
}
# --- Load Generator ---
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load(GENERATOR_CKPT, map_location='cpu'), strict=False)
generator.eval().to(device)
z = torch.randn([1, generator.z_dim]).cuda()
c = None
img = generator(z, c, noise_mode='const')
print("Warm-up complete.")

clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
clip_model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
coca_clip_model, _, _ = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k")
coca_clip_model.half().to(device)
vgg = VGG19()
vgg.load_state_dict(torch.load("./vgg19.pth"))
vgg.eval().to(device)
stsim_loss = STSIM_VGG([5900,10],grayscale=False).to(device).double()

#is natural predictor
ts.add_safe_globals([np.core.multiarray.scalar])
ckpt = torch.load("trained_dirs/best.pt",map_location="cpu",weights_only=False)
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
in_features = model.classifier[2].in_features
model.classifier[2] = torch.nn.Linear(in_features, len(ckpt["classes"]))
model.load_state_dict(ckpt["model"])
model.eval().to("cuda")
IMAGENET_MEAN = torch.tensor(OPENAI_DATASET_MEAN).view(1,3,1,1).to("cuda")
IMAGENET_STD  = torch.tensor(OPENAI_DATASET_STD).view(1,3,1,1).to("cuda")

# ------------------------------
# Helper Functions
# ------------------------------
def render_preview(filename, save_path):
    # Just copy the cached preview image from static
    src_path = os.path.join("static", "previews", f"{filename}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(src_path):
        Image.open(src_path).save(save_path)
    else:
        raise FileNotFoundError(f"Preview image not found: {src_path}")

def remap_strengths(signed_scalar):
    return np.exp(signed_scalar) if signed_scalar < 0 else signed_scalar

def preprocess_batch(x):
    if x.min() < 0:  # assume [-1,1]
        x = (x + 1) / 2.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD

@torch.no_grad()
def is_natural(x, threshold=0.5):
    """
    x: [B,3,H,W] tensor
    returns: list[bool] for each image
    """
    x = preprocess_batch(x).to("cuda")
    logits = model(x)                # [B,2]
    probs = F.softmax(logits, dim=-1)
    # assume class order = ["natural", "unnatural"]
    p_natural = probs[:, ckpt["classes"].index("natural")]
    return (p_natural >= threshold).cpu().tolist()

def get_clip_similarity(img_tensor, feat, clip_model, device):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img_feat = clip_model.encode_image(norm_tensor)
    img_feat = F.normalize(img_feat, dim=-1)
    return (img_feat @ feat.T).mean(dim=1)

def get_clip_feat(img_tensor, clip_model, device):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img_feat = clip_model.encode_image(norm_tensor)
    img_feat = F.normalize(img_feat, dim=-1)
    return img_feat
def encode_prompts(prompt_list, tokenizer, clip_model, device):
    with torch.no_grad():
        tokens = tokenizer(prompt_list).to(device)
        features = clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)
with open("prompts.json", "r") as f:
    prompts = json.load(f)
feat_dict = {k: encode_prompts(v, tokenizer, clip_model, device) for k, v in prompts.items()}
attr_pairs = {
    "glossy": ("glossy", "matte"),
    "matte": ("glossy", "matte"),
    "rough": ("rough", "smooth"),
    "smooth": ("rough", "smooth"),
    "depth": ("deep", "shallow"),
    "random": ("random", "regular"),
    "coarse": ("coarse", "fine"),
    "fine": ("fine", "coarse"),
}
def compute_luminance_histogram_and_skew(img_tensor, save_path=None):
    """
    img_tensor: [3, H, W] or [1, 3, H, W] in [0, 1]
    Returns:
        - hist: list of 256 ints
        - skew_val: float
        - if save_path is provided, saves a histogram bar chart there
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    with torch.no_grad():
        R, G, B = img_tensor[0], img_tensor[1], img_tensor[2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Y_np = Y.cpu().numpy()
        # 256-bin histogram over [0,1]
        hist, _ = np.histogram(Y_np, bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.int64)

        # Skewness on histogram counts
        # Handle degenerate cases (all zeros except a few) to avoid NaN
        def safe_skew(x):
            if np.all(x == x[0]):  # constant vector
                return 0.0
            val = skew(x, bias=False)
            if np.isnan(val):
                return 0.0
            return float(val)

        skew_full = safe_skew(hist)
        hist_inner = hist[1:255]  # exclude bins 0 and 255
        skew_inner = safe_skew(hist_inner)

        # Optional plot
        if save_path is not None:
            plt.figure(figsize=(5.0, 2.4))
            plt.bar(np.arange(256), hist)
            plt.title(f"Skew(full)={skew_full:.3f} | Skew(1..254)={skew_inner:.3f}")
            plt.xlabel("Luminance bin (0..255)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(save_path, dpi=120)
            plt.close()

    return hist.tolist(), round(skew_full, 3), round(skew_inner, 3)
label_path = "captions_labels.json"
labels_dict = {}
if os.path.exists(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

caption_path = "captions.json"
captions_dict = {}
if os.path.exists(caption_path):
    with open(caption_path, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

gloss_data = torch.load("trained_dirs/textcond_delta_step02000.pt")
textcond_gloss = TextCondDelta(d_s=gloss_data["D_s"], text_dim=768, hidden=1024, edit_mask=gloss_data["edit_mask_bool"].to(device), weight_scale=0.25).to(device)
textcond_gloss.load_state_dict(gloss_data["state_dict"])

rough_data = torch.load("trained_dirs/textcond_delta_step02000rough.pt")
textcond_rough = TextCondDelta(d_s=rough_data["D_s"], text_dim=768, hidden=1024, edit_mask=rough_data["edit_mask_bool"].to(device), weight_scale=0.25).to(device)
textcond_rough.load_state_dict(rough_data["state_dict"])

# random_data = torch.load("trained_dirs/textcond_delta_step02000random.pt")
# textcond_random = TextCondDelta(d_s=random_data["D_s"], text_dim=768, hidden=1024, edit_mask=random_data["edit_mask_bool"].to(device), weight_scale=0.1).to(device)
# textcond_random.load_state_dict(random_data["state_dict"])

depth_data = torch.load("trained_dirs/textcond_delta_step02000depth.pt")
textcond_depth = TextCondDelta(d_s=depth_data["D_s"], text_dim=768, hidden=1024, edit_mask=depth_data["edit_mask_bool"].to(device), weight_scale=0.1).to(device)
textcond_depth.load_state_dict(depth_data["state_dict"])

moss_data = torch.load("trained_dirs/textcond_delta_moss.pt")
textcond_moss = TextCondDeltaW(d_s=512, text_dim=768, hidden=1024, edit_mask=moss_data["edit_mask_bool"].to(device), weight_scale=0.05).to(device)
textcond_moss.load_state_dict(moss_data["state_dict"])

rust_data = torch.load("trained_dirs/textcond_delta_rust.pt")
textcond_rust = TextCondDeltaW(d_s=512, text_dim=768, hidden=1024, edit_mask= rust_data["edit_mask_bool"].to(device), weight_scale=0.03).to(device)
textcond_rust.load_state_dict(rust_data["state_dict"])

random_data = torch.load("trained_dirs/textcond_delta_rust.pt")
textcond_random = TextCondDeltaW(d_s=512, text_dim=768, hidden=1024, edit_mask= random_data["edit_mask_bool"].to(device), weight_scale=0.01).to(device)
textcond_random.load_state_dict(random_data["state_dict"])

def CLIP_editing(generator, latent_s, weights_deltas, alpha, attr, device, text_feat=None, ws=None):
    if attr == "glossy":
        s_dir = torch.zeros_like(latent_s).to(device)
        s_dir = textcond_gloss(latent_s,text_feat) * latent_s * 0.01
        s_dir[:,12287] = latent_s[:, 12287] * 0.2
        
        # s_dir[:,12318] = latent_s[:, 12318] * -0.05
        GLOBAL_S_ANALYSIS["latent_s"].append(latent_s.detach().cpu())
        GLOBAL_S_ANALYSIS["s_dir"].append(s_dir.detach().cpu())

        # ---- analyze variation in real time ----
        s_tensor = torch.stack([s.flatten() for s in GLOBAL_S_ANALYSIS["latent_s"]], dim=0)  # [N, D]
        d_tensor = torch.stack([d.flatten() for d in GLOBAL_S_ANALYSIS["s_dir"]], dim=0)     # [N, D]


        with torch.no_grad():
            # mean/std for direction magnitude and cosine similarity
            dir_norms = d_tensor.norm(dim=1)
            mean_norm = dir_norms.mean().item()
            std_norm = dir_norms.std().item()

            # cosine similarity between all pairs of directions
            d_flat = F.normalize(d_tensor, dim=1)
            sim_matrix = torch.matmul(d_flat, d_flat.T)
            avg_sim = sim_matrix.mean().item()
            offdiag_sim = (sim_matrix.fill_diagonal_(0).sum() / (sim_matrix.numel() - len(sim_matrix))).item()

        print(f"[Analysis] glossy s_dir stats (N={len(d_tensor)}): "
              f"norm μ={mean_norm:.4f}, σ={std_norm:.4f}, "
              f"avg cos={avg_sim:.4f}, offdiag cos={offdiag_sim:.4f}")
        # s_dir[:,12020] = latent_s[:, 12020] * 0.4 # metallic
    elif attr == "rough":
        # s_dir = torch.zeros_like(latent_s).to(device)
        s_dir = torch.load("trained_dirs/rough_weights.pt").to(device) * 0.5 * latent_s
        s_dir += textcond_rough(latent_s,text_feat) * latent_s * 0.03
        s_dir += torch.load("trained_dirs/rough_full.pt").to(device) * 0.5
        
        # s_dir[:,12308] = latent_s[:, 12308] * 0.01
    elif attr == "coarse":
        s_dir = torch.load("trained_dirs/coarse_full.pt").to(device) * 0.2
        s_dir[0:4]=0
    elif attr == "random":
        # s_dir = textcond_random(latent_s,text_feat) * latent_s * 0.1
        w0 = ws.mean(dim=1)
        delta = textcond_random(w0, text_feat)   # [B, D]
        w1 = w0 + delta.unsqueeze(1) * alpha * 0.1
        ws = w1.repeat(1, 16, 1)
        img = generator.synthesis(ws=ws, weights_deltas=weights_deltas, noise_mode='const', update_emas=False).clamp(-1, 1).squeeze(0)
        return (img + 1) / 2
    elif attr == "depth":
        s_dir = torch.load("trained_dirs/depth.pt").unsqueeze(0).to(device) * 0.5
        s_dir += textcond_depth(latent_s,text_feat) * latent_s * 0.03


    img_clip = generator.synthesis(ss=latent_s + s_dir * alpha, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1).squeeze(0)
    return (img_clip + 1) / 2

def apply_bs_gradual(img, effect: str, strength: float, device):
    """
    Apply band_sifting_editing in staged increments:
      - for strength in (2^n, 2^(n+1)) -> apply n passes of 2.0, then one pass of strength / 2^n
      - for strength < 2.0 -> single pass of `strength`
    """
    strength = float(strength)
    edited = img

    if strength < 0:
        return band_sifting_editing(edited, effect=effect, strength=strength).to(device)


    if strength < 2.0:
        return band_sifting_editing(edited, effect=effect, strength=strength).to(device)

    # strength >= 2
    n = int(math.floor(math.log2(strength)))           # number of 2.0 passes
    n = max(1, n)                                      # ensure at least one pass when >= 2

    # n passes at strength=2.0
    for _ in range(n):
        edited = band_sifting_editing(edited, effect=effect, strength=2.0).to(device)

    # final residual pass
    residual = strength / (2.0 ** n)                   # e.g., s/2, s/4, s/8, ...
    if residual > 0:
        edited = band_sifting_editing(edited, effect=effect, strength=residual).to(device)

    return edited

def generate_caption(img):
    img_tensor = F.interpolate(img.unsqueeze(0).to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img16 = norm_tensor.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        generated = coca_clip_model.generate(img16)
        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        caption = caption.split('.')[0]
    return caption

@torch.no_grad()
def run_inference(filename, method, strength, pt_dir="real_latent", attr="glossy"):
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)

    if isinstance(data, dict) and "s_code" in data:
        s_code = generator.synthesis.get_s_codes(data["s_code"].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data["delta_weights"]]
    elif isinstance(data, dict) and "ws" in data:
        s_code = generator.synthesis.get_s_codes(data["ws"].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data["weight_deltas"]]
    else:
        s_code = data.to(device)
        weights_deltas = None

    base_img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode="const").clamp(-1, 1)
    img = (base_img + 1) / 2
    img = img.squeeze(0)
    img_feat = get_clip_feat(img, clip_model, device)

    labels = labels_dict.get(filename+".png", [])
    text = "A picture of " + ", ".join([str(x).strip() for x in labels]) + " texture."
    text_feat = encode_prompts(text,tokenizer, clip_model, device)
    # === Dispatch editing ===
    if method == "bs":
        effect = "shine" if attr in ["glossy", "matte"] else "rough"
        edited = apply_bs_gradual(img, effect=effect, strength=strength, device=device)
    elif method == "bs-metal":
        edited = apply_bs_gradual(img, effect="metal", strength=strength, device=device)
    elif method == "clip-styleGAN":
        w = data["ws"].to(device) if isinstance(data, dict) and "ws" in data else None
        edited = CLIP_editing(generator, s_code, weights_deltas, strength, attr, device, text_feat, w)
    else:
        edited = img

    # === Save outputs ===
    save_root = f"static/tmp/{filename}_{method}_{attr or 'none'}_{strength}"
    img_path = f"{save_root}.png"
    hist_path = f"{save_root}_hist.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    save_image(edited, img_path)

    attr1, attr2 = attr_pairs.get(attr, ("glossy", "matte"))

    sim_attr1 = get_clip_similarity(edited, feat_dict[attr1], clip_model, device).item()
    sim_attr2 = get_clip_similarity(edited, feat_dict[attr2], clip_model, device).item()
    sim_img = get_clip_similarity(edited, img_feat, clip_model, device).item()
    stsim = stsim_loss(img.unsqueeze(0).double(), edited.unsqueeze(0).double()).item()
    sw = slicing_loss(vgg(img.unsqueeze(0)), vgg(edited.unsqueeze(0))).item()

    hist, skew_full, skew_inner = compute_luminance_histogram_and_skew(edited, save_path=hist_path)
    caption_base = captions_dict.get(filename+".png", [])
    caption = generate_caption(edited)

    sim_nat = get_clip_similarity(img, feat_dict["natural"], clip_model, device).item()
    sim_un = get_clip_similarity(img, feat_dict["unnatural"], clip_model, device).item()
    print(sim_nat, sim_un)
    offset = sim_nat - sim_un - 0.01 if sim_nat -0.01 < sim_un else 0.0
    artifact_ok, guard_info = artifact_check_tensor(edited, clip_model, tokenizer, device, caption_base, caption)
    nat = bool(artifact_ok)
    # nat = True

    # log + persist guard info if it failed (shows in Django runserver logs; also saved as JSON)
    if True:
        logger.warning(
            "ARTIFACT_GUARD_FAIL file=%s method=%s strength=%s attr=%s guard=%s",
            filename, method, str(strength), attr, json.dumps(guard_info)
        )
    

    return (
        f"/{img_path}",
        round(sim_attr1, 3),
        round(sim_attr2, 3),
        round(sim_img, 3),
        round(stsim, 3),
        round(sw, 3),
        nat,
        f"/{hist_path}",
        skew_full,
        caption
    )

# =========================================
# artifact_check (tensor-only, minimal)
# =========================================
@torch.no_grad()
def artifact_check_tensor(
    img_tensor,
    clip_model=clip_model,
    tokenizer=open_clip.get_tokenizer("ViT-L-14"),
    device=device,
    caption_base=None,
    caption = None,
    offset = 0.0
):
    """
    CLIP-based classifier for unnatural artifacts.
    Adds semantic-drift (caption similarity) detection.
    Returns (ok: bool, info: dict)
    """

    with torch.no_grad():
        sim_nat = get_clip_similarity(img_tensor, feat_dict["natural"], clip_model, device).item()
        sim_un = get_clip_similarity(img_tensor, feat_dict["unnatural"], clip_model, device).item()
        score = sim_nat - sim_un - offset
    # --- interpret score
    ok = score > 0
    confidence = abs(score)

    # === Caption semantic similarity ===
    if caption_base is not None:
        # use same tokenizer
        tok1 = tokenizer([caption_base]).to(device)
        tok2 = tokenizer([caption]).to(device)
        txt1 = F.normalize(clip_model.encode_text(tok1), dim=-1)
        txt2 = F.normalize(clip_model.encode_text(tok2), dim=-1)
        caption_sim = (txt1 @ txt2.T).item()
    else:
        caption_sim = 1.0  # assume perfect match if no base caption

    # --- combine signals
    sem_drift = caption_sim < 0.7   # threshold for semantic change
    if sem_drift:
        ok = False

    # --- fallback when uncertain
    if confidence < 0.003:
        fallback_ok, fallback_info = _legacy_artifact_guard(img_tensor)
        ok = fallback_ok and not sem_drift
    else:
        fallback_info = None

    info = {
        "guard": {
            "score": round(float(score), 4),
            "sim_natural": round(float(sim_nat), 4),
            "sim_unnatural": round(float(sim_un), 4),
            "caption_similarity": round(float(caption_sim), 4),
            "semantic_drift": sem_drift,
            "confidence": round(float(confidence), 4),
            "status": "ok" if ok else "artifact_detected",
            "method": "CLIP-naturalness+caption",
            "fallback_used": fallback_info is not None
        }
    }
    if fallback_info:
        info["fallback"] = fallback_info

    return ok, info

def _legacy_artifact_guard(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    gray = (0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]).float().clamp(0,1)
    H, W = gray.shape
    device = gray.device
    g = gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # 1) Clipping
    y255 = (gray * 255.0).round().clamp(0, 255).long().view(-1)
    hist = torch.bincount(y255, minlength=256).float()
    total = float(hist.sum().item() + 1e-8)
    frac255 = float(hist[255].item() / total)
    h254, h255 = float(hist[254].item()), float(hist[255].item())
    jump_ratio = float(h255 / (h254 + 1e-8))


    score = 0.15 * jump_ratio
    ok = score < 1.0
    d = {
        "guard": {
            "frac255": frac255,
            "jump_ratio": jump_ratio,
            "score": score,
            "status": "ok" if ok else "artifact_detected"
        }
    }
    return ok, d

@torch.no_grad()
def detect_safe_ranges_from_file(filename: str, pt_dir: str = "real_latent") -> Dict[str, float]:
    """
    Detects safe editing strength ranges for 'bs' and 'clip' on a given latent file.
    Returns dict like: {"bs": safe_bs_alpha, "clip": safe_clip_alpha}
    """
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)
    text = labels_dict.get(filename+".png", [])
    text_feat = encode_prompts(text,tokenizer, clip_model, device)

    # --- Load latent & reconstruct baseline image ---
    if isinstance(data, dict) and "s_code" in data:
        s_code = generator.synthesis.get_s_codes(data["s_code"].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data["delta_weights"]]
    elif isinstance(data, dict) and "ws" in data:
        s_code = generator.synthesis.get_s_codes(data["ws"].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data["weight_deltas"]]
    else:
        s_code = data.to(device)
        weights_deltas = None

    base_img = generator.synthesis(
        ss=s_code, weights_deltas=weights_deltas, noise_mode="const"
    ).clamp(-1, 1)
    img = (base_img + 1) / 2
    img = img.squeeze(0)

    # --- Probe safe alphas ---
    safe_dict = {"bs": 0.0,"metal": 0.0, "clip": 0.0}
    # alphas = [i * 0.1 for i in range(1, 101)]

    # # BS safe alpha
    # def gen_bs(a):
    #     return apply_bs_gradual(img, effect="shine", strength=a, device=device)
    # def gen_metal(a):
    #     return apply_bs_gradual(img, effect="metal", strength=a, device=device)

    # for a in alphas:
    #     ok, _, _ = artifact_check_tensor(gen_bs(a), clip_model, tokenizer, device, baseline_tensor=img)
    #     if not ok:
    #         break
    #     safe_dict["bs"] = round(a,1)
    # for a in alphas:
    #     ok, _, _ = artifact_check_tensor(gen_metal(a), clip_model, tokenizer, device, baseline_tensor=img)
    #     if not ok:
    #         break
    #     safe_dict["metal"] = round(a,1)

    # # CLIP safe alpha (glossy direction)
    # def gen_clip(a):
    #     return CLIP_editing(generator, s_code, weights_deltas, a, "glossy", device, text_feat)

    # for a in alphas:
    #     ok, _, _ = artifact_check_tensor(gen_clip(a), clip_model, tokenizer, device, baseline_tensor=img)
    #     if not ok:
    #         break
    #     safe_dict["clip"] = round(a,1)

    return safe_dict
