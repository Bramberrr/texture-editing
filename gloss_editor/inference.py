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
import open_clip
import json
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.serialization as ts
from scipy.stats import skew
from .loss_and_scores import VGG19, STSIM_VGG, slicing_loss
from .predict import FullDirectionInterpolator, prompts_to_centroid
from typing import Optional
import matplotlib
matplotlib.use("Agg")   # use headless backend
import matplotlib.pyplot as plt
import math
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
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
clip_model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

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
        hist, _ = np.histogram(Y_np, bins=256, range=(0, 1))
        skew_val = float(skew(Y_np.flatten()))

        if save_path is not None:
            plt.figure(figsize=(4, 2))
            plt.bar(np.arange(256), hist, color="gray")
            plt.title(f"Skew: {skew_val:.3f}")
            plt.xlabel("Luminance Bin")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return hist.tolist(), round(skew_val, 3)

    
def CLIP_editing(generator, latent_s, weights_deltas, alpha, attr, device):
    if attr == "glossy":
        # s_dir = torch.zeros_like(latent_s).to(device)
        s_dir = torch.load("trained_dirs/glossy.pt").to(device) * latent_s
        # s_dir[:,12318] = latent_s[:, 12318] * 0.01
        # s_dir[:,12287] = latent_s[:, 12287] * 0.2
        # s_dir[:,12020] = latent_s[:, 12020] * 0.4 # metallic
        

    elif attr == "rough":
        s_dir = torch.zeros_like(latent_s).to(device)
        s_dir[:,11933] = latent_s[:, 11933] * 0.4 
        # s_dir = torch.load("trained_dirs/rough_weights.pt").to(device)  * 2
    elif attr == "coarse":
        s_dir = torch.load("trained_dirs/coarse_full.pt").to(device) * 0.2
        s_dir[0:4]=0
    else:
        attr1, attr2 = attr_pairs.get(attr, ("glossy", "matte"))
        with torch.no_grad():
            feat1 = prompts_to_centroid(tokenizer, clip_model, prompts[attr1], device)
            feat2 = prompts_to_centroid(tokenizer, clip_model, prompts[attr2], device)
            img = (generator.synthesis(ss=latent_s, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1) + 1) / 2
            f = get_clip_feat(img, clip_model, device)
            p = f@feat1.T
            n =  f@feat2.T
            ext = F.normalize(torch.cat([f, p,n] , dim=1))
        if attr == "depth":
            s_dir = torch.load("trained_dirs/depth.pt").to(device) * 0.5
            # if alpha > 0:
            #     pack = torch.load('trained_dirs/full_cluster_pack_period_regular.pt', map_location='cpu')
            #     interp = FullDirectionInterpolator(pack['cluster']['medoid_ext_feats'], pack['directions_full'], pack['s_offsets'], device=device, tau=float(pack['tau']))
            #     interp.load_state_dict(torch.load('trained_dirs/full_interpolator_period_regular.pt', map_location='cpu'))
            #     s_dir = interp(ext, zero_layers=[0,13,14,15]) *0.5
            # else: 
            #     pack = torch.load('trained_dirs/full_cluster_pack_pattern_random.pt', map_location='cpu')
            #     interp = FullDirectionInterpolator(pack['cluster']['medoid_ext_feats'], pack['directions_full'], pack['s_offsets'], device=device, tau=float(pack['tau']))
            #     interp.load_state_dict(torch.load('trained_dirs/full_interpolator_pattern_random.pt', map_location='cpu'))
            #     s_dir = -interp(ext, zero_layers=[0,1,2,3,4,5,6,7,8,11,12,13,14,15])
            # # s_dir = torch.zeros_like(latent_s).to(device)
            # # s_dir[:,2602] = latent_s[:, 2602]
            # # s_dir[:,4491] = latent_s[:, 4491]
            # # s_dir[:,3603] = latent_s[:, 3603]
        elif attr == "random":
            # s_dir = torch.zeros_like(latent_s).to(device)
            # s_dir[:,12207] = latent_s[:, 12207]
            pack = torch.load('trained_dirs/full_cluster_pack_color_random.pt', map_location='cpu')
            interp = FullDirectionInterpolator(pack['cluster']['medoid_ext_feats'], pack['directions_full'], pack['s_offsets'], device=device, tau=float(pack['tau']))
            interp.load_state_dict(torch.load('trained_dirs/full_interpolator_color_random.pt', map_location='cpu'))
            s_dir = interp(ext, zero_layers=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15]) 
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

    if strength <= 0:
        return img

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

    # === Dispatch editing ===
    if method == "bs":
        effect = "shine" if attr in ["glossy", "matte"] else "rough"
        edited = apply_bs_gradual(img, effect=effect, strength=strength, device=device)
    elif method == "scurve":
        edited = S_Transformation(img, lam=remap_strengths(strength)).to(device)
    elif method == "clip":
        edited = CLIP_editing(generator, s_code, weights_deltas, strength, attr, device)
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
    nat = is_natural(edited.unsqueeze(0))[0]
    hist, skew_val = compute_luminance_histogram_and_skew(edited, save_path=hist_path)

    return (
        f"/{img_path}",
        round(sim_attr1, 3),
        round(sim_attr2, 3),
        round(sim_img, 3),
        round(stsim, 3),
        round(sw, 3),
        nat,
        f"/{hist_path}",
        skew_val
    )