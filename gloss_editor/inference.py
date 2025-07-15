import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import os
from model.trainer import get_generator_kwargs
import dnnlib
import numpy as np
import torch
import torch.nn as nn
from model.s_transformation import S_Transformation
from model.band_sifting import band_sifting_editing
import open_clip
import json
import torch.nn.functional as F

OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
GENERATOR_CKPT = 'stylegan3-generator.pt'

CFG = 'stylegan3-r'
RES = 256
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Generator ---
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load(GENERATOR_CKPT, map_location='cpu'), strict=False)
generator.eval().to(device)

clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
clip_model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

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

def CLIP_editing_gloss(generator, latent_s, weights_deltas, alpha, device):
    s_dir = torch.load("dir_glossy_matte.pt").to(device) * 10
    img_clip = generator.synthesis(ss=latent_s + s_dir * alpha, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1)
    return (img_clip + 1) / 2

def CLIP_editing_rough(generator, latent_s, weights_deltas, alpha, device):
    s_dir = torch.load("dir_rough_smooth.pt").to(device) * 10
    img_clip = generator.synthesis(ss=latent_s + s_dir * alpha, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1)
    return (img_clip + 1) / 2

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
def encode_prompts(prompt_list, tokenizer, clip_model, device):
    with torch.no_grad():
        tokens = tokenizer(prompt_list).to(device)
        features = clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)
def run_inference(filename, method, strength, pt_dir="real_latent"):
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)

    s_code = generator.synthesis.get_s_codes(data['s_code'].to(device)).to(device)
    weights_deltas = [w.to(device) if w is not None else None for w in data['delta_weights']]

    base_img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode='const')
    img = (base_img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0)

    if method == "bs":
        edited = band_sifting_editing(img, effect="shine", strength=strength)
    elif method == "scurve":
        edited = S_Transformation(img, lam=remap_strengths(strength))
    elif method == "clip":
        edited = CLIP_editing_gloss(generator, s_code, weights_deltas, alpha=strength, device=device)
        edited = edited.squeeze(0)
    else:
        edited = img

    save_path = f"static/tmp/{filename}_{method}_{strength}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(edited, save_path)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    feat_pos = encode_prompts(prompts["glossy"], tokenizer, clip_model, device)
    feat_neg = encode_prompts(prompts["matte"], tokenizer, clip_model, device)
    sim_glossy = get_clip_similarity(edited, feat_pos, clip_model, device).item()
    sim_matte = get_clip_similarity(edited, feat_neg, clip_model, device).item()

    return f"/{save_path}", sim_glossy, sim_matte


def run_inference_roughness(filename, method, strength, pt_dir="real_latent"):
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)

    s_code = generator.synthesis.get_s_codes(data['s_code'].to(device)).to(device)
    weights_deltas = [w.to(device) if w is not None else None for w in data['delta_weights']]

    base_img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode='const')
    img = (base_img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0)

    if method == "bs":
        edited = band_sifting_editing(img, effect="rough", strength=strength)
    elif method == "clip":
        edited = CLIP_editing_rough(generator, s_code, weights_deltas, alpha=strength, device=device)
        edited = edited.squeeze(0)
    else:
        edited = img

    save_path = f"static/tmp/{filename}_rough_{method}_{strength}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(edited, save_path)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    feat_pos = encode_prompts(prompts["rough"], tokenizer, clip_model, device)
    feat_neg = encode_prompts(prompts["smooth"], tokenizer, clip_model, device)
    sim_rough = get_clip_similarity(edited, feat_pos, clip_model, device).item()
    sim_smooth = get_clip_similarity(edited, feat_neg, clip_model, device).item()

    return f"/{save_path}", sim_rough, sim_smooth

