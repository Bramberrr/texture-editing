import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from model.networks_stylegan3 import Generator
from torchvision.utils import save_image
import numpy as np
from model.trainer import get_generator_kwargs
import dnnlib
import torch

GENERATOR_CKPT = 'G_ema_weights.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load(GENERATOR_CKPT, map_location='cpu'), strict=False)
generator.eval().to(device)

pt_root = "real_latent_9475"
output_root = "static/previews/nuur_9475"
os.makedirs(output_root, exist_ok=True)

# Helper: render preview image
def render_preview(pt_path, save_path):
    data = torch.load(pt_path, map_location=device)
    s_code = generator.synthesis.get_s_codes(data['s_code'].to(device)).to(device)
    weights_deltas = [w.to(device) if w is not None else None for w in data['delta_weights']]
    img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(img, save_path)

# Step 1: process top-level .pt files
for fname in sorted(os.listdir(pt_root)):
    fpath = os.path.join(pt_root, fname)
    if os.path.isfile(fpath) and fname.endswith('.pt'):
        out_path = os.path.join(output_root, fname + ".png")
        render_preview(fpath, out_path)

# Step 2: process .pt files in subfolders
for subdir in sorted(os.listdir(pt_root)):
    subpath = os.path.join(pt_root, subdir)
    if os.path.isdir(subpath):
        for fname in sorted(os.listdir(subpath)):
            if fname.endswith('.pt'):
                rel_path = os.path.join(subdir, fname).replace("\\", "/")
                in_path = os.path.join(pt_root, rel_path)
                out_path = os.path.join(output_root, rel_path + ".png")
                render_preview(in_path, out_path)

print("All previews generated in:", output_root)
