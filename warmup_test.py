import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from model.trainer import get_generator_kwargs
import dnnlib

GENERATOR_CKPT = 'stylegan3-generator.pt'
# GENERATOR_CKPT = "G_ema_weights.pt"
SCALING_FACTOR = 1  # or 4, depending on how dense you want generated activations
CFG = 'stylegan3-r'
RES = 256
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

# --- Load Generator ---
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load(GENERATOR_CKPT, map_location='cpu'), strict=False)
generator.eval().to(device)
z = torch.randn([1, generator.z_dim]).cuda()
c = None
img = generator(z, c, noise_mode='const')
print("Warm-up complete.")