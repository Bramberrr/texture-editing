import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision.utils import save_image
import dnnlib
from model.trainer import get_generator_kwargs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load("stylegan3-generator.pt", map_location='cpu'), strict=False)
generator.eval().to(device)

os.makedirs("real_latent/generated", exist_ok=True)
os.makedirs("static/previews/generated", exist_ok=True)

N = 10000  # number of synthetic textures to generate

for i in range(N):
    z = torch.randn(1, generator.z_dim).to(device)
    w_plus = generator.mapping(z, None, truncation_psi=1.0)
    s_code = generator.synthesis.get_s_codes(w_plus)

    img = generator.synthesis(ss=s_code, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2

    save_image(img, f"static/previews/generated/{i}.pt.png")
    torch.save(s_code.detach().cpu(), f"real_latent/generated/{i}.pt")

print("Generated synthetic dataset.")
