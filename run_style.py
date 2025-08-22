import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import copy
import torch
from torchvision.utils import save_image
import dnnlib
from model.trainer import get_generator_kwargs
from model.restyle_e4e_encoders import ProgressiveBackboneEncoder


# ---------------------- CONFIG ---------------------- #
image_path = "static\\previews\\cherry-tomatoes_.5K.pt.png"
delta_s_path = "dir_rough_smooth_clip.pt"
out_dir = "out_dir"
num_edit_layers = 1
alpha = 10.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_CKPT = 'encoder.pt'
# ---------------------- HELPERS ---------------------- #
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    img = transforms.ToPILImage()(((tensor.squeeze(0)+1)/2).clamp(0, 1).cpu())
    img.save(path)

# ---------------------- MODEL ---------------------- #
class TextureStyleTransferNet(nn.Module):
    def __init__(self, generator, n_layers=2):
        super().__init__()
        self.layer_entries = generator.synthesis.layer_dim_table[-n_layers:]
        self.s_dims = [entry["dim"] for entry in self.layer_entries]
        self.in_proj = nn.Sequential(
            nn.Conv2d(3, self.s_dims[0], kernel_size=1),
            nn.BatchNorm2d(self.s_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.in_proj.load_state_dict(torch.load("in_proj_epoch60.pt"))
        self.layers = nn.ModuleList()
        for entry in self.layer_entries:
            self.layers.append(copy.deepcopy(getattr(generator.synthesis, entry["name"])))

    def forward(self, img, s_list):
        x = self.in_proj(img)
        for layer, s in zip(self.layers, s_list):
            x = layer(x, s=s, noise_mode='const')
        return x

# ---------------------- MAIN PIPELINE ---------------------- #
def run_edit_pipeline(G, w_encoder):
    os.makedirs(out_dir, exist_ok=True)
    G.eval().to(device)
    w_encoder.eval().to(device)

    # 1. Load image
    x_real = load_image(image_path)

    # 2. Encode to W+
    with torch.no_grad():
        ws = w_encoder(x_real)

    # 3. Get S-space codes
    s_full = G.synthesis.get_s_codes(ws)
    # 4. Load delta_s edits
    # delta_s = torch.load(delta_s_path, map_location=device)  # shape: [1, 12354]
    delta_s = torch.zeros_like(s_full).to(device)
    delta_s[0,12308]=1
    # delta_s[0,12107] = 0.6
    # delta_s[0,12192] = 0.1
    # delta_s[0,12156] = 0.1  
    # delta_s[0,12277] = 0.1 
    # delta_s[0,12200] = 0.1  

    # Apply to last `num_edit_layers` layers
    s_table = G.synthesis.layer_dim_table
    s_full = s_full + alpha * delta_s.to(device)

    # Re-split into s_list
    s_list = [s_full[:, entry['start']:entry['end']] for entry in s_table[-num_edit_layers:]]


    # 5. Build and run texture style transfer network
    style_transfer_net = TextureStyleTransferNet(G, num_edit_layers).eval().to(device)
    with torch.no_grad():
        out = style_transfer_net(x_real, s_list)

    # 6. Save result
    save_path = os.path.join(out_dir, "edited.png")
    save_image(out, save_path)
    print(f"Edited image saved to: {save_path}")

# Load Generator
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load("stylegan3-generator.pt", map_location='cpu'), strict=False)
generator.eval().to(device)

# --- Load Encoder ---
encoder = ProgressiveBackboneEncoder(num_layers=50, mode='ir', n_styles=16, opts=type('opts', (), {'input_nc': 3})()).to(device)
encoder.load_state_dict(torch.load('encoder_epoch_100.pt', map_location=device), strict=False)
encoder.eval()
run_edit_pipeline(generator, encoder)