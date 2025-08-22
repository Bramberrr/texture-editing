import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === Paths ===
input_folder = 'D:\\projects\\texture_annotation\\annotation\\static\\NUUR-Textures6K'
output_folder = 'static/previews/nuur'
os.makedirs(output_folder, exist_ok=True)

# === Transform ===
transform = transforms.Compose([
    transforms.CenterCrop(256),
])

# === Process all .png files ===
for fname in tqdm(os.listdir(input_folder)):
    if fname.lower().endswith('.png'):
        path = os.path.join(input_folder, fname)
        img = Image.open(path).convert('RGB')

        cropped = transform(img)
        cropped.save(os.path.join(output_folder, fname))
