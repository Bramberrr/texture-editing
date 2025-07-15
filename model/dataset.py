import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class RandomCropPNGDataset(Dataset):
    def __init__(self, folder_path, device):
        super().__init__()
        self.image_paths = [os.path.join(folder_path, fname)
                            for fname in os.listdir(folder_path)
                            if fname.lower().endswith(".png")]
        self.device = device
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.crop_size = 256

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = image.size

        # Ensure image is larger than crop size
        if w < self.crop_size or h < self.crop_size:
            image = transforms.Resize((max(self.crop_size, h), max(self.crop_size, w)))(image)
            w, h = image.size

        # Random crop
        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))

        # Apply transform and move to device
        img_tensor = self.base_transform(image).unsqueeze(0).to(self.device).float()  # shape: (1, 3, 256, 256)

        return img_tensor.squeeze(0)  # shape: (3, 256, 256)