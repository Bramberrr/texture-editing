import torch
import torch.nn.functional as F
import json
import argparse
import os
from tqdm import tqdm
import kornia
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import open_clip
import dnnlib

# Define OpenAI CLIP dataset mean and std
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP image preprocessing transform
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to CLIP's input size
    transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
])

def get_generator_kwargs(cfg='stylegan3-r', resolution=256, cbase=32768, cmax=512, map_depth=None, seed=0, device=torch.device('cuda')):
    G_kwargs = dnnlib.EasyDict(
        class_name='model.networks_stylegan3.Generator',
        z_dim=512,
        w_dim=512,
        c_dim=0,
        img_resolution=resolution,
        img_channels=3,
        mapping_kwargs=dnnlib.EasyDict(num_layers=2),
        conv_kernel=1,
        channel_base=cbase * 2,
        channel_max=cmax * 2,
        use_radial_filters=True,
        magnitude_ema_beta=0.5 ** (32 / (20 * 1e3))
    )
    return G_kwargs

class AttributeTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')

        with open(args.prompt_json, 'r') as f:
            self.prompts = json.load(f)

        self.attr_pairs = [
            ('glossy', 'matte'),
            ('rough', 'smooth'),
            ('high contrast', 'low contrast'),
            ('regular', 'stochastic'),
            ('large scale', 'small scale'),
            ('grayscale', 'saturated color')
        ]
        self.generator = self.load_generator(args.generator_ckpt)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained
        )
        self.clip_model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(args.clip_model)

    def load_generator(self, ckpt_path):
        G_kwargs = get_generator_kwargs(device=self.device)
        G = dnnlib.util.construct_class_by_name(**G_kwargs)
        G.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        return G.eval().to(self.device)

    def encode_prompts(self, prompt_list):
        with torch.no_grad():
            tokens = self.tokenizer(prompt_list).to(self.device)
            features = self.clip_model.encode_text(tokens)
            return F.normalize(features, dim=-1)

    def train(self, attr_pair_names):
        z_samples = torch.randn(self.args.total_samples, self.generator.z_dim).to(self.device)
        w_plus_codes = self.generator.mapping(z_samples, None, truncation_psi=1.0)
        s_codes = self.generator.synthesis.get_s_codes(w_plus_codes)

        layer_dims = [4] + [1024] * 10 + [724, 512, 362, 256, 256]
        s_offsets = torch.cumsum(torch.tensor([0] + layer_dims), dim=0)

        for attr_pair in attr_pair_names:
            print(f"\nTraining for attribute pair: {attr_pair}")
            a_pos, a_neg = attr_pair[0], attr_pair[1]
            positive_feats = self.encode_prompts(self.prompts[a_pos])
            negative_feats = self.encode_prompts(self.prompts[a_neg])

            delta_s = self.optimize_direction_multi_layer(
                s_codes, positive_feats, negative_feats,
                s_offsets, layer_dims
            )

            save_path = os.path.join(self.args.experiment_folder, f"delta_s_{a_pos}-{a_neg}.pt")
            os.makedirs(self.args.experiment_folder, exist_ok=True)
            torch.save(delta_s, save_path)
            print(f"Saved delta_s to {save_path}")

    def optimize_direction_multi_layer(self, s_codes, feat_pos, feat_neg, s_offsets, layer_dims):
        layers_to_optimize = list(range(9, 16))
        delta_s_slices = torch.nn.ParameterList()
        slice_offsets = []
        for layer in layers_to_optimize:
            start, end = s_offsets[layer], s_offsets[layer+1]
            slice_offsets.append((start, end))
            param = torch.nn.Parameter(0.01 * torch.randn(end - start, device=self.device))
            delta_s_slices.append(param)

        optimizer = torch.optim.Adam(delta_s_slices, lr=self.args.lr)

        for epoch in range(self.args.epochs):
            step_bar = tqdm(range(0, len(s_codes), self.args.batch_size), desc=f"Epoch {epoch+1}")
            for i in step_bar:
                batch = s_codes[i:i+self.args.batch_size].detach()
                delta_s_full = torch.zeros_like(batch[0], device=self.device)
                for param, (start, end) in zip(delta_s_slices, slice_offsets):
                    delta_s_full[start:end] = torch.tanh(param)

                imgs_base = self.generator.synthesis(ss=batch, noise_mode='const')
                imgs_base = (imgs_base.clamp(-1, 1) + 1) / 2
                mean_rgb_base = imgs_base.mean(dim=[2,3])
                lab_base = kornia.color.rgb_to_lab(imgs_base)
                mean_lab_base = lab_base.mean(dim=[2,3]) 
                imgs_base = clip_transform(imgs_base)

                base_feats = self.clip_model.encode_image(imgs_base)
                base_feats = F.normalize(base_feats, dim=-1)

                batch_mod = batch + delta_s_full.unsqueeze(0)
                imgs_mod = self.generator.synthesis(ss=batch_mod, noise_mode='const')
                imgs_mod = (imgs_mod.clamp(-1, 1) + 1) / 2
                mean_rgb_mod  = imgs_mod.mean(dim=[2,3])   # [B, 3]
                lab_mod  = kornia.color.rgb_to_lab(imgs_mod)
                mean_lab_mod  = lab_mod.mean(dim=[2,3])
                imgs_mod = clip_transform(imgs_mod)

                img_feats = self.clip_model.encode_image(imgs_mod)
                img_feats = F.normalize(img_feats, dim=-1)

                text_dir = F.normalize(feat_pos.mean(dim=0, keepdim=True) - feat_neg.mean(dim=0, keepdim=True), dim=-1)
                img_dir = F.normalize(img_feats - base_feats, dim=-1)

                dir_sim = (img_dir * text_dir).sum(dim=-1).mean()
                loss_dir = 1.0 - dir_sim
                reg = sum(param.norm() for param in delta_s_slices)
                loss_rgb = F.mse_loss(mean_rgb_mod, mean_rgb_base)
                loss_lab = F.mse_loss(mean_lab_mod, mean_lab_base)
                loss = loss_dir + loss_rgb +loss_lab + 1e-3 * reg

                sim_pos = (img_feats @ feat_pos.T).mean()
                sim_neg = (img_feats @ feat_neg.T).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_bar.set_postfix({"Loss": f"{loss.item():.4f}",
                                    "Pos": f"{sim_pos.item():.4f}",
                                    "Neg": f"{sim_neg.item():.4f}",
                                        "Reg": f"{reg.item():.4f}"})

        final_delta_s = torch.zeros_like(s_codes[0])
        for param, (start, end) in zip(delta_s_slices, slice_offsets):
            final_delta_s[start:end] = torch.tanh(param).detach().cpu()

        return final_delta_s