import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import torch.nn.functional as F
import os
import json
layer_dims = [4] + [1024] * 10 + [724, 512, 362, 256, 256]
s_offsets = torch.cumsum(torch.tensor([0] + layer_dims), dim=0) # for masking

def get_slice(s_offsets: torch.Tensor, L: int):
    """Return (start, end) indices for S-layer L."""
    return int(s_offsets[L].item()), int(s_offsets[L+1].item())

class FullDirectionInterpolator(nn.Module):
    def __init__(self, medoid_ext_feats, directions_full,
                 s_offsets, device, tau = 0.07):
        super().__init__()
        self.register_buffer("medoid_ext_feats", F.normalize(medoid_ext_feats, dim=-1).to(device))
        self.register_buffer("directions_full", directions_full.to(device))   # [K, D_s]
        self.register_buffer("s_offsets", s_offsets.to(device))               # [17]
        self.tau = tau

    def _zero_layers(self, s_dir: torch.Tensor, zero_layers: Optional[List[int]]):
        if not zero_layers:
            return s_dir
        s_dir = s_dir.clone()
        for L in zero_layers:
            s = int(self.s_offsets[L].item())
            e = int(self.s_offsets[L+1].item())
            s_dir[..., s:e] = 0.0
        return s_dir

    def forward(self, ext_feat: torch.Tensor, zero_layers: Optional[List[int]] = None, normalize: bool = False):
        if ext_feat.dim() == 1:
            ext_feat = ext_feat.unsqueeze(0)
        ext_feat = F.normalize(ext_feat, dim=-1)
        sims = ext_feat @ self.medoid_ext_feats.t()         # [B, K]
        w = F.softmax(sims / self.tau, dim=-1)              # [B, K]
        s_dir = w @ self.directions_full                    # [B, D_s]
        s_dir = self._zero_layers(s_dir, zero_layers)
        if normalize:
            s_dir = s_dir / (s_dir.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return s_dir
    
def fuse_layer_dirs(
    layer_dirs: Dict[int, torch.Tensor],
    layer_dims: list,
    s_offsets: torch.Tensor,
    layer_scales: Optional[Dict[int, float]] = None,
    device: Optional[torch.device] = None,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Merge per-layer directions (7 layers like 9..15) into a single [1, D_s] s_dir.

    layer_dirs: {layer_index -> tensor}, where tensor can be:
        - full-length [D_s] or [1, D_s]
        - compact   [d_L]  or [1, d_L] (d_L = layer_dims[L])
    layer_scales: optional per-layer scalar multipliers
    """
    # Determine D_s from s_offsets
    D_s = int(s_offsets[-1].item())
    # Choose device
    if device is None:
        # pick the device of the first tensor we find
        first = next(iter(layer_dirs.values()))
        device = first.device
    s_dir = torch.zeros(1, D_s, device=device)

    for L, v in layer_dirs.items():
        if v.dim() == 1:
            v_ = v.unsqueeze(0)  # [1, *]
        else:
            v_ = v

        # optional per-layer scaling
        if layer_scales is not None and L in layer_scales:
            v_ = v_ * float(layer_scales[L])

        start, end = get_slice(s_offsets, L)
        dL = end - start
        # If the vector already spans D_s, add as-is.
        if v_.shape[-1] == D_s:
            s_dir = s_dir + v_.to(device)
        else:
            # Expect compact length = layer slice
            assert v_.shape[-1] == dL, f"Layer {L}: expected length {dL}, got {v_.shape[-1]}"
            s_dir[:, start:end] = s_dir[:, start:end] + v_.to(device)

    if normalize:
        s_dir = s_dir / (s_dir.norm(p=2, dim=-1, keepdim=True) + eps)
    return s_dir

def prompts_to_centroid(tokenizer, model, prompts, device) -> torch.Tensor:
    toks = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        centroid = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
    return centroid

class TextCondDelta(nn.Module):
    def __init__(self, d_s: int, text_dim: int, hidden: int, edit_mask: torch.Tensor, weight_scale: float = 0.25):
        super().__init__()
        self.d_s = d_s
        self.edit_mask = edit_mask.bool().detach()
        self.weight_scale = weight_scale

        # project CLIP text embedding -> 128-d
        self.txt_proj = nn.Linear(text_dim, 128)

        # MLP over [s, txt_emb_128] -> per-dim weight
        self.mlp = nn.Sequential(
            nn.Linear(d_s + 128, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_s)
        )

        # ---- global additive bias on delta (masked later)
        self.global_bias = nn.Parameter(torch.zeros(d_s))
        with torch.no_grad():
            if 0 <= 12287 < d_s:
                self.global_bias[12287] = 0.0  # initialize delta[12287]=0.2

        # init small so weights start near zero
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.txt_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.txt_proj.bias)

    def forward(self, s: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        delta = s * weight(s, text) + global_bias
        then masked to editable dims.
        """
        txt128 = self.txt_proj(text_feat)                 # [B,128]
        h = torch.cat([s, txt128], dim=-1)                # [B, D_s+128]
        raw = self.mlp(h)                                  # [B, D_s]
        delta = torch.tanh(raw) * self.weight_scale      # [-w,w]
        # add global bias (broadcast)
        delta = delta + self.global_bias.view(1, -1).to(delta.dtype)
        # apply mask to BOTH (multiplicative + bias)
        delta = delta * self.edit_mask.view(1, -1).to(delta.dtype)
        return delta
# def get_layer_layout():
#     layer_dims=[4]+[1024]*10+[724,512,362,256,256]
#     offsets=torch.cumsum(torch.tensor([0]+layer_dims),0)
#     return layer_dims,offsets

# def build_edit_mask(Ds,layers_to_skip,device):
#     layer_dims,offsets=get_layer_layout()
#     mask=torch.ones(Ds,dtype=torch.bool)
#     for l in layers_to_skip:
#         s=int(offsets[l]); e=int(offsets[l+1])
#         mask[s:e]=False
#     return mask.to(device)
# class TextCondDelta(nn.Module):
#     """delta_s = f(CLIP_img_feat, CLIP_txt_feat)"""
#     def __init__(self, d_s, img_dim, txt_dim, hidden, edit_mask, scale=0.5):
#         super().__init__()
#         self.edit_mask = edit_mask.bool().detach()
#         self.scale = scale

#         self.img_proj = nn.Linear(img_dim, 512)
#         self.txt_proj = nn.Linear(txt_dim, 512)

#         self.mlp = nn.Sequential(
#             nn.Linear(1024, hidden), nn.GELU(),
#             nn.Linear(hidden, hidden), nn.GELU(),
#             nn.Linear(hidden, d_s)
#         )

#         # ---- global additive bias on delta (masked later)
#         self.bias = nn.Parameter(torch.zeros(d_s))

#         self.apply(self._init)

#     def _init(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(m.bias)

#     def forward(self, img_feat, txt_feat):
#         img512 = self.img_proj(img_feat)
#         txt512 = self.txt_proj(txt_feat)
#         x = torch.cat([img512, txt512], dim=-1)
#         delta = torch.tanh(self.mlp(x)) * self.scale + self.bias
#         return delta * self.edit_mask
    # else:
        # attr1, attr2 = attr_pairs.get(attr, ("glossy", "matte"))
        # with torch.no_grad():
        #     feat1 = prompts_to_centroid(tokenizer, clip_model, prompts[attr1], device)
        #     feat2 = prompts_to_centroid(tokenizer, clip_model, prompts[attr2], device)
        #     img = (generator.synthesis(ss=latent_s, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1) + 1) / 2
        #     f = get_clip_feat(img, clip_model, device)
        #     p = f@feat1.T
        #     n =  f@feat2.T
        #     ext = F.normalize(torch.cat([f, p,n] , dim=1))
        # if attr == "depth":
        #     s_dir = torch.load("trained_dirs/depth.pt").to(device) * 0.5
        # elif attr == "random":
        #     pack = torch.load('trained_dirs/full_cluster_pack_color_random.pt', map_location='cpu')
        #     interp = FullDirectionInterpolator(pack['cluster']['medoid_ext_feats'], pack['directions_full'], pack['s_offsets'], device=device, tau=float(pack['tau']))
        #     interp.load_state_dict(torch.load('trained_dirs/full_interpolator_color_random.pt', map_location='cpu'))
        #     s_dir = interp(ext, zero_layers=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15]) 
class TextCondDeltaW(nn.Module):
    def __init__(self, d_s: int, text_dim: int, hidden: int, edit_mask: torch.Tensor, weight_scale: float = 0.05):
        super().__init__()
        self.d_s = d_s
        self.edit_mask = edit_mask.bool().detach()
        self.weight_scale = weight_scale

        # project CLIP text embedding -> 128-d
        self.txt_proj = nn.Linear(text_dim, 128)

        # MLP over [s, txt_emb_128] -> per-dim weight
        self.mlp = nn.Sequential(
            nn.Linear(d_s + 128, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_s)
        )

        # ---- global additive bias on delta (masked later)
        self.global_bias = nn.Parameter(torch.zeros(d_s))
        # with torch.no_grad():
        #     if 0 <= 12287 < d_s:
        #         self.global_bias[12287] = 0.0  # initialize delta[12287]=0.2
            # w = torch.load("pretrained_weights/depth_weights.pt", map_location="cpu")
            # self.global_bias.data.copy_(w.view(-1).to(self.global_bias.device, dtype=self.global_bias.dtype))

        # init small so weights start near zero
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.txt_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.txt_proj.bias)

    def forward(self, s: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        delta = s * weight(s, text) + global_bias
        then masked to editable dims.
        """
        txt128 = self.txt_proj(text_feat)                 # [B,128]
        h = torch.cat([s, txt128], dim=-1)                # [B, D_s+128]
        raw = self.mlp(h)                                  # [B, D_s]
        delta = torch.tanh(raw) * self.weight_scale      # [-w,w]
        # add global bias (broadcast)
        delta = delta + self.global_bias.view(1, -1).to(delta.dtype)
        # # apply mask to BOTH (multiplicative + bias)
        # delta = delta * self.edit_mask.view(1, -1).to(delta.dtype)
        return delta