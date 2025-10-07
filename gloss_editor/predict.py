import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import torch.nn.functional as F

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
