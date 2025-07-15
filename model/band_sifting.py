#  Python implementation by Zhiwei for paper 'Band-Sifting 
#  Decomposition for Image-Based Material Editing'
# ——————————————————————————————————————————————————————————
import numpy as np
from skimage import color
import torch
import cv2
import cv2.ximgproc as xip

def band_sifting_editing(input_tensor, effect="shine", strength=2.0, log_epsilon=1e-4, filter_epsilon=1e-2):
    """
    input_tensor: torch.Tensor [3,H,W] in [0,1]
    output: torch.Tensor [3,H,W] in [0,1]
    """
    # --- Decompose ---
    def guided_filter(I, radius, eps):
        return xip.guidedFilter(I, I, radius, eps)

    def decompose(L, depth=None):
        if depth is None:
            min_size = min(L.shape)
            depth = int(np.floor(np.log2(min_size))) - 1
        L_log = np.log(L + log_epsilon)
        res = []
        Lp = L_log.copy()
        for k in range(1, depth + 1):
            r = 2 ** k
            Ln = guided_filter(Lp, r, filter_epsilon)
            D = Lp - Ln
            res.append(D)
            Lp = Ln
        res.append(Lp)
        return res

    def reconstruct(res):
        L_log = np.sum(res, axis=0)
        return np.exp(L_log) - log_epsilon

    # --- Convert to NumPy ---
    rgb = input_tensor.permute(1, 2, 0).detach().cpu().numpy()

    # --- RGB -> Lab ---
    lab = color.rgb2lab(rgb)
    L = lab[..., 0] / 100.0
    a = lab[..., 1]
    b = lab[..., 2]

    residuals = decompose(L)
    depth = len(residuals) - 1

    # --- Apply effect ---
    if strength != 1.0:
        effect_table = {
            "shine":   ("H", "H", "P"),
            "spots":   ("H", "H", "N"),
            "rough":   ("H", "L", "P"),
            "stain":   ("H", "L", "N"),
            "blemish": ("H", "L", "A"),
            "gloss":   ("H", "H", "A"),
            "shadow":  ("L", "A", "N"),
            "sharp":   ("H", "H", "N"),
            "metal":   ("L", "A", "A"),
        }
        freq, amp, sign = effect_table[effect]

        if freq == "H":
            freq_num = list(range(0, int(np.floor(depth/2))))
        elif freq == "L":
            freq_num = list(range(int(np.floor(depth/2)), depth))
        else:
            freq_num = list(range(0, depth))

        for fi in freq_num:
            D = residuals[fi]
            if amp == "H":
                mask = (np.abs(D) > np.std(D))
            elif amp == "L":
                mask = (np.abs(D) <= np.std(D))
            else:
                mask = np.ones_like(D, dtype=bool)

            if sign == "P":
                mask = mask & (D > 0)
            elif sign == "N":
                mask = mask & (D < 0)

            D_mod = D.copy()
            D_mod[mask] *= strength
            residuals[fi] = D_mod
    else:
        return input_tensor

    # --- Reconstruct ---
    L_new = reconstruct(residuals)

    # --- Combine Lab ---
    lab_new = np.stack([L_new * 100.0, a, b], axis=-1)
    rgb_new = color.lab2rgb(lab_new)
    rgb_new = np.clip(rgb_new, 0, 1)

    # --- Back to torch ---
    output = torch.from_numpy(rgb_new).permute(2, 0, 1).float()
    return output