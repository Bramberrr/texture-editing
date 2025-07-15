import numpy as np
import torch
from skimage import color

import warnings

def lambdaS_curve(x, lam, alpha):
    if x < 0 or x > 1:
        return x
    if x < 1 / alpha:
        x_scaled = alpha * x
        eps = 1e-6
        ys = 1 - np.sqrt((lam ** 2) * (1 - x_scaled) ** 2 / (((1 - x_scaled) ** 2 * (lam ** 2 - 1)) + 1 + eps))
        return ys / alpha
    else:
        x_scaled = (alpha * x - 1) / (alpha - 1)
        ys = np.sqrt(x_scaled ** 2 / (x_scaled ** 2 + (1 - x_scaled ** 2) * (1 / lam) ** 2))
        return (ys * (alpha - 1) + 1) / alpha


lambdaS_vectorized = np.vectorize(lambdaS_curve)


def generate_raised_cosine_filters_numpy(Gsize, bandwidth, centerf):
    xs, ys = np.meshgrid(np.arange(Gsize) - Gsize / 2, np.arange(Gsize) - Gsize / 2, indexing='ij')
    radfreq = np.sqrt(xs ** 2 + ys ** 2)
    sfnum = len(centerf)
    filts = np.zeros((sfnum, Gsize, Gsize), dtype=np.complex64)

    for x in range(sfnum):
        lowcutoff = centerf[x] * 2 ** (-bandwidth)
        highcutoff = centerf[x] * 2 ** (+bandwidth)
        sfmask = (radfreq >= lowcutoff) & (radfreq < highcutoff)
        sffilter = 0.5 + 0.5 * np.cos(np.pi * (np.log2(radfreq + 1e-6) - np.log2(centerf[x])) / bandwidth)
        filts[x] = sfmask * sffilter
        filts[x, 0, 0] = 0

    bandpass_stack = np.sum(filts, axis=0)
    lowfilt = 1.0 - bandpass_stack
    return filts, lowfilt


def enhance_l_channel(l_channel: np.ndarray, lam: float, bandwidth: float = 1.8):
    H, W = l_channel.shape
    sfs = 2 ** np.arange(0, int(np.floor(np.log2(H / 2)))+1)
    filts, lowfilt = generate_raised_cosine_filters_numpy(H, bandwidth, sfs)

    imgfftspec = np.fft.fft2(l_channel)
    local_lum = np.fft.ifft2(imgfftspec * lowfilt).real

    subimg_s = np.zeros((H, W, len(sfs)))

    for x in range(len(sfs)):
        band = np.fft.ifft2(imgfftspec * filts[x]).real
        ratio = (band + 1e-6) / (local_lum + 1e-6)

        r_min, r_max = ratio.min(), ratio.max()
        ratio_norm = (ratio - r_min) / (r_max - r_min + 1e-6)
        alpha = 1 / np.clip(ratio_norm.mean(), 1e-3, 1 - 1e-3)
        ratio_s = lambdaS_vectorized(ratio_norm, lam, alpha)
        ratio_s = ratio_s * (r_max - r_min) + r_min
        subimg_s[:, :, x] = ratio_s * (local_lum + 1e-6)

    img_s = np.sum(subimg_s, axis=2) + local_lum
    return img_s



def S_Transformation(img_tensor: torch.Tensor, lam: float) -> np.ndarray:
    """
    Args:
        img_tensor (torch.Tensor): RGB image tensor [3, H, W], float, [0,1], on device
        lam (float): lambda for S-curve enhancement

    Returns:
        torch.Tensor [3,H,W] in [0,1]
    """
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    img_lab = color.rgb2lab(img_np)
    l_channel = img_lab[:, :, 0] / 100.0

    enhanced_l = enhance_l_channel(l_channel, lam)
    img_lab[:, :, 0] = np.clip(enhanced_l * 100.0, 0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        enhanced_rgb = np.clip(color.lab2rgb(img_lab), 0, 1)

    return torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float()
