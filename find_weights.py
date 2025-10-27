import os, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from scipy.optimize import minimize
import torch.nn.functional as F
from scipy.stats import kurtosis
@torch.no_grad()
def extract_artifact_features(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    gray = (0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]).float().clamp(0,1)
    H, W = gray.shape
    device = gray.device
    g = gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # 1) Clipping
    y255 = (gray * 255.0).round().clamp(0, 255).long().view(-1)
    hist = torch.bincount(y255, minlength=256).float()
    total = float(hist.sum().item() + 1e-8)
    frac255 = float(hist[255].item() / total)
    h254, h255 = float(hist[254].item()), float(hist[255].item())
    jump_ratio = float(h255 / (h254 + 1e-8))

    # 2) Frequency / grain
    F2 = torch.fft.fftshift(torch.fft.fft2(gray))
    P2 = (F2.real**2 + F2.imag**2)
    P2 = P2 / (P2.sum() + 1e-8)
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=device),
        torch.linspace(-0.5, 0.5, W, device=device),
        indexing='ij'
    )
    r = torch.sqrt(xx**2 + yy**2)
    hf_ratio = float(P2[r > 0.35].sum().item() / (P2[r <= 0.35].sum().item() + 1e-8))

    mean = float(gray.mean().item())
    std  = float(gray.std().item() + 1e-8)
    bright_density = float((gray > (mean + 3.0 * std)).float().mean().item())

    r_bins = torch.linspace(0, 0.5, 64, device=device)
    radial_p = []
    for i in range(len(r_bins)-1):
        m = (r >= r_bins[i]) & (r < r_bins[i+1])
        radial_p.append(float(P2[m].mean().item()) if m.any() else 0.0)
    radial_p = np.asarray(radial_p, dtype=np.float64)
    if (radial_p > 0).sum() >= 3:
        logf = np.log(np.arange(1, len(radial_p)+1))
        logp = np.log(radial_p + 1e-12)
        slope = float(np.polyfit(logf, logp, 1)[0])
    else:
        slope = 0.0

    kt = float(kurtosis(gray.cpu().numpy().ravel()))

    # 3) Edge/saturation interaction
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
    gx = F.conv2d(g, sobel_x, padding=1)
    gy = F.conv2d(g, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx*gx + gy*gy).squeeze()
    sat_edge_ratio = float(((gray >= 0.995) & (grad_mag >= (grad_mag.mean() + 1.5*grad_mag.std()))).float().mean().item())

    # 4) Isolated bright hotspots
    nbr_kernel = torch.ones((1,1,3,3), device=device)
    bright_mask = (g >= 0.98).float()
    neigh = F.conv2d(bright_mask, nbr_kernel, padding=1).squeeze()
    hotspot_isolated_ratio = float(((gray >= 0.98) & (neigh <= 2)).float().mean().item())

    # 5) Banding score
    h = (hist.cpu().numpy().astype(np.float64))
    h = h / (h.sum() + 1e-12)
    d2 = np.diff(h, n=2)
    banding_score = float(np.sqrt((d2**2).mean()))

    # 6) Blockiness (fixed)
    block = 8
    if W >= (block + 1):
        cols_right = gray[:, block::block]
        cols_left  = gray[:, block-1:W:block]
        m = min(cols_right.shape[1], cols_left.shape[1])
        dv = torch.abs(cols_right[:, :m] - cols_left[:, :m]).mean().item()
    else:
        dv = 0.0
    if H >= (block + 1):
        rows_bottom = gray[block::block, :]
        rows_top    = gray[block-1:H:block, :]
        m = min(rows_bottom.shape[0], rows_top.shape[0])
        dh = torch.abs(rows_bottom[:m, :] - rows_top[:m, :]).mean().item()
    else:
        dh = 0.0
    intra_v = torch.abs(gray[:, 1:] - gray[:, :-1]).mean().item() if W > 1 else 0.0
    intra_h = torch.abs(gray[1:, :] - gray[:-1, :]).mean().item() if H > 1 else 0.0
    denom = (intra_v + intra_h) / 2.0 + 1e-6
    blockiness = float((dv + dh) / 2.0 / denom)

    # 7) Laplacian overshoot
    lap_kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=device).view(1,1,3,3)
    lap = F.conv2d(g, lap_kernel, padding=1).squeeze().abs()
    thr = float(lap.mean().item() + 2.0 * lap.std().item())
    laplacian_overshoot_ratio = float((lap > thr).float().mean().item())

    # 8) Bright & locally high variance
    k5 = torch.ones((1,1,5,5), device=device) / 25.0
    mu = F.conv2d(g, k5, padding=2)
    mu2 = F.conv2d(g*g, k5, padding=2)
    local_var = (mu2 - mu*mu).clamp_min(0).squeeze()
    lv_thr = float(local_var.mean().item() + 2.0 * local_var.std().item())
    bright_highvar_ratio = float(((gray > mean + 2.0*std) & (local_var > lv_thr)).float().mean().item())

    # 9) Gradient orientation entropy
    eps = 1e-8
    theta = torch.atan2(gy + eps, gx + eps).squeeze().cpu().numpy().ravel()
    mag = grad_mag.cpu().numpy().ravel()
    mask = mag > (mag.mean() + 0.5 * mag.std())
    if mask.sum() > 100:
        bins = np.linspace(-np.pi, np.pi, 37)
        h_orient, _ = np.histogram(theta[mask], bins=bins)
        p = h_orient / (h_orient.sum() + 1e-12)
        grad_orient_entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))
    else:
        grad_orient_entropy = 0.0

    return {
        "frac255": frac255,
        "jump_ratio": jump_ratio,
        "hf_ratio": hf_ratio,
        "bright_density": bright_density,
        "slope": slope,
        "kurtosis": kt,
        "sat_edge_ratio": sat_edge_ratio,
        "hotspot_isolated_ratio": hotspot_isolated_ratio,
        "banding_score": banding_score,
        "blockiness": blockiness,
        "laplacian_overshoot_ratio": laplacian_overshoot_ratio,
        "bright_highvar_ratio": bright_highvar_ratio,
        "grad_orient_entropy": grad_orient_entropy,
    }



def _load_features(folder, device):
    to_tensor = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    feats, names = [], None
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        raise ValueError(f"No images in {folder}")
    for fname in tqdm(files, desc=f"Loading {folder}"):
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        f = extract_artifact_features(to_tensor(img).unsqueeze(0).to(device))
        if names is None: names = list(f.keys())
        feats.append([float(f[k]) for k in names])
    return np.array(feats, dtype=np.float64), names


def softmax(u):
    u = u - np.max(u)
    e = np.exp(u)
    return e / (e.sum() + 1e-12)

def softmin(x, tau=0.02):
    # smooth approximation of min(x)
    # tau is small ⇒ tighter to the true min
    m = np.min(x)
    z = np.exp(-(x - m)/max(tau,1e-12)).sum()
    return float(m - tau*np.log(z + 1e-12))

def softmax_val(x, tau=0.02):
    # smooth approximation of max(x)
    M = np.max(x)
    z = np.exp((x - M)/max(tau,1e-12)).sum()
    return float(M + tau*np.log(z + 1e-12))


def fit_artifact_weights_simplex_balanced(
    margin_dir, under_dir, device="cuda",
    target_center=1.005,            # center for positive scores
    target_width=0.010,             # want pos range ≈ 0.01
    tau_extrema=0.02,               # softness for softmin/softmax
    w_reg=1e-3,                     # small L2 on weights (via u’s induced w)
    neg_margin=0.000,               # push negatives below cutoff - neg_margin
    maxiter=800
):
    """
    Learn weights w (sum=1, w>=0) so that:
      - Positive (margin) scores cluster tightly (width ~ target_width) around target_center
      - Negative (under_margin) scores fall below the positive soft-min cutoff
    Returns: dict(pack) with 'names','weights','mins','maxs','cutoff'
    """
    Xp_raw, names = _load_features(margin_dir, device)
    Xn_raw, _     = _load_features(under_dir, device)

    # Joint min-max normalization
    X_all = np.vstack([Xp_raw, Xn_raw])
    mins, maxs = X_all.min(0), X_all.max(0)
    spans = (maxs - mins + 1e-8)
    Xp = (Xp_raw - mins) / spans
    Xn = (Xn_raw - mins) / spans

    d = Xp.shape[1]

    # optimize u (unconstrained), map to w=softmax(u) (on simplex)
    u0 = np.zeros(d, dtype=np.float64)

    def loss(u):
        w = softmax(u)                      # sum=1, >=0
        s_pos = Xp @ w                      # scores for positives
        s_neg = Xn @ w                      # scores for negatives

        # Tight cluster for positive scores:
        # - center near target_center
        # - small width (softmax - softmin) near target_width
        smin = softmin(s_pos, tau=tau_extrema)
        smax = softmax_val(s_pos, tau=tau_extrema)
        width = smax - smin
        center = s_pos.mean()

        # negatives should be below the positive "cutoff"
        cutoff = smin  # learned by the model
        # soft hinge: penalize any negative above (cutoff - neg_margin)
        neg_violation = np.clip(s_neg - (cutoff - neg_margin), 0, None)

        # Loss terms
        L_center = (center - target_center)**2
        L_width  = (width - target_width)**2
        L_neg    = (neg_violation**2).mean()
        L_reg    = w_reg * np.sum(w**2)

        # Slight variance penalty on positives to tighten cluster further
        L_var = np.var(s_pos)

        # Combine
        # weights chosen empirically; adjust if you need tighter negatives or tighter cluster
        total = (1.0 * L_center) + (4.0 * L_width) + (2.0 * L_var) + (4.0 * L_neg) + L_reg
        return float(total)

    res = minimize(loss, u0, method="L-BFGS-B", options=dict(maxiter=maxiter))
    u = res.x
    w = softmax(u)
    s_pos = Xp @ w
    s_neg = Xn @ w
    cutoff = softmin(s_pos, tau=tau_extrema)

    # Reporting
    print("\n===== Simplex-balanced artifact weights =====")
    for k, v in zip(names, w):
        print(f"{k:15s}: {v:.6f}")
    print(f"Pos: min={s_pos.min():.4f} mean={s_pos.mean():.4f} max={s_pos.max():.4f}  width={s_pos.max()-s_pos.min():.4f}")
    print(f"Pos (soft): softmin={softmin(s_pos, tau_extrema):.4f} softmax={softmax_val(s_pos, tau_extrema):.4f}")
    print(f"Neg: min={s_neg.min():.4f} mean={s_neg.mean():.4f} max={s_neg.max():.4f}")
    print(f"Cutoff (pos softmin): {cutoff:.4f}")
    print(f"% pos in [cutoff, cutoff+0.01]: {100*np.mean((s_pos>=cutoff)&(s_pos<=cutoff+0.01)):.2f}%")
    print(f"% neg below cutoff:             {100*np.mean(s_neg < cutoff):.2f}%")

    pack = {
        "names": names,
        "weights": w,
        "mins": mins,
        "maxs": maxs,
        "cutoff": cutoff,          # use this as the decision threshold
        "target_center": target_center,
        "target_width": target_width,
        "tau_extrema": tau_extrema,
    }
    np.save("artifact_guard_simplex_balanced.npy", pack)
    return pack


if __name__ == "__main__":
    folder = "margin"
    under_folder = "under_margin"
    pack = fit_artifact_weights_simplex_balanced(folder, under_folder)
