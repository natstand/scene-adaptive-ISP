#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import *
from .common import _rgb_luminance

# ============================================================
# Metrics
# ============================================================

def _masked_pixels_chw(x: torch.Tensor, mask: torch.Tensor, max_samples=200000):
    x = x.detach().clamp(0, 1)
    mask = mask.detach().clamp(0, 1)
    idx = torch.nonzero(mask.flatten() > 0.2, as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(x.shape[-2] * x.shape[-1], device=x.device)
    if idx.numel() > max_samples:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_samples]]
    return x.permute(0, 2, 3, 1).reshape(-1, 3)[idx]


def ciede2000_lab(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 color difference for tensors shaped [N,3] in Lab."""
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    C1 = torch.sqrt(a1 * a1 + b1 * b1 + 1e-12)
    C2 = torch.sqrt(a2 * a2 + b2 * b2 + 1e-12)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - torch.sqrt(Cbar.pow(6) / (Cbar.pow(6) + 25.0 ** 6 + 1e-12)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1 + 1e-12)
    C2p = torch.sqrt(a2p * a2p + b2 * b2 + 1e-12)
    h1p = torch.remainder(torch.atan2(b1, a1p) * 180.0 / math.pi, 360.0)
    h2p = torch.remainder(torch.atan2(b2, a2p) * 180.0 / math.pi, 360.0)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dhp = torch.where((C1p * C2p) <= 1e-12, torch.zeros_like(dh), dh)
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0 * torch.sqrt(C1p * C2p + 1e-12) * torch.sin((dhp * math.pi / 180.0) / 2.0)
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = (h1p - h2p).abs()
    hbarp = torch.where((C1p * C2p) <= 1e-12, hsum, 0.5 * hsum)
    hbarp = torch.where(((C1p * C2p) > 1e-12) & (hdiff > 180.0) & (hsum < 360.0), 0.5 * (hsum + 360.0), hbarp)
    hbarp = torch.where(((C1p * C2p) > 1e-12) & (hdiff > 180.0) & (hsum >= 360.0), 0.5 * (hsum - 360.0), hbarp)
    hp = hbarp * math.pi / 180.0
    T = 1.0 - 0.17 * torch.cos(hp - 30.0 * math.pi / 180.0) + 0.24 * torch.cos(2.0 * hp) + 0.32 * torch.cos(3.0 * hp + 6.0 * math.pi / 180.0) - 0.20 * torch.cos(4.0 * hp - 63.0 * math.pi / 180.0)
    dtheta = 30.0 * torch.exp(-((hbarp - 275.0) / 25.0).pow(2))
    Rc = 2.0 * torch.sqrt(Cbarp.pow(6) / (Cbarp.pow(6) + 25.0 ** 6 + 1e-12))
    Sl = 1.0 + 0.015 * (Lbarp - 50.0).pow(2) / torch.sqrt(20.0 + (Lbarp - 50.0).pow(2))
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -torch.sin(2.0 * dtheta * math.pi / 180.0) * Rc
    return torch.sqrt((dLp / Sl).pow(2) + (dCp / Sc).pow(2) + (dHp / Sh).pow(2) + Rt * (dCp / Sc) * (dHp / Sh) + 1e-12)


def compute_hue_error_metrics(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, chroma_min: float = 3.0):
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid = mask.flatten() > 0.2
    if valid.sum() == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    p_ab = pred_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    t_ab = tgt_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    p_chroma = p_ab.norm(dim=1)
    t_chroma = t_ab.norm(dim=1)
    hue_valid = (p_chroma > float(chroma_min)) & (t_chroma > float(chroma_min))
    if hue_valid.sum() == 0:
        hue_valid = torch.ones_like(p_chroma, dtype=torch.bool)
    p_hue = torch.atan2(p_ab[:, 1], p_ab[:, 0]) * (180.0 / math.pi)
    t_hue = torch.atan2(t_ab[:, 1], t_ab[:, 0]) * (180.0 / math.pi)
    hue_diff = torch.remainder(p_hue - t_hue + 180.0, 360.0) - 180.0
    hue_err = hue_diff.abs()[hue_valid]
    return {
        "mean_deg": float(hue_err.mean()),
        "median_deg": float(hue_err.median()),
        "p95_deg": float(torch.quantile(hue_err, 0.95)),
        "valid_ratio": float(hue_valid.float().mean()),
    }


def _add_prefixed_metrics(dst: dict, prefix: str, values: dict):
    for k, v in values.items():
        dst[f"{prefix}_{k}"] = v


def compute_color_metrics(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
    pred_lab, tgt_lab = srgb_to_lab_fast(pred), srgb_to_lab_fast(tgt)
    delta = ((pred_lab - tgt_lab).pow(2).sum(1, keepdim=True).sqrt() * mask).flatten()
    valid = mask.flatten() > 0.2
    if valid.sum() == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    de = delta[valid]
    pred_lab_pix = pred_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid]
    tgt_lab_pix = tgt_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid]
    if pred_lab_pix.shape[0] > 200000:
        idx = torch.randperm(pred_lab_pix.shape[0], device=pred_lab_pix.device)[:200000]
        pred_lab_pix = pred_lab_pix[idx]
        tgt_lab_pix = tgt_lab_pix[idx]
    de00 = ciede2000_lab(pred_lab_pix, tgt_lab_pix)
    p_pix = _masked_pixels_chw(pred, mask)
    t_pix = _masked_pixels_chw(tgt, mask)
    dot = (p_pix * t_pix).sum(1)
    angle = torch.acos((dot / (p_pix.norm(dim=1) * t_pix.norm(dim=1) + 1e-8)).clamp(-1, 1)) * (180.0 / math.pi)
    hue = compute_hue_error_metrics(pred, tgt, mask)
    mse = ((pred - tgt).pow(2) * mask).sum() / (mask.sum() * 3.0 + 1e-6)
    return {
        "deltaE_mean": float(de.mean()),
        "deltaE_median": float(de.median()),
        "deltaE_p95": float(torch.quantile(de, 0.95)),
        "deltaE00_mean": float(de00.mean()),
        "deltaE00_median": float(de00.median()),
        "deltaE00_p95": float(torch.quantile(de00, 0.95)),
        "rgb_mae": float(((pred - tgt).abs() * mask).sum() / (mask.sum() * 3.0 + 1e-6)),
        "color_angle_mean_deg": float(angle.mean()),
        "color_angle_median_deg": float(angle.median()),
        "hue_error_mean_deg": hue["mean_deg"],
        "hue_error_median_deg": hue["median_deg"],
        "hue_error_p95_deg": hue["p95_deg"],
        "hue_valid_ratio": hue["valid_ratio"],
        "psnr_rgb": float(10.0 * torch.log10(1.0 / (mse + 1e-8))),
        "mask_mean": float(mask.mean()),
    }


def save_color_histogram(path: str, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, bins=64):
    pred_np = _masked_pixels_chw(pred, mask).detach().cpu().numpy()
    tgt_np = _masked_pixels_chw(tgt, mask).detach().cpu().numpy()
    W, H = 900, 520
    pad_l, pad_t, plot_w, plot_h = 56, 36, 820, 390
    img = Image.new("RGB", (W, H), (248, 250, 253))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = font_b = ImageFont.load_default()
    draw.text((pad_l, 10), "Color histogram distribution: predicted vs aligned phone", fill=(22, 45, 76), font=font_b)
    draw.rectangle([pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(180, 192, 208), width=1)
    colors = [(214, 54, 54), (47, 150, 74), (55, 105, 210)]
    labels = ["R", "G", "B"]
    for c in range(3):
        hp, _ = np.histogram(pred_np[:, c], bins=bins, range=(0, 1), density=True)
        ht, _ = np.histogram(tgt_np[:, c], bins=bins, range=(0, 1), density=True)
        maxv = max(float(hp.max()), float(ht.max()), 1e-6)
        pts_p, pts_t = [], []
        for i in range(bins):
            x = pad_l + int((i / (bins - 1)) * plot_w)
            pts_p.append((x, pad_t + plot_h - int((hp[i] / maxv) * (plot_h - 12))))
            pts_t.append((x, pad_t + plot_h - int((ht[i] / maxv) * (plot_h - 12))))
        draw.line(pts_t, fill=tuple(min(255, v + 80) for v in colors[c]), width=2)
        draw.line(pts_p, fill=colors[c], width=3)
        draw.text((pad_l + 12 + c * 145, H - 62), f"{labels[c]} pred", fill=colors[c], font=font)
        draw.text((pad_l + 12 + c * 145, H - 36), f"{labels[c]} phone", fill=tuple(min(255, v + 80) for v in colors[c]), font=font)
    draw.text((pad_l, pad_t + plot_h + 12), "x: intensity 0..1, y: normalized frequency. Thick lines are predicted RGB.", fill=(58, 74, 96), font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def resize_like(x: torch.Tensor, ref: torch.Tensor, mode="bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def masked_psnr(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    mask3 = mask.expand_as(pred)
    mse = ((pred.clamp(0, 1) - tgt.clamp(0, 1)).pow(2) * mask3).sum() / (mask3.sum() + 1e-8)
    return float(10.0 * torch.log10(1.0 / (mse + 1e-8)))


def masked_ssim(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    tgt = tgt.clamp(0, 1)
    mask = mask.clamp(0, 1)
    pad = 5
    x = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(tgt, (pad, pad, pad, pad), mode="reflect")
    ux = F.avg_pool2d(x, 11, stride=1)
    uy = F.avg_pool2d(y, 11, stride=1)
    uxx = F.avg_pool2d(x * x, 11, stride=1)
    uyy = F.avg_pool2d(y * y, 11, stride=1)
    uxy = F.avg_pool2d(x * y, 11, stride=1)
    vx = (uxx - ux * ux).clamp_min(0)
    vy = (uyy - uy * uy).clamp_min(0)
    vxy = uxy - ux * uy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
    mask3 = mask.expand_as(ssim)
    return float((ssim * mask3).sum() / (mask3.sum() + 1e-8))


def masked_deltae00(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid = mask.flatten() > 0.2
    if valid.sum() == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    p = pred_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid]
    t = tgt_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid]
    if p.shape[0] > 200000:
        idx = torch.linspace(0, p.shape[0] - 1, 200000, device=p.device).long()
        p, t = p[idx], t[idx]
    return float(ciede2000_lab(p, t).mean())


def dark_mask_from_target(tgt: torch.Tensor, mask: torch.Tensor, q=0.20) -> torch.Tensor:
    y = _rgb_luminance(tgt.clamp(0, 1))
    valid = mask > 0.2
    vals = y[valid]
    if vals.numel() < 8:
        return mask
    thr = torch.quantile(vals, q)
    return (mask * (y <= thr).float()).clamp(0, 1)


def masked_chroma_error(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid = mask.flatten() > 0.2
    if valid.sum() == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    p_ab = pred_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    t_ab = tgt_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    return float((p_ab.norm(dim=1) - t_ab.norm(dim=1)).abs().mean())


def masked_hue_hist_emd(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, chroma_min=10.0, bins=36) -> float:
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid = mask.flatten() > 0.2
    if valid.sum() == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    p_ab = pred_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    t_ab = tgt_lab[:, 1:3].permute(0, 2, 3, 1).reshape(-1, 2)[valid]
    p_chroma = p_ab.norm(dim=1)
    t_chroma = t_ab.norm(dim=1)
    hue_valid = (p_chroma > float(chroma_min)) & (t_chroma > float(chroma_min))
    if hue_valid.sum() < 128:
        return 0.0
    p_hue = torch.atan2(p_ab[:, 1], p_ab[:, 0]) * (180.0 / math.pi)
    t_hue = torch.atan2(t_ab[:, 1], t_ab[:, 0]) * (180.0 / math.pi)
    p_hue = torch.remainder(p_hue[hue_valid], 360.0)
    t_hue = torch.remainder(t_hue[hue_valid], 360.0)
    hp = torch.histc(p_hue, bins=int(bins), min=0.0, max=360.0)
    ht = torch.histc(t_hue, bins=int(bins), min=0.0, max=360.0)
    hp = hp / (hp.sum() + 1e-12)
    ht = ht / (ht.sum() + 1e-12)
    diff = hp - ht
    best = None
    bin_width = 360.0 / float(bins)
    for shift in range(int(bins)):
        cdf = torch.cumsum(torch.roll(diff, shifts=shift, dims=0), dim=0)
        cost = cdf.abs().sum() * bin_width
        best = cost if best is None else torch.minimum(best, cost)
    return float(best)


def masked_lab_hist_js(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, bins=16, max_samples=200000) -> float:
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid_idx = torch.nonzero(mask.flatten() > 0.2, as_tuple=False).flatten()
    if valid_idx.numel() < 128:
        return 0.0
    if valid_idx.numel() > max_samples:
        sample_idx = torch.linspace(0, valid_idx.numel() - 1, int(max_samples), device=valid_idx.device).long()
        valid_idx = valid_idx[sample_idx]
    p = pred_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx]
    t = tgt_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx]

    def to_hist(lab: torch.Tensor) -> torch.Tensor:
        norm = torch.empty_like(lab)
        norm[:, 0] = (lab[:, 0] / 100.0).clamp(0, 1)
        norm[:, 1] = ((lab[:, 1] + 128.0) / 255.0).clamp(0, 1)
        norm[:, 2] = ((lab[:, 2] + 128.0) / 255.0).clamp(0, 1)
        q = torch.clamp((norm * int(bins)).long(), 0, int(bins) - 1)
        idx = (q[:, 0] * int(bins) + q[:, 1]) * int(bins) + q[:, 2]
        hist = torch.bincount(idx, minlength=int(bins) ** 3).float()
        return hist / (hist.sum() + 1e-12)

    hp = to_hist(p)
    ht = to_hist(t)
    eps = 1e-12
    m = 0.5 * (hp + ht)
    js = 0.5 * (hp * ((hp + eps) / (m + eps)).log()).sum() + 0.5 * (ht * ((ht + eps) / (m + eps)).log()).sum()
    return float(js)


def masked_lab_swd_fixed(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, max_samples=120000) -> float:
    pred_lab = srgb_to_lab_fast(pred.clamp(0, 1))
    tgt_lab = srgb_to_lab_fast(tgt.clamp(0, 1))
    valid_idx = torch.nonzero(mask.flatten() > 0.2, as_tuple=False).flatten()
    if valid_idx.numel() < 128:
        return 0.0
    if valid_idx.numel() > max_samples:
        sample_idx = torch.linspace(0, valid_idx.numel() - 1, int(max_samples), device=valid_idx.device).long()
        valid_idx = valid_idx[sample_idx]
    p = pred_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx].clone()
    t = tgt_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx].clone()
    p[:, 0] *= 0.6
    t[:, 0] *= 0.6
    dirs = p.new_tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ])
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
    pa = p @ dirs.T
    ta = t @ dirs.T
    loss = p.new_tensor(0.0)
    for k in range(dirs.shape[0]):
        ua, _ = torch.sort(pa[:, k])
        ub, _ = torch.sort(ta[:, k])
        loss = loss + (ua - ub).abs().mean()
    return float(loss / dirs.shape[0])


def compute_extended_metrics(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> dict:
    pred = pred.clamp(0, 1)
    tgt = tgt.clamp(0, 1)
    mask = mask.clamp(0, 1)
    dark = dark_mask_from_target(tgt, mask)
    return {
        "psnr": masked_psnr(pred, tgt, mask),
        "ssim": masked_ssim(pred, tgt, mask),
        "deltaE00": masked_deltae00(pred, tgt, mask),
        "dark20_deltaE00": masked_deltae00(pred, tgt, dark),
        "chroma_error": masked_chroma_error(pred, tgt, mask),
        "lab_hist_js": masked_lab_hist_js(pred, tgt, mask),
        "hue_hist_emd": masked_hue_hist_emd(pred, tgt, mask),
        "lab_swd": masked_lab_swd_fixed(pred, tgt, mask),
        "valid_ratio": float((mask > 0.2).float().mean()),
        "dark_ratio": float((dark > 0.2).float().mean()),
    }
