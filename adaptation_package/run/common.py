#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from .full4_denoiser import (
    complete_denoiser_output,
    make_condition_tensor,
    pack_bayer_tensor,
    unpack_bayer_tensor,
)

def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

def add_bool_arg(parser, name, default):
    parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default)

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _rgb_luminance(rgb: torch.Tensor) -> torch.Tensor:
    w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb.clamp(0, 1) * w).sum(1, keepdim=True)


# ============================================================
# IO
# ============================================================

def save_rgb(path: str, rgb01: torch.Tensor):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = rgb01.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_mask(path: str, mask01: torch.Tensor):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = mask01.detach().cpu().clamp(0, 1)[0, 0].numpy()
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_gray_vis(path: str, x: torch.Tensor, percentile=(1.0, 99.0)):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().float().cpu()
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    arr_np = arr.numpy()
    lo, hi = np.percentile(arr_np, percentile)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        lo = float(arr_np.min())
        hi = float(arr_np.max() + 1e-12)
    vis = np.clip((arr_np - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    Image.fromarray((vis * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def load_rgb(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].clamp(0, 1)


def load_mask(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].clamp(0, 1)


def read_12in16_left_aligned(raw_path, width, height, little_endian=True, zero_ratio_th=0.58):
    dtype = np.dtype("<u2" if little_endian else ">u2")
    
    arr = np.fromfile(raw_path, dtype=dtype)
    expected = int(height) * int(width)
    if arr.size < expected:
        raise ValueError(f".raw too small: got {arr.size}, expected {expected}")
    if arr.size > expected:
        arr = arr[:expected]
    img16 = arr.reshape(int(height), int(width))
    low4 = img16 & 0x000F
    high4 = img16 & 0xF000
    low4_zero_ratio = float(np.mean(low4 == 0))
    high4_zero_ratio = float(np.mean(high4 == 0))
    img12_shift = (img16 >> 4).astype(np.uint16)
    img12_mask = (img16 & 0x0FFF).astype(np.uint16)
    if low4_zero_ratio > zero_ratio_th and img12_shift.max() <= 4095:
        img12 = img12_shift
        mode = "12-bit in uint16 container, left-aligned -> shift right by 4"
    elif high4_zero_ratio > zero_ratio_th and img12_mask.max() <= 4095:
        img12 = img12_mask
        mode = "12-bit in uint16 container, right-aligned -> mask 0x0FFF"
    else:
        img12 = img16.astype(np.uint16)
        mode = "unknown layout, using raw uint16 values directly"
    print(

        f"[read_12in16] low4_zero_ratio={low4_zero_ratio:.6f}, "
        f"high4_zero_ratio={high4_zero_ratio:.6f}, img16_min={img16.min()}, "
        f"img16_max={img16.max()}, img12_max={img12.max()}, mode={mode}"
    )
    return img16, img12, mode


def load_hs_raw(path: str,
                height: int = None,
                width: int = None,
                raw_dtype: str = "uint16",
                little_endian: bool = True,
                black_level: float = 0.0,
                white_level: float = None,
                use_12in16: bool = True,
                force_single_channel: bool = True) -> torch.Tensor:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".raw", ".raww"):
        if height is None or width is None:
            raise ValueError(f"Binary RAW loading requires raw_h/raw_w, got {height}/{width}")
        raw_dtype = str(raw_dtype).lower()
        if raw_dtype == "uint16":
            if white_level is None:
                white_level = 4095.0
            if use_12in16:
                _, arr, mode = read_12in16_left_aligned(str(p), width, height, little_endian=little_endian)
                print(f"[load_hs_raw:raw] mode = {mode}")
            else:
                dtype = np.dtype("<u2" if little_endian else ">u2")
                arr = np.fromfile(str(p), dtype=dtype)
                expected = int(height) * int(width)
                if arr.size < expected:
                    raise ValueError(f".raw too small: got {arr.size}, expected {expected}")
                if arr.size > expected:
                    arr = arr[:expected]
                arr = arr.reshape(int(height), int(width)).astype(np.uint16)
        elif raw_dtype == "uint8":
            if white_level is None:
                white_level = 255.0
            arr = np.fromfile(str(p), dtype=np.uint8)
            expected = int(height) * int(width)
            if arr.size < expected:
                raise ValueError(f".raw too small: got {arr.size}, expected {expected}")
            if arr.size > expected:
                arr = arr[:expected]
            arr = arr.reshape(int(height), int(width)).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported raw_dtype: {raw_dtype}. Use uint16 or uint8.")
    elif suffix in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
        if not HAS_CV2:
            raise RuntimeError("Reading image RAW requires opencv-python.")
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read RAW image: {path}")
        if img.ndim == 3:
            if force_single_channel:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                if img.shape[2] == 3:
                    ch0, ch1, ch2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                    if np.array_equal(ch0, ch1) and np.array_equal(ch0, ch2):
                        img = ch0
                    else:
                        raise ValueError(f"3-channel input cannot be treated as Bayer RAW: {path}")
                else:
                    raise ValueError(f"Unsupported RAW image shape: {img.shape}")
            else:
                img = img[:, :, 0]
        if img.ndim != 2:
            raise ValueError(f"Expected single-channel Bayer RAW image, got shape={img.shape}")
        if img.dtype == np.uint8:
            inferred_white = 255.0
        elif img.dtype == np.uint16:
            inferred_white = 65535.0 if int(img.max()) > 4095 else 4095.0
        else:
            inferred_white = float(img.max()) if img.size > 0 and float(img.max()) > 0 else 1.0
        if white_level is None:
            white_level = inferred_white
        arr = img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported RAW input suffix: {suffix}")

    arr = arr.astype(np.float32)
    raw = (arr - float(black_level)) / (float(white_level) - float(black_level) + 1e-12)
    raw = np.clip(raw, 0.0, 1.0)
    print(
        f"[load_hs_raw] path={path}, shape={arr.shape}, dtype={raw_dtype}, "
        f"min={arr.min():.3f}, max={arr.max():.3f}, black={black_level}, white={white_level}"
    )
    return torch.from_numpy(raw)[None, None]


# ============================================================
# Color and demosaic helpers
# ============================================================

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * x.clamp_min(1e-6).pow(1 / 2.4) - a)


def bayer_site_masks(h, w, pattern="RGGB", device=None, dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    yy2 = yy & 1
    xx2 = xx & 1
    pattern = pattern.upper()
    if pattern == "RGGB":
        R = (yy2 == 0) & (xx2 == 0); Gr = (yy2 == 0) & (xx2 == 1); Gb = (yy2 == 1) & (xx2 == 0); B = (yy2 == 1) & (xx2 == 1)
    elif pattern == "BGGR":
        B = (yy2 == 0) & (xx2 == 0); Gb = (yy2 == 0) & (xx2 == 1); Gr = (yy2 == 1) & (xx2 == 0); R = (yy2 == 1) & (xx2 == 1)
    elif pattern == "GRBG":
        Gr = (yy2 == 0) & (xx2 == 0); R = (yy2 == 0) & (xx2 == 1); B = (yy2 == 1) & (xx2 == 0); Gb = (yy2 == 1) & (xx2 == 1)
    elif pattern == "GBRG":
        Gb = (yy2 == 0) & (xx2 == 0); B = (yy2 == 0) & (xx2 == 1); R = (yy2 == 1) & (xx2 == 0); Gr = (yy2 == 1) & (xx2 == 1)
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    def m(x):
        return x.to(dtype=dtype)[None, None]
    return m(R), m(B), m(Gr), m(Gb)


def _conv5_reflect(x, k):
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k)


def demosaic_malvar2004(raw: torch.Tensor, pattern="RGGB") -> torch.Tensor:
    _, _, H, W = raw.shape
    device, dtype = raw.device, raw.dtype
    Rm, Bm, Grm, Gbm = bayer_site_masks(H, W, pattern, device=device, dtype=dtype)
    Gm = Grm + Gbm
    kG = torch.tensor([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], device=device, dtype=dtype).view(1, 1, 5, 5) / 8.0
    kRG_rrow = torch.tensor([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], device=device, dtype=dtype).view(1, 1, 5, 5) / 8.0
    kRG_brow = torch.tensor([[0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0.5, 0, 5, 0, 0.5], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0]], device=device, dtype=dtype).view(1, 1, 5, 5) / 8.0
    kRB = torch.tensor([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], device=device, dtype=dtype).view(1, 1, 5, 5) / 8.0
    G_est = _conv5_reflect(raw, kG)
    G = raw * Gm + G_est * (1 - Gm)
    R = raw * Rm + _conv5_reflect(raw, kRG_rrow) * Grm + _conv5_reflect(raw, kRG_brow) * Gbm + _conv5_reflect(raw, kRB) * Bm
    Bc = raw * Bm + _conv5_reflect(raw, kRG_brow) * Grm + _conv5_reflect(raw, kRG_rrow) * Gbm + _conv5_reflect(raw, kRB) * Rm
    return torch.cat([R, G, Bc], dim=1).clamp(0, 1)


def srgb_to_lab_fast(srgb: torch.Tensor) -> torch.Tensor:
    lin = srgb_to_linear(srgb)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=lin.device, dtype=lin.dtype)
    B, _, H, W = lin.shape
    x = lin.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = (x @ M.T).reshape(B, H, W, 3).permute(0, 3, 1, 2)
    X = (xyz[:, 0] / 0.95047).clamp_min(0)
    Y = xyz[:, 1].clamp_min(0)
    Z = (xyz[:, 2] / 1.08883).clamp_min(0)
    delta = 6 / 29
    def f(t):
        return torch.where(t > delta ** 3, t.clamp_min(delta ** 3).pow(1 / 3), t / (3 * delta ** 2) + 4 / 29)
    fx, fy, fz = f(X), f(Y), f(Z)
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def gaussian_kernel1d(kernel_size=11, sigma=3.0, device="cpu", dtype=torch.float32):
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, kernel_size=11, sigma=3.0):
    if kernel_size <= 1:
        return x
    k1 = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    C = x.shape[1]
    x = F.pad(x, (kernel_size // 2, kernel_size // 2, 0, 0), mode="reflect")
    x = F.conv2d(x, kx.expand(C, 1, 1, kernel_size), groups=C)
    x = F.pad(x, (0, 0, kernel_size // 2, kernel_size // 2), mode="reflect")
    return F.conv2d(x, ky.expand(C, 1, kernel_size, 1), groups=C)


def normalized_blur(x: torch.Tensor, mask: torch.Tensor, kernel_size=11, sigma=3.0):
    mb = gaussian_blur(mask, kernel_size, sigma).clamp_min(1e-6)
    return gaussian_blur(x * mask, kernel_size, sigma) / mb, mb



def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
    diff = (pred - tgt).abs() * mask
    return diff.sum() / (mask.sum() * pred.shape[1] + 1e-6)


def masked_patch_stats_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, patch=16):
    num_p = F.avg_pool2d(pred * mask, kernel_size=patch, stride=patch)
    num_t = F.avg_pool2d(tgt * mask, kernel_size=patch, stride=patch)
    den = F.avg_pool2d(mask, kernel_size=patch, stride=patch).clamp_min(1e-6)
    mean_p, mean_t = num_p / den, num_t / den
    var_p = (F.avg_pool2d((pred ** 2) * mask, kernel_size=patch, stride=patch) / den - mean_p ** 2).clamp_min(0)
    var_t = (F.avg_pool2d((tgt ** 2) * mask, kernel_size=patch, stride=patch) / den - mean_t ** 2).clamp_min(0)
    patch_w = (den > 0.10).float()
    loss_mean = ((mean_p - mean_t).abs() * patch_w).sum() / (patch_w.sum() * pred.shape[1] + 1e-6)
    loss_var = ((var_p - var_t).abs() * patch_w).sum() / (patch_w.sum() * pred.shape[1] + 1e-6)
    return loss_mean + 0.5 * loss_var


def swd(points_a: torch.Tensor, points_b: torch.Tensor, n_proj=8):
    d = points_a.shape[1]
    dirs = torch.randn(n_proj, d, device=points_a.device, dtype=points_a.dtype)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
    pa, pb = points_a @ dirs.T, points_b @ dirs.T
    n = min(points_a.shape[0], points_b.shape[0])
    ia = torch.randperm(points_a.shape[0], device=points_a.device)[:n]
    ib = torch.randperm(points_b.shape[0], device=points_b.device)[:n]
    pa, pb = pa[ia], pb[ib]
    loss = points_a.new_tensor(0.0)
    for k in range(n_proj):
        ua, _ = torch.sort(pa[:, k])
        ub, _ = torch.sort(pb[:, k])
        loss = loss + (ua[:n] - ub[:n]).abs().mean()
    return loss / n_proj


def masked_swd_lab(pred_lab: torch.Tensor, tgt_lab: torch.Tensor, mask: torch.Tensor, max_samples=120000, n_proj=8):
    valid_idx = torch.nonzero(mask.flatten() > 0.10, as_tuple=False).flatten()
    if valid_idx.numel() < 128:
        return pred_lab.new_tensor(0.0)
    if valid_idx.numel() > max_samples:
        valid_idx = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_samples]]
    p = pred_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx]
    t = tgt_lab.permute(0, 2, 3, 1).reshape(-1, 3)[valid_idx]
    p2, t2 = p.clone(), t.clone()
    p2[:, 0] *= 0.6
    t2[:, 0] *= 0.6
    return swd(p2, t2, n_proj=n_proj)


def normalize_lab_for_hist(lab: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(lab)
    out[:, 0:1] = (lab[:, 0:1] / 100.0).clamp(0.0, 1.0)
    out[:, 1:2] = ((lab[:, 1:2] + 128.0) / 255.0).clamp(0.0, 1.0)
    out[:, 2:3] = ((lab[:, 2:3] + 128.0) / 255.0).clamp(0.0, 1.0)
    return out


def soft_histogram_loss_lab(
    pred_lab: torch.Tensor,
    tgt_lab: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 32,
    sigma: float = 0.035,
):
    """Differentiable masked Lab histogram loss."""
    pred_n = normalize_lab_for_hist(pred_lab)
    tgt_n = normalize_lab_for_hist(tgt_lab)
    weights = mask.detach().clamp(0.0, 1.0).flatten().clamp_min(0.0)
    valid = weights > 0.05
    if valid.sum() < 128:
        valid = torch.ones_like(weights, dtype=torch.bool)
        weights = torch.ones_like(weights)
    weights = weights[valid]
    centers = torch.linspace(0.0, 1.0, int(bins), device=pred_lab.device, dtype=pred_lab.dtype)
    sigma = max(float(sigma), 1e-4)
    loss = pred_lab.new_tensor(0.0)
    for c in range(3):
        p = pred_n[:, c:c + 1].flatten()[valid]
        t = tgt_n[:, c:c + 1].flatten()[valid]
        hp = torch.exp(-0.5 * ((p[:, None] - centers[None, :]) / sigma) ** 2) * weights[:, None]
        ht = torch.exp(-0.5 * ((t[:, None] - centers[None, :]) / sigma) ** 2) * weights[:, None]
        hp = hp.sum(dim=0)
        ht = ht.sum(dim=0)
        hp = hp / (hp.sum() + 1e-6)
        ht = ht / (ht.sum() + 1e-6)
        loss = loss + (hp - ht).abs().mean()
    return loss / 3.0

