#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from .common import *

try:
    from .warp import raft
    from .backwarp import forward_backward_consistency, flow_warp
except Exception:
    raft = None
    forward_backward_consistency = None
    flow_warp = None

# ============================================================
# Alignment
# ============================================================

def torch_rgb_to_numpy(rgb: torch.Tensor) -> np.ndarray:
    return rgb.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy().astype(np.float32)


def numpy_rgb_to_torch(x: np.ndarray, device=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float()
    return t.to(device) if device is not None else t


def numpy_gray_to_torch(x: np.ndarray, device=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None].float()
    return t.to(device) if device is not None else t


def _to_float01_3ch(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _robust01_np(x: np.ndarray, p_low=1.0, p_high=99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        lo, hi = float(np.min(x)), float(np.max(x) + 1e-6)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _clahe01(gray01: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray:
    gray_u8 = np.clip(gray01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray_u8).astype(np.float32) / 255.0


def _census_proxy01(gray01: np.ndarray) -> np.ndarray:
    g = np.asarray(gray01, dtype=np.float32)
    acc = np.zeros_like(g, dtype=np.float32)
    count = 0
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        acc += (shifted > g).astype(np.float32)
        count += 1
    return acc / max(count, 1)


def make_structural_proxy(img: np.ndarray, source: str = "raw") -> np.ndarray:
    if not HAS_CV2:
        raise RuntimeError("OpenCV not installed.")
    rgb = _to_float01_3ch(img)
    if str(source).lower() == "phone":
        y = cv2.cvtColor(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    else:
        y = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([-1], [0])).astype(np.float32)
        y = np.arcsinh(12.0 * np.clip(y, 0.0, 1.0)) / np.arcsinh(12.0)
    local = _clahe01(_robust01_np(y, 0.5, 99.5), clip_limit=2.0, tile_grid=(8, 8))
    blur = cv2.GaussianBlur(local, (0, 0), 1.6)
    high = np.clip(0.5 + 2.0 * (local - blur), 0.0, 1.0)
    gx = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
    mag = _robust01_np(np.sqrt(gx * gx + gy * gy), 1.0, 99.7)
    edges = cv2.Canny(np.clip(local * 255.0 + 0.5, 0, 255).astype(np.uint8), 50, 160).astype(np.float32) / 255.0
    census = _census_proxy01(local)
    edge_mix = np.clip(0.55 * mag + 0.30 * edges + 0.15 * census, 0.0, 1.0)
    return np.stack([local, high, edge_mix], axis=-1).astype(np.float32)


def prep_color_for_align(rgb: np.ndarray, color_space: str = "structure", source: str = "phone") -> np.ndarray:
    if str(color_space).lower() == "structure":
        return np.clip(make_structural_proxy(rgb, source=source) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x_u8 = np.clip(_to_float01_3ch(rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if str(color_space).lower() == "lab":
        lab = cv2.cvtColor(x_u8, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        return cv2.merge([L, A, B])
    if str(color_space).lower() == "rgb":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.merge([clahe.apply(c) for c in cv2.split(x_u8)])
    raise ValueError(f"Unsupported align_space: {color_space}")


def make_match_input(img: np.ndarray, source: str = "phone", align_space: str = "structure") -> np.ndarray:
    if str(align_space).lower() == "structure":
        return make_structural_proxy(img, source=source)
    return prep_color_for_align(img, color_space=align_space, source=source).astype(np.float32) / 255.0


def resize_keep_max_side_img(img: np.ndarray, max_side: int = 1024):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img.copy(), 1.0, 1.0
    scale = float(max_side) / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), new_w / float(w), new_h / float(h)


def scale_homography_to_fullres(H_small, sx_m, sy_m, sx_t, sy_t):
    S_m = np.array([[sx_m, 0, 0], [0, sy_m, 0], [0, 0, 1]], dtype=np.float32)
    S_t_inv = np.array([[1.0 / sx_t, 0, 0], [0, 1.0 / sy_t, 0], [0, 0, 1]], dtype=np.float32)
    return (S_t_inv @ H_small.astype(np.float32) @ S_m).astype(np.float32)


def estimate_homography_feature_rgb(moving_rgb_3ch: np.ndarray,
                                    ref_rgb_3ch: np.ndarray,
                                    ratio_thresh: float = 0.75,
                                    ransac_reproj_thresh: float = 3.0,
                                    max_kp_per_channel: int = 3000,
                                    min_homography_inlier_ratio: float = 0.25):
    use_sift = hasattr(cv2, "SIFT_create")
    detector = cv2.SIFT_create(nfeatures=max_kp_per_channel) if use_sift else cv2.ORB_create(nfeatures=max_kp_per_channel)
    matcher = cv2.BFMatcher(cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING, crossCheck=False)
    src_pts_all, dst_pts_all, per_channel_info = [], [], []
    for c in range(3):
        kp1, des1 = detector.detectAndCompute(moving_rgb_3ch[:, :, c], None)
        kp2, des2 = detector.detectAndCompute(ref_rgb_3ch[:, :, c], None)
        n1, n2 = 0 if kp1 is None else len(kp1), 0 if kp2 is None else len(kp2)
        good = []
        if des1 is not None and des2 is not None and n1 >= 4 and n2 >= 4:
            for m_n in matcher.knnMatch(des1, des2, k=2):
                if len(m_n) >= 2 and m_n[0].distance < ratio_thresh * m_n[1].distance:
                    good.append(m_n[0])
            if len(good) >= 4:
                src_pts_all.append(np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2))
                dst_pts_all.append(np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2))
        per_channel_info.append({"channel": c, "n_kp_moving": n1, "n_kp_ref": n2, "n_good": len(good)})
    if not src_pts_all:
        return np.eye(3, dtype=np.float32), {"ok": False, "reason": "no_matches", "per_channel": per_channel_info, "n_total_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0}
    src_pts_all, dst_pts_all = np.concatenate(src_pts_all, axis=0), np.concatenate(dst_pts_all, axis=0)
    M_aff, aff_mask = cv2.estimateAffinePartial2D(src_pts_all, dst_pts_all, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh, maxIters=4000, confidence=0.999, refineIters=30)
    affine_inliers = 0 if aff_mask is None else int(aff_mask.sum())
    H, inlier_mask = cv2.findHomography(src_pts_all, dst_pts_all, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh)
    if H is None or inlier_mask is None:
        if M_aff is not None and affine_inliers >= 4:
            H_aff = np.eye(3, dtype=np.float32); H_aff[:2] = M_aff.astype(np.float32)
            return H_aff, {"ok": True, "reason": "homography_failed_affine_fallback", "model": "affine_fallback", "per_channel": per_channel_info, "n_total_matches": int(len(src_pts_all)), "n_inliers": affine_inliers, "inlier_ratio": float(affine_inliers / max(len(src_pts_all), 1)), "affine_inliers": affine_inliers}
        return np.eye(3, dtype=np.float32), {"ok": False, "reason": "findHomography_failed", "model": "identity", "per_channel": per_channel_info, "n_total_matches": int(len(src_pts_all)), "n_inliers": 0, "inlier_ratio": 0.0, "affine_inliers": affine_inliers}
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    n_inliers, n_total = int(inlier_mask.sum()), int(len(inlier_mask))
    inlier_ratio = float(n_inliers / max(n_total, 1))
    if inlier_ratio < min_homography_inlier_ratio and M_aff is not None and affine_inliers >= 4:
        H_aff = np.eye(3, dtype=np.float32); H_aff[:2] = M_aff.astype(np.float32)
        return H_aff, {"ok": True, "reason": "low_inlier_affine_fallback", "model": "affine_fallback", "per_channel": per_channel_info, "n_total_matches": n_total, "n_inliers": affine_inliers, "inlier_ratio": float(affine_inliers / max(n_total, 1)), "homography_inliers": n_inliers, "affine_inliers": affine_inliers}
    return H.astype(np.float32), {"ok": True, "reason": "success", "model": "homography", "per_channel": per_channel_info, "n_total_matches": n_total, "n_inliers": n_inliers, "inlier_ratio": inlier_ratio, "affine_inliers": affine_inliers}


 

def estimate_homography_multiscale(moving_align: np.ndarray, ref_align: np.ndarray, align_max_side: int = 1024):
    best, candidates = None, []
    candidate_sides = []
    for side in (512, int(align_max_side), 1536):
        if side > 0 and side not in candidate_sides:
            candidate_sides.append(side)
    for side in candidate_sides:
        ref_s, sx_t, sy_t = resize_keep_max_side_img(ref_align, side)
        mov_s, sx_m, sy_m = resize_keep_max_side_img(moving_align, side)
        H_small, info = estimate_homography_feature_rgb(mov_s, ref_s)
        H_full = scale_homography_to_fullres(H_small, sx_m, sy_m, sx_t, sy_t)
        score = float(info.get("n_inliers", 0)) * max(float(info.get("inlier_ratio", 0.0)), 1e-3) + 0.25 * float(info.get("affine_inliers", 0))
        candidates.append({"max_side": int(side), "score": float(score), "info": info})
        if best is None or score > best[0]:
            best = (score, H_full, info, side)
    if best is None:
        return np.eye(3, dtype=np.float32), {"ok": False, "reason": "no_candidate", "model": "identity", "candidates": candidates}
    _, H_full, info, side = best
    out = dict(info); out["selected_max_side"] = int(side); out["candidates"] = candidates
    return H_full.astype(np.float32), out


def warp_perspective_and_mask(rgb: np.ndarray, H: np.ndarray, out_h: int, out_w: int):
    warped = cv2.warpPerspective(rgb.astype(np.float32), H.astype(np.float32), (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpPerspective(np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32), H.astype(np.float32), (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(np.float32), np.clip(valid, 0, 1)[..., None].astype(np.float32)


def resize_to_canvas_and_mask(rgb: np.ndarray, out_h: int, out_w: int):
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if out_h * out_w < h * w else cv2.INTER_LINEAR
    resized = cv2.resize(rgb.astype(np.float32), (out_w, out_h), interpolation=interp)
    valid = np.ones((out_h, out_w, 1), dtype=np.float32)
    return resized.astype(np.float32), valid


def _np_rgb01_to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(_to_float01_3ch(rgb), 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _roma_match_rgb(rgb: np.ndarray, source: str, align_space: str) -> np.ndarray:
    if str(align_space).lower() == "structure":
        return make_structural_proxy(rgb, source=source)
    if str(align_space).lower() == "lab":
        return prep_color_for_align(rgb, color_space="lab", source=source).astype(np.float32) / 255.0
    return _to_float01_3ch(rgb)


def _resize_grid_like(grid: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if grid.shape[1] == out_h and grid.shape[2] == out_w:
        return grid
    return F.interpolate(grid.permute(0, 3, 1, 2), size=(out_h, out_w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def build_roma_model(device: torch.device, variant: str = "outdoor", coarse_res: int = 560, upsample_res: int = 560):
    try:
        from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor
    except Exception as exc:
        raise RuntimeError("RoMa requires the `romatch` package. Install it with `pip install romatch`.") from exc

    variant = str(variant).lower()
    coarse_res = int(coarse_res)
    upsample_res = int(upsample_res)
    if variant == "outdoor":
        model = roma_outdoor(device=str(device), coarse_res=(coarse_res, coarse_res), upsample_res=(upsample_res, upsample_res), use_custom_corr=False)
    elif variant == "indoor":
        model = roma_indoor(device=str(device), coarse_res=(coarse_res, coarse_res), upsample_res=(upsample_res, upsample_res), use_custom_corr=False)
    elif variant == "tiny_outdoor":
        model = tiny_roma_v1_outdoor(device=str(device))
    else:
        raise ValueError(f"Unsupported roma_variant: {variant}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def roma_warp_to_reference(reference_rgb, phone_rgb, roma_model, out_dir: Path, device="cuda", align_space="rgb", certainty_threshold=0.0, certainty_power=1.0):
    if roma_model is None:
        raise ValueError("--alignment_mode roma requires a loaded RoMa model")
    reference_rgb = _to_float01_3ch(reference_rgb)
    phone_rgb = _to_float01_3ch(phone_rgb)
    Ht, Wt = reference_rgb.shape[:2]
    phone_canvas, canvas_valid = resize_to_canvas_and_mask(phone_rgb, out_h=Ht, out_w=Wt)

    ref_match = _roma_match_rgb(reference_rgb, source="raw", align_space=align_space)
    mov_match = _roma_match_rgb(phone_canvas, source="phone", align_space=align_space)
    with torch.no_grad():
        warp, certainty = roma_model.match(_np_rgb01_to_pil(ref_match), _np_rgb01_to_pil(mov_match), device=str(device))
        if warp.ndim == 3:
            warp = warp[None]
        if certainty.ndim == 2:
            certainty = certainty[None]
        out_w2 = int(warp.shape[2])
        if bool(getattr(roma_model, "symmetric", False)):
            w_half = out_w2 // 2
            sample_grid = warp[:, :, :w_half, 2:]
            cert = certainty[:, :, :w_half]
        else:
            sample_grid = warp[..., 2:]
            cert = certainty
        sample_grid = _resize_grid_like(sample_grid.float(), Ht, Wt).clamp(-1, 1)
        cert = F.interpolate(cert[:, None].float(), size=(Ht, Wt), mode="bilinear", align_corners=False).clamp(0, 1)
        source_t = numpy_rgb_to_torch(phone_canvas, device=device).clamp(0, 1)
        warped_t = F.grid_sample(source_t, sample_grid.to(source_t.device), mode="bilinear", padding_mode="zeros", align_corners=False)
        bound_t = ((sample_grid[..., 0].abs() <= 1.0) & (sample_grid[..., 1].abs() <= 1.0)).float()[:, None].to(source_t.device)
        cert_t = cert.to(source_t.device)
        if float(certainty_power) != 1.0:
            cert_t = cert_t.clamp_min(0).pow(float(certainty_power)).clamp(0, 1)
        if float(certainty_threshold) > 0.0:
            cert_t = cert_t * (cert_t >= float(certainty_threshold)).float()
        canvas_valid_t = numpy_gray_to_torch(canvas_valid[..., 0], device=device).clamp(0, 1)
        valid_t = (canvas_valid_t * bound_t * cert_t).clamp(0, 1)

    save_rgb(str(out_dir / "roma_phone_warped.png"), warped_t)
    save_mask(str(out_dir / "roma_certainty.png"), cert_t)
    save_mask(str(out_dir / "roma_bound_mask.png"), bound_t)
    save_mask(str(out_dir / "roma_valid_mask.png"), valid_t)
    feat_info = {
        "ok": True,
        "reason": "roma_dense_match",
        "model": f"roma_{getattr(roma_model, 'name', 'model')}",
        "source_hw": [int(phone_rgb.shape[0]), int(phone_rgb.shape[1])],
        "source_canvas_hw": [int(phone_canvas.shape[0]), int(phone_canvas.shape[1])],
        "target_hw": [int(Ht), int(Wt)],
        "certainty_mean": float(cert_t.mean().item()),
        "certainty_p50": float(torch.quantile(cert_t.flatten(), 0.50).item()),
        "certainty_p90": float(torch.quantile(cert_t.flatten(), 0.90).item()),
        "certainty_threshold": float(certainty_threshold),
        "certainty_power": float(certainty_power),
        "bound_valid_ratio": float(bound_t.mean().item()),
    }
    return torch_rgb_to_numpy(warped_t), valid_t[0, 0].detach().cpu().numpy()[..., None].astype(np.float32), feat_info


def warp_source_with_fb_flow_inputs(source_rgb, source_flow_in, target_flow_in, optic, fb_alpha=0.01, fb_beta=0.5, flow_iters=32):
    with torch.no_grad():
        _, flow_t2s = optic(target_flow_in, source_flow_in, iters=int(flow_iters), test_mode=True)
        _, flow_s2t = optic(source_flow_in, target_flow_in, iters=int(flow_iters), test_mode=True)
        warped_source, bound_mask = flow_warp(source_rgb, flow_t2s)
        fb_mask, fb_error, flow_s2t_w, valid_t2s = forward_backward_consistency(flow_t2s, flow_s2t, alpha=fb_alpha, beta=fb_beta)
        valid_mask = bound_mask * fb_mask
        debug = {
            "bound_mask": bound_mask,
            "fb_mask": fb_mask,
            "fb_error": fb_error,
            "valid_t2s": valid_t2s,
            "flow_t2s_mag": torch.sqrt((flow_t2s ** 2).sum(dim=1, keepdim=True) + 1e-12),
            "flow_s2t_w_mag": torch.sqrt((flow_s2t_w ** 2).sum(dim=1, keepdim=True) + 1e-12),
        }
        return warped_source, valid_mask, fb_error, debug


def _torch_mask_to_np(mask_t: torch.Tensor) -> np.ndarray:
    return mask_t[0, 0].detach().float().cpu().numpy()[..., None].astype(np.float32)


def _save_flow_debug_maps(out_dir: Path, flow_debug: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bound_mask", "fb_mask", "valid_t2s"):
        if name in flow_debug:
            m = _torch_mask_to_np(flow_debug[name])[..., 0]
            cv2.imwrite(str(out_dir / f"flow_{name}.png"), (np.clip(m, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    for name in ("fb_error", "flow_t2s_mag", "flow_s2t_w_mag"):
        if name not in flow_debug:
            continue
        x = _torch_mask_to_np(flow_debug[name])[..., 0]
        x = np.log1p(np.maximum(x, 0.0))
        hi = float(np.percentile(x[np.isfinite(x)], 99.0)) if np.isfinite(x).any() else 1.0
        vis = np.clip(x / max(hi, 1e-6), 0, 1)
        heat = cv2.applyColorMap((vis * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_dir / f"flow_{name}.png"), heat)


def refine_with_structural_flow(reference_rgb, moving_rgb, optic, device="cuda", fb_alpha=0.01, fb_beta=0.5, align_space="structure", flow_iters=32):
    ref_flow_np = make_match_input(reference_rgb, source="raw", align_space=align_space)
    mov_flow_np = make_match_input(moving_rgb, source="phone", align_space=align_space)
    ref_flow_t = numpy_rgb_to_torch(ref_flow_np, device=device).clamp(0, 1)
    mov_flow_t = numpy_rgb_to_torch(mov_flow_np, device=device).clamp(0, 1)
    mov_rgb_t = numpy_rgb_to_torch(_to_float01_3ch(moving_rgb), device=device).clamp(0, 1)
    warped_rgb_t, valid_mask_t, fb_error, flow_debug = warp_source_with_fb_flow_inputs(mov_rgb_t, mov_flow_t, ref_flow_t, optic, fb_alpha, fb_beta, flow_iters)
    flow_info = {
        "flow_valid_ratio": float(valid_mask_t.mean().item()),
        "flow_bound_valid_ratio": float(flow_debug["bound_mask"].mean().item()),
        "flow_fb_consistency_ratio": float(flow_debug["fb_mask"].mean().item()),
        "flow_valid_t2s_ratio": float(flow_debug["valid_t2s"].mean().item()),
        "fb_error_mean": float(fb_error.mean().item()),
    }
    return torch_rgb_to_numpy(warped_rgb_t), valid_mask_t[0, 0].detach().cpu().numpy()[..., None].astype(np.float32), flow_info, flow_debug


def masked_ncc_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray, eps=1e-6) -> float:
    a, b, m = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    w = (m[..., None] if a.ndim == 3 else m) > 0.2
    w = w.astype(np.float32)
    if float(w.sum()) < 16:
        return 0.0
    ma, mb = (a * w).sum() / (w.sum() + eps), (b * w).sum() / (w.sum() + eps)
    da, db = (a - ma) * w, (b - mb) * w
    return float((da * db).sum() / math.sqrt(max(float((da * da).sum() * (db * db).sum()), eps)))


def structural_alignment_metrics(reference_rgb, aligned_rgb, valid, align_space="structure"):
    ref_struct = make_match_input(reference_rgb, source="raw", align_space=align_space)
    ali_struct = make_match_input(aligned_rgb, source="phone", align_space=align_space)
    return {
        "structural_ncc": float(masked_ncc_np(ref_struct, ali_struct, valid)),
        "edge_ncc": float(masked_ncc_np(ref_struct[..., 2], ali_struct[..., 2], valid)),
        "confidence_mean": float(np.asarray(valid, dtype=np.float32).mean()),
        "accepted_mask_ratio": float((np.asarray(valid, dtype=np.float32) > 0.5).mean()),
    } 


def _rgb_luminance(rgb: torch.Tensor) -> torch.Tensor:
    w = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb.clamp(0, 1) * w).sum(1, keepdim=True)


def local_ncc_map(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, kernel_size=41, sigma=9.0, eps=1e-6) -> torch.Tensor:
    kernel_size = max(3, int(kernel_size) | 1)
    mask = mask.detach().clamp(0, 1)
    denom = gaussian_blur(mask, kernel_size, sigma).clamp_min(eps)
    ma = gaussian_blur(a * mask, kernel_size, sigma) / denom
    mb = gaussian_blur(b * mask, kernel_size, sigma) / denom
    da = (a - ma) * mask
    db = (b - mb) * mask
    cov = gaussian_blur(da * db, kernel_size, sigma) / denom
    va = gaussian_blur(da * da, kernel_size, sigma) / denom
    vb = gaussian_blur(db * db, kernel_size, sigma) / denom
    ncc = cov / (torch.sqrt(va * vb + eps))
    return ((ncc + 1.0) * 0.5).clamp(0, 1)


def compute_structure_consistency_mask(
    reference_rgb: torch.Tensor,
    aligned_phone: torch.Tensor,
    valid_mask: torch.Tensor,
    kernel_size: int = 41,
    sigma: float = 9.0,
    ncc_threshold: float = 0.45,
    temperature: float = 0.08,
    edge_percentile: float = 60.0,
    min_weight: float = 0.20,
):
    """Fixed soft mask for aligned-phone supervision reliability.

    Flat low-structure areas keep high weight; textured areas are down-weighted
    when local luminance/edge NCC between reference and aligned phone is low.
    """
    with torch.no_grad():
        kernel_size = max(3, int(kernel_size) | 1)
        sigma = float(max(sigma, 1e-3))
        valid = valid_mask.detach().clamp(0, 1)
        ref_y = _rgb_luminance(reference_rgb.detach()).clamp(0, 1)
        phone_y = _rgb_luminance(aligned_phone.detach()).clamp(0, 1)
        ref_y = gaussian_blur(ref_y, 5, 1.0)
        phone_y = gaussian_blur(phone_y, 5, 1.0)
        gray_score = local_ncc_map(ref_y, phone_y, valid, kernel_size=kernel_size, sigma=sigma)

        ref_edge = sobel_mag(ref_y)
        phone_edge = sobel_mag(phone_y)
        edge_score = local_ncc_map(ref_edge, phone_edge, valid, kernel_size=kernel_size, sigma=sigma)
        score = (0.55 * gray_score + 0.45 * edge_score).clamp(0, 1)

        edge_support = gaussian_blur(torch.maximum(ref_edge, phone_edge), kernel_size, sigma).clamp_min(0)
        valid_flat = valid.flatten() > 0.2
        if int(valid_flat.sum().item()) >= 128:
            edge_thr = torch.quantile(edge_support.flatten()[valid_flat], float(np.clip(edge_percentile, 1.0, 99.0)) / 100.0)
        else:
            edge_thr = edge_support.flatten().median()
        edge_thr = edge_thr.clamp_min(1e-6)
        support_gate = torch.sigmoid((edge_support - edge_thr) / (0.30 * edge_thr + 1e-6)).clamp(0, 1)

        score_gate = torch.sigmoid((score - float(ncc_threshold)) / max(float(temperature), 1e-4)).clamp(0, 1)
        min_weight = float(np.clip(min_weight, 0.0, 1.0))
        reliable_structured = min_weight + (1.0 - min_weight) * score_gate
        weight = ((1.0 - support_gate) + support_gate * reliable_structured).clamp(0, 1)

        valid_pixels = valid_flat
        if int(valid_pixels.sum().item()) < 128:
            valid_pixels = torch.ones_like(valid_pixels, dtype=torch.bool)
        weight_valid = weight.flatten()[valid_pixels]
        score_valid = score.flatten()[valid_pixels]
        support_valid = support_gate.flatten()[valid_pixels]
        info = {
            "enabled": True,
            "kernel_size": int(kernel_size),
            "sigma": float(sigma),
            "ncc_threshold": float(ncc_threshold),
            "temperature": float(temperature),
            "edge_percentile": float(edge_percentile),
            "edge_threshold": float(edge_thr),
            "min_weight": float(min_weight),
            "mask_mean": float(weight.mean()),
            "mask_valid_mean": float(weight_valid.mean()),
            "mask_valid_lt_0p5": float((weight_valid < 0.5).float().mean()),
            "mask_valid_lt_0p8": float((weight_valid < 0.8).float().mean()),
            "score_valid_mean": float(score_valid.mean()),
            "support_gate_valid_mean": float(support_valid.mean()),
        }
        debug = {
            "gray_ncc": gray_score,
            "edge_ncc": edge_score,
            "score": score,
            "edge_support": edge_support / (edge_thr + 1e-6),
            "support_gate": support_gate,
            "combined_valid_mask": (valid * weight).clamp(0, 1),
        }
        return weight, info, debug


def combine_supervision_weight(base: torch.Tensor | None, extra: torch.Tensor):
    extra = extra.detach().clamp(0, 1)
    if base is None:
        return extra
    return (base.detach().clamp(0, 1) * extra).clamp(0, 1)


def align_phone_to_reference(
    reference_rgb,
    phone_rgb,
    optic,
    out_dir: Path,
    align_max_side=1024,
    do_local_flow=True,
    device="cuda",
    fb_alpha=0.01,
    fb_beta=0.5,
    align_space="structure",
    flow_iters=32,
    debug=False,
    alignment_mode="homography",
    roma_model=None,
    roma_certainty_threshold=0.0,
    roma_certainty_power=1.0,
):
    reference_rgb = _to_float01_3ch(reference_rgb)
    phone_rgb = _to_float01_3ch(phone_rgb)
    Ht, Wt = reference_rgb.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_align = prep_color_for_align(reference_rgb, color_space=align_space, source="raw")
    phone_align = prep_color_for_align(phone_rgb, color_space=align_space, source="phone")
    if debug:
        cv2.imwrite(str(out_dir / "ref_aug.png"), cv2.cvtColor(ref_align, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "phone_aug.png"), cv2.cvtColor(phone_align, cv2.COLOR_RGB2BGR))
    alignment_mode = str(alignment_mode).lower()
    if alignment_mode == "flow_only":
        if not do_local_flow or optic is None:
            raise ValueError("--alignment_mode flow_only requires local flow / RAFT")
        phone_warped, base_valid = resize_to_canvas_and_mask(phone_rgb, out_h=Ht, out_w=Wt)
        feat_info = {
            "ok": True,
            "reason": "sift_skipped_direct_resize",
            "model": "direct_resize",
            "source_hw": [int(phone_rgb.shape[0]), int(phone_rgb.shape[1])],
            "target_hw": [int(Ht), int(Wt)],
            "n_total_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "selected_max_side": 0,
        }
    elif alignment_mode == "homography":
        H_full, feat_info = estimate_homography_multiscale(phone_align, ref_align, align_max_side=align_max_side)
        phone_warped, base_valid = warp_perspective_and_mask(phone_rgb, H_full, out_h=Ht, out_w=Wt)
    elif alignment_mode == "roma":
        phone_warped, base_valid, feat_info = roma_warp_to_reference(
            reference_rgb,
            phone_rgb,
            roma_model=roma_model,
            out_dir=out_dir,
            device=device,
            align_space=align_space,
            certainty_threshold=roma_certainty_threshold,
            certainty_power=roma_certainty_power,
        )
    else:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
    if debug:
        cv2.imwrite(str(out_dir / "phone_warped.png"), cv2.cvtColor((np.clip(phone_warped, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "base_valid.png"), (np.clip(base_valid.squeeze(-1), 0, 1) * 255).astype(np.uint8))
    if do_local_flow:
        phone_refined, flow_valid, flow_info, flow_debug = refine_with_structural_flow(reference_rgb, phone_warped, optic, device=device, fb_alpha=fb_alpha, fb_beta=fb_beta, align_space=align_space, flow_iters=flow_iters)
        _save_flow_debug_maps(out_dir, flow_debug)
        valid = base_valid * flow_valid
        aligned = phone_refined
        if alignment_mode == "flow_only":
            mode = "flow_only_resize_structure" if str(align_space).lower() == "structure" else f"flow_only_resize_{align_space}"
        elif alignment_mode == "roma":
            mode = "roma_flow_structure" if str(align_space).lower() == "structure" else f"roma_flow_{align_space}"
        else:
            mode = "homography_flow_structure" if str(align_space).lower() == "structure" else f"homography_flow_{align_space}"
    else:
        valid, aligned, flow_info = base_valid, phone_warped, {"used": False}
        if alignment_mode == "roma":
            mode = "roma_only_structure" if str(align_space).lower() == "structure" else f"roma_only_{align_space}"
        else:
            mode = "homography_only_structure" if str(align_space).lower() == "structure" else f"homography_only_{align_space}"
    metrics = structural_alignment_metrics(reference_rgb, aligned, valid, align_space=align_space)
    info = {"mode": mode, "feature_info": feat_info, "base_valid_ratio": float(base_valid.mean()), "flow_info": flow_info, "valid_ratio": float(valid.mean()), "align_space": align_space, "flow_fb_error_mean": float(flow_info.get("fb_error_mean", 0.0)), **metrics}
    return np.clip(aligned, 0, 1).astype(np.float32), valid.astype(np.float32), info


def build_custom_raft(device: torch.device, ckpt_path: str):
    raft_args = argparse.Namespace()
    raft_args.small = False
    raft_args.mixed_precision = False
    raft_args.alternate_corr = False
    raft_args.dropout = 0.0
    optic = raft.RAFT(raft_args).to(device)
    for p in optic.parameters():
        p.requires_grad = False
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    optic.load_state_dict(ckpt, strict=True)
    optic.eval()
    return optic


def refresh_alignment(
    reference_t,
    phone_t,
    optic,
    out_dir,
    device,
    align_max_side,
    do_local_flow,
    fb_alpha,
    fb_beta,
    align_space="structure",
    flow_iters=32,
    debug=False,
    alignment_mode="homography",
    roma_model=None,
    roma_certainty_threshold=0.0,
    roma_certainty_power=1.0,
):
    aligned_np, valid_np, info = align_phone_to_reference(
        torch_rgb_to_numpy(reference_t),
        torch_rgb_to_numpy(phone_t),
        optic=optic,
        out_dir=out_dir,
        align_max_side=align_max_side,
        do_local_flow=do_local_flow,
        device=str(device),
        fb_alpha=fb_alpha,
        fb_beta=fb_beta,
        align_space=align_space,
        flow_iters=flow_iters,
        debug=debug,
        alignment_mode=alignment_mode,
        roma_model=roma_model,
        roma_certainty_threshold=roma_certainty_threshold,
        roma_certainty_power=roma_certainty_power,
    )
    return numpy_rgb_to_torch(aligned_np, device=device).clamp(0, 1), numpy_gray_to_torch(valid_np[..., 0], device=device).clamp(0, 1), info

