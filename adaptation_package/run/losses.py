#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import *
from .isp import *

def build_loss_static_cache(
    raw,
    tgt_srgb_full,
    valid_mask_full,
    src_lin_full,
    down=256,
    raw_dark_threshold=8.0 / 1023.0,
    raw_saturation_threshold=0.95,
    texture_percentile=0.0,
    phone_saturation_threshold=0.98,
    base_supervision_mask_full=None,
):
    with torch.no_grad():
        tgt = F.interpolate(tgt_srgb_full, size=(down, down), mode="area")
        src_lin = F.interpolate(src_lin_full, size=(down, down), mode="area")
        raw_valid_full = ((raw.detach() > float(raw_dark_threshold)) & (raw.detach() < float(raw_saturation_threshold))).float()
        raw_valid_small = F.interpolate(raw_valid_full, size=(down, down), mode="area").clamp(0, 1)
        wp = tgt.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        src_y = (src_lin.detach() * wp).sum(1, keepdim=True)
        tgt_max = tgt.detach().amax(dim=1, keepdim=True)
        supervision_mask_full = valid_mask_full
        if base_supervision_mask_full is not None:
            supervision_mask_full = (valid_mask_full * base_supervision_mask_full).clamp(0, 1)
        align_mask = gaussian_blur(F.interpolate(supervision_mask_full, size=(down, down), mode="area").detach(), 5, 1.0).clamp(0, 1)
        tex = sobel_mag(gaussian_blur(src_y, 5, 1.0)).detach()
        tex_thr = torch.quantile(tex.flatten(), float(np.clip(texture_percentile, 0, 99)) / 100.0) if tex.numel() > 0 else tex.new_tensor(0.0)
        texture_mask = (tex >= tex_thr).float()
        phone_not_saturated = (tgt_max < float(phone_saturation_threshold)).float()
        base_mask = (align_mask * raw_valid_small * texture_mask * phone_not_saturated).clamp(0, 1)
        return {
            "down": int(down),
            "tgt": tgt.detach(),
            "tgt_lab": srgb_to_lab_fast(tgt).detach(),
            "src_lin": src_lin.detach(),
            "src_y": src_y.detach(),
            "raw_valid_small": raw_valid_small.detach(),
            "supervision_mask_mean": float(supervision_mask_full.detach().mean()),
            "align_mask": align_mask.detach(),
            "texture_mask": texture_mask.detach(),
            "phone_not_saturated": phone_not_saturated.detach(),
            "base_mask": base_mask.detach(),
            "raw_valid_mean": float(raw_valid_small.detach().mean()),
        }


def total_loss_residual_v4(raw, exp_t, tgt_srgb_full, valid_mask_full, isp, src_lin_full, down=256, patch=32, lam_low=0.6, lam_patch=0.6, lam_swd=0.3, swd_max_samples=120000, color_loss_mode="current", lam_hist=1.0, hist_bins=32, hist_sigma=0.035, lam_struct=0.1, lam_reg=0.01, lam_range=0.02, lam_lut_smooth=0.01, lam_lut_mono=0.01, raw_dark_threshold=8.0 / 1023.0, raw_saturation_threshold=0.95, texture_percentile=0.0, phone_saturation_threshold=0.98, lam_neutral=0.0, uncertainty_mask_weight=1.0, uncertainty_mask_power=1.0, uncertainty_loss_routing="all", lam_uncertainty_reg=0.0, base_supervision_mask_full=None, loss_output_key="final", static_cache=None, ltm_ev_weight=1e-3, ltm_ev_tv_weight=1e-3):
    dbg = isp.forward_debug(raw, exp_t)
    if loss_output_key not in dbg:
        raise KeyError(f"loss_output_key={loss_output_key!r} not found in ISP debug outputs: {sorted(dbg.keys())}")
    pred_srgb = dbg[loss_output_key].clamp(0, 1)
    pred = F.interpolate(pred_srgb, size=(down, down), mode="area")
    if static_cache is not None:
        tgt = static_cache["tgt"]
        tgt_lab = static_cache["tgt_lab"]
        src_lin = static_cache["src_lin"]
        src_y = static_cache["src_y"]
        raw_valid_small = static_cache["raw_valid_small"]
        base_mask = static_cache["base_mask"]
        supervision_mask_mean = float(static_cache["supervision_mask_mean"])
        raw_valid_mean = float(static_cache["raw_valid_mean"])
    else:
        tgt = F.interpolate(tgt_srgb_full, size=(down, down), mode="area")
        src_lin = F.interpolate(src_lin_full, size=(down, down), mode="area")
        raw_valid_full = ((raw.detach() > float(raw_dark_threshold)) & (raw.detach() < float(raw_saturation_threshold))).float()
        raw_valid_small = F.interpolate(raw_valid_full, size=(down, down), mode="area").clamp(0, 1)
        wp = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        src_y = (src_lin.detach() * wp).sum(1, keepdim=True)
        tgt_max = tgt.detach().amax(dim=1, keepdim=True)
        supervision_mask_full = valid_mask_full
        if base_supervision_mask_full is not None:
            supervision_mask_full = (valid_mask_full * base_supervision_mask_full).clamp(0, 1)
        align_mask = gaussian_blur(F.interpolate(supervision_mask_full, size=(down, down), mode="area").detach(), 5, 1.0).clamp(0, 1)
        tex = sobel_mag(gaussian_blur(src_y, 5, 1.0)).detach()
        tex_thr = torch.quantile(tex.flatten(), float(np.clip(texture_percentile, 0, 99)) / 100.0) if tex.numel() > 0 else tex.new_tensor(0.0)
        texture_mask = (tex >= tex_thr).float()
        phone_not_saturated = (tgt_max < float(phone_saturation_threshold)).float()
        base_mask = (align_mask * raw_valid_small * texture_mask * phone_not_saturated).clamp(0, 1)
        tgt_lab = srgb_to_lab_fast(tgt)
        supervision_mask_mean = float(supervision_mask_full.detach().mean())
        raw_valid_mean = float(raw_valid_small.detach().mean())
    wp = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    pixel_mask = base_mask
    dist_mask = base_mask
    if "raw_confidence" in dbg and float(uncertainty_mask_weight) > 0:
        raw_conf = gaussian_blur(F.interpolate(dbg["raw_confidence"].detach(), size=(down, down), mode="area"), 5, 1.0).clamp(0, 1)
        conf_factor = ((1.0 - float(uncertainty_mask_weight)) + float(uncertainty_mask_weight) * raw_conf.pow(float(uncertainty_mask_power))).clamp(0, 1)
        routing = str(uncertainty_loss_routing).lower()
        if routing in ("all", "mask_all", "legacy"):
            pixel_mask = base_mask * conf_factor
            dist_mask = pixel_mask
        elif routing in ("pixel", "pixel_only", "low_only", "routing"):
            pixel_mask = base_mask * conf_factor
            dist_mask = base_mask
        elif routing in ("none", "off", "disabled"):
            pixel_mask = base_mask
            dist_mask = base_mask
        else:
            raise ValueError(f"Unsupported uncertainty_loss_routing: {uncertainty_loss_routing}")
    pixel_mask = gaussian_blur(pixel_mask, 5, 1.0).clamp(0, 1)
    dist_mask = gaussian_blur(dist_mask, 5, 1.0).clamp(0, 1)
    mask = pixel_mask
    color_loss_mode = str(color_loss_mode).lower()
    pred_lab = srgb_to_lab_fast(pred)
    L_low = pred.new_tensor(0.0)
    L_patch = pred.new_tensor(0.0)
    L_swd = pred.new_tensor(0.0)
    L_hist = pred.new_tensor(0.0)
    L_neutral = pred.new_tensor(0.0)
    if color_loss_mode in ("current", "current_hist", "current+hist", "hybrid"):
        if float(lam_low) != 0.0:
            pred_blur, mask_blur = normalized_blur(pred_lab, pixel_mask, 11, 3.0)
            tgt_blur, _ = normalized_blur(tgt_lab, pixel_mask, 11, 3.0)
            L_low = (((pred_blur - tgt_blur).abs() * pred.new_tensor([0.35, 1.0, 1.0]).view(1, 3, 1, 1)) * (mask_blur > 0.05).float()).sum() / ((mask_blur > 0.05).float().sum() * 3.0 + 1e-6)
        if float(lam_patch) != 0.0:
            L_patch = masked_patch_stats_loss(pred_lab, tgt_lab, dist_mask, patch=patch)
        if float(lam_swd) != 0.0:
            L_swd = masked_swd_lab(pred_lab, tgt_lab, dist_mask, max_samples=int(swd_max_samples), n_proj=8)
    if color_loss_mode in ("hist", "histogram", "current_hist", "current+hist", "hybrid") and float(lam_hist) != 0.0:
        L_hist = soft_histogram_loss_lab(pred_lab, tgt_lab, dist_mask, bins=hist_bins, sigma=hist_sigma)
    if float(lam_neutral) != 0.0:
        L_neutral = neutral_ratio_loss(pred, tgt, pixel_mask)
    Yp = (srgb_to_linear(pred) * wp).sum(1, keepdim=True)
    Ys = (src_lin * wp).sum(1, keepdim=True)
    L_struct = masked_l1(sobel_mag(gaussian_blur(Yp, 5, 1.0)), sobel_mag(gaussian_blur(Ys, 5, 1.0)), dist_mask)
    ltm_param_reg = pred.new_tensor(0.0)
    if hasattr(isp, "ltm_maps"):
        ltm_param_reg = 0.1 * isp.ltm_maps.pow(2).mean()
    reg = isp.du_s.pow(2).mean() + isp.du_wb.pow(2).mean() + isp.du_ccm.pow(2).mean() + isp.du_gtm.pow(2).mean() + isp.du_sat.pow(2).mean() + ltm_param_reg + isp.ltm_regularization()
    L_ltm_ev = pred.new_tensor(0.0)
    L_ltm_ev_tv = pred.new_tensor(0.0)
    if "ltm_local_ev" in dbg:
        local_ev = dbg["ltm_local_ev"]
        L_ltm_ev = local_ev.square().mean()
        L_ltm_ev_tv = (
            (local_ev[:, :, 1:, :] - local_ev[:, :, :-1, :]).abs().mean()
            + (local_ev[:, :, :, 1:] - local_ev[:, :, :, :-1]).abs().mean()
        )
    lut_reg = pred.new_tensor(0.0)
    if isp.lut is not None:
        if bool(getattr(isp.lut, "chroma_only", False)):
            delta_scale = float(getattr(isp.lut, "delta_scale", 1.0))
            lut_delta = torch.tanh(isp.lut.lut) * delta_scale
            lut_reg = lam_lut_smooth * lut_smooth_loss(lut_delta)
        else:
            lut_reg = lam_lut_smooth * lut_smooth_loss(isp.lut.lut) + lam_lut_mono * lut_monotonic_loss(isp.lut.lut)
    L_range = isp_range_loss(dbg)
    L_unc = pred.new_tensor(0.0)
    if color_loss_mode == "current":
        L_color = lam_low * L_low + lam_patch * L_patch + lam_swd * L_swd
    elif color_loss_mode in ("hist", "histogram"):
        L_color = lam_hist * L_hist
    elif color_loss_mode in ("current_hist", "current+hist", "hybrid"):
        L_color = lam_low * L_low + lam_patch * L_patch + lam_swd * L_swd + lam_hist * L_hist
    else:
        raise ValueError(f"Unsupported color_loss_mode: {color_loss_mode}")
    loss_ltm_dynamic = float(ltm_ev_weight) * L_ltm_ev + float(ltm_ev_tv_weight) * L_ltm_ev_tv
    loss = L_color + float(lam_neutral) * L_neutral + lam_struct * L_struct + lam_reg * reg + lam_range * L_range + lut_reg + lam_uncertainty_reg * L_unc + loss_ltm_dynamic
    stats = {"loss": float(loss.detach()), "color": float(L_color.detach()), "low": float(L_low.detach()), "patch": float(L_patch.detach()), "swd": float(L_swd.detach()), "swd_max_samples": int(swd_max_samples), "hist": float(L_hist.detach()), "neutral": float(L_neutral.detach()), "struct": float(L_struct.detach()), "reg": float(reg.detach()), "range": float(L_range.detach()), "lut_reg": float(lut_reg.detach()), "ltm_ev": float(L_ltm_ev.detach()), "ltm_ev_tv": float(L_ltm_ev_tv.detach()), "ltm_dynamic": float(loss_ltm_dynamic.detach()), "mask_mean": float(pixel_mask.detach().mean()), "dist_mask_mean": float(dist_mask.detach().mean()), "supervision_mask_mean": supervision_mask_mean, "raw_valid_mean": float(raw_valid_mean), "raw_conf_mean": float(dbg.get("raw_confidence", torch.ones_like(pixel_mask)).detach().mean()), "loss_output_key": str(loss_output_key)}
    return loss, stats, pred_srgb
