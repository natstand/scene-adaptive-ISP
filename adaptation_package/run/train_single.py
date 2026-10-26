#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "run"

from .alignment import build_roma_model, refresh_alignment
from .common import (
    _json_safe,
    add_bool_arg,
    demosaic_malvar2004,
    load_hs_raw,
    load_mask,
    load_rgb,
    save_mask,
    save_rgb,
)
from .defaults import DEFAULT_CONFIG
from .full4_denoiser import build_full4_denoiser_from_checkpoint
from .isp import InitISPNet, init_exposure_wb_base, init_from_unaligned_phone, optimize_init_net
from .isp_scene_adaptive_ltm import ResidualISPV4SceneAdaptiveLTM as ResidualISPV4
from .losses import build_loss_static_cache, total_loss_residual_v4
from .metadata import read_cihx_metadata
from .metrics import _add_prefixed_metrics, compute_color_metrics, compute_hue_error_metrics, save_color_histogram
from .stages import (
    build_optimizer_for_isp_stage,
    capture_denoiser_trainable_mask,
    isp_parameter_snapshot,
    parse_isp_stage_schedule,
    save_stage_snapshot,
    write_stage_summary,
)


def build_parser() -> argparse.ArgumentParser:
    cfg = DEFAULT_CONFIG
    p = argparse.ArgumentParser(description="Frozen PFV4 RoMa-only single-frame ISP training")
    for key in (
        "hs_raw",
        "raw_dtype",
        "phone_rgb",
        "device",
        "out_dir",
        "out",
        "pattern",
        "align_space",
        "alignment_mode",
        "roma_variant",
        "color_loss_mode",
        "lr_schedule",
        "denoiser_trainable",
        "uncertainty_confidence_mode",
        "uncertainty_loss_routing",
        "stage_loss_output_key",
        "scene_adaptive_ltm_init",
    ):
        default = "" if key not in cfg else str(cfg[key])
        p.add_argument(f"--{key}", type=str, default=default)
    p.add_argument("--denoiser_ckpt", type=str, default=str(cfg["denoiser_ckpt"]))
    p.add_argument(
        "--init_isp_ckpt",
        type=str,
        default="",
        help="Optional ISP checkpoint used to initialize scene adaptation.",
    )
    p.add_argument("--cihx", type=str, default="")
    p.add_argument("--fixed_aligned_phone", type=str, default="")
    p.add_argument("--fixed_valid_mask", type=str, default="")
    p.add_argument("--save_isp_ckpt", type=str, default="")

    for key in (
        "raw_h",
        "raw_w",
        "steps",
        "param_trace_interval",
        "train_forward_size",
        "down",
        "patch",
        "swd_max_samples",
        "init_net_steps",
        "init_net_down",
        "align_max_side",
        "hist_bins",
        "seed",
        "roma_coarse_res",
        "roma_upsample_res",
        "structure_mask_kernel",
    ):
        if key in ("raw_h", "raw_w"):
            default = 0
        else:
            default = int(cfg[key])
        p.add_argument(f"--{key}", type=int, default=default)

    for key in (
        "black_level",
        "white_level",
        "lr",
        "lr_start_mult",
        "lr_peak_mult",
        "lr_final_mult",
        "lr_warmup_frac",
        "exposure_t",
        "init_net_lr",
        "roma_certainty_threshold",
        "roma_certainty_power",
        "lam_low",
        "lam_patch",
        "lam_swd",
        "lam_hist",
        "hist_sigma",
        "lam_struct",
        "lam_reg",
        "lam_range",
        "lam_lut_smooth",
        "lam_lut_mono",
        "raw_dark_threshold",
        "raw_saturation_threshold",
        "texture_percentile",
        "phone_saturation_threshold",
        "lam_neutral",
        "denoiser_lr",
        "denoiser_condition_min_us",
        "denoiser_condition_max_us",
        "uncertainty_temperature",
        "uncertainty_mask_weight",
        "uncertainty_mask_power",
        "lam_uncertainty_reg",
        "ltm_reliability_power",
        "ltm_structure_power",
        "ltm_highlight_guard",
        "ltm_min_gate",
        "ltm_black_guard_min",
        "ltm_black_threshold",
        "ltm_black_softness",
        "ltm_raw_black_threshold",
        "ltm_raw_black_softness",
        "ltm_fixed_blend_strength",
        "structure_mask_sigma",
        "structure_mask_ncc_threshold",
        "structure_mask_temperature",
        "structure_mask_edge_percentile",
        "structure_mask_min_weight",
        "init_wb_min",
        "init_wb_max",
        "ltm_ev_weight",
        "ltm_ev_tv_weight",
        "lut_near_black_low",
        "lut_near_black_high",
    ):
        default = 0.0 if key == "exposure_t" else float(cfg[key])
        p.add_argument(f"--{key}", type=float, default=default)

    p.add_argument("--isp_stage_schedule", type=str, default=str(cfg["schedule"]))
    p.add_argument("--init_wb_mode", type=str, default=str(cfg["init_wb_mode"]), choices=("ratio", "phone"))
    p.add_argument("--init_wb_stat", type=str, default=str(cfg["init_wb_stat"]), choices=("median", "mean"))
    p.add_argument("--raft_ckpt", type=str, default="")
    p.add_argument("--flow_iters", type=int, default=32)
    p.add_argument("--flow_fb_alpha", type=float, default=0.3)
    p.add_argument("--flow_fb_beta", type=float, default=0.8)

    for key in (
        "little_endian",
        "debug",
        "save_intermediate_images",
        "disable_local_flow",
        "use_raw_denoiser",
        "cache_frozen_denoiser",
        "cache_static_loss",
        "train_forward_downsample",
        "use_denoiser_clean_prior",
        "use_denoiser_uncertainty",
        "ltm_noise_aware",
        "disable_ltm_dark_boost",
        "disable_scene_adaptive_ltm",
        "disable_ltm_highlight_compression",
        "ltm_fixed_thresholds",
        "ltm_use_rgb_black_guard",
        "ltm_use_raw_black_guard",
        "gtm_luminance_only",
        "ltm_luminance_only",
        "tone_hue_preserving",
        "lut_black_anchor",
        "lut_near_black_fade",
        "save_tone_stages",
        "save_stage_snapshots",
        "enable_structure_consistency_mask",
        "structure_mask_apply_to_init",
    ):
        add_bool_arg(p, key, bool(cfg[key]))
    p.add_argument(
        "--freeze_scene_adaptive_ltm", action="store_true",
        help="Keep all five Scene-adaptive LTM scene parameters fixed during every optimization stage.",
    )
    return p


def _find_cihx(raw_path: Path, cihx_arg: str = "") -> Path | None:
    if str(cihx_arg).strip():
        path = Path(cihx_arg)
        return path if path.is_absolute() else Path.cwd() / path
    files = sorted(raw_path.parent.glob("*.cihx")) + sorted(raw_path.parent.glob("*.CIHX"))
    return files[0] if files else None


def _apply_cihx_metadata(args) -> dict:
    path = _find_cihx(Path(args.hs_raw), args.cihx)
    if path is None:
        if float(args.exposure_t) <= 0:
            args.exposure_t = 100.0
        return {"enabled": False}
    meta = read_cihx_metadata(path)
    if int(args.raw_h) <= 0:
        args.raw_h = int(meta["height"])
    if int(args.raw_w) <= 0:
        args.raw_w = int(meta["width"])
    if float(args.exposure_t) <= 0:
        args.exposure_t = float(meta["exposure_us"])
    return {
        "enabled": True,
        "path": str(path),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "exposure_us": float(meta["exposure_us"]),
        "frame_rate": meta.get("frame_rate"),
    }


def _trace_vectors(isp: ResidualISPV4):
    with torch.no_grad():
        a, b, c = isp.gtm_params(isp.du_gtm.dtype, isp.du_gtm.device)
        ltm_param = isp.du_ltm if hasattr(isp, "du_ltm") else isp.ltm_maps
        vectors = {
            "exposure": isp.exposure().detach().float().flatten().cpu(),
            "wb": isp.wb_gain().detach().float().flatten().cpu(),
            "ccm": isp.ccm().detach().float().flatten().cpu(),
            "gtm": torch.stack([a, b, c]).detach().float().flatten().cpu(),
            "saturation": isp.saturation().detach().float().flatten().cpu(),
            "ltm": ltm_param.detach().float().flatten().cpu(),
        }
        if isp.lut is not None:
            vectors["lut"] = isp.lut.lut.detach().float().flatten().cpu()
    return vectors


@torch.no_grad()
def _apply_scene_adaptive_ltm_init(isp: ResidualISPV4, spec: str):
    text = str(spec or "").strip()
    if not text:
        return
    if not hasattr(isp, "du_ltm"):
        raise ValueError("--scene_adaptive_ltm_init requires a Scene-adaptive LTM ISP with du_ltm.")
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    expected = int(isp.du_ltm.numel())
    if len(vals) == 6 and expected == 5:
        vals = vals[:5]
    if len(vals) != expected:
        raise ValueError(f"--scene_adaptive_ltm_init needs {expected} comma-separated values, got {len(vals)}")
    init = torch.tensor(vals, device=isp.du_ltm.device, dtype=isp.du_ltm.dtype).reshape_as(isp.du_ltm)
    isp.du_ltm.copy_(init)
    print("[scene-adaptive-ltm-init]", vals, flush=True)


def _rms_delta(curr: torch.Tensor, prev: torch.Tensor | None):
    if prev is None:
        return None
    return float((curr - prev).pow(2).mean().sqrt())


def _build_param_trace_row(step: int, stage: str, loss, stats: dict, isp: ResidualISPV4, prev_vectors: dict | None):
    snap = isp_parameter_snapshot(isp)
    vectors = _trace_vectors(isp)
    row = {
        "step": int(step),
        "stage": str(stage),
        "loss": "" if loss is None else float(loss.detach()),
        "color": stats.get("color", ""),
        "low": stats.get("low", ""),
        "patch": stats.get("patch", ""),
        "swd": stats.get("swd", ""),
        "struct": stats.get("struct", ""),
        "reg": stats.get("reg", ""),
        "range": stats.get("range", ""),
        "mask_mean": stats.get("mask_mean", ""),
        "exposure": snap["exposure"],
        "wb_r": snap["wb_gain_rgb"][0],
        "wb_g": snap["wb_gain_rgb"][1],
        "wb_b": snap["wb_gain_rgb"][2],
        "gtm_a": snap["gtm_abc"][0],
        "gtm_b": snap["gtm_abc"][1],
        "gtm_c": snap["gtm_abc"][2],
        "saturation": snap["saturation"],
        "du_ccm_abs_mean": snap["du_ccm_abs_mean"],
        "du_ccm_abs_max": snap["du_ccm_abs_max"],
        "ltm_maps_abs_mean": snap["ltm_maps_abs_mean"],
        "ltm_maps_abs_max": snap["ltm_maps_abs_max"],
        "lut_delta_abs_mean": snap.get("lut_delta_abs_mean", ""),
        "lut_delta_abs_max": snap.get("lut_delta_abs_max", ""),
    }
    total_sq = []
    for key, curr in vectors.items():
        prev = None if prev_vectors is None else prev_vectors.get(key)
        delta = _rms_delta(curr, prev)
        row[f"delta_{key}_rms"] = "" if delta is None else delta
        if prev is not None:
            total_sq.append((curr - prev).pow(2).flatten())
    row["delta_total_rms"] = "" if not total_sq else float(torch.cat(total_sq).pow(2).mean().sqrt())
    return row, vectors


def _write_param_trace_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _lr_schedule_multiplier(args, step: int, total_steps: int) -> float:
    mode = str(args.lr_schedule).strip().lower()
    if mode in ("", "none", "off", "constant"):
        return 1.0
    if mode not in ("warmup_peak_decay", "warmup_cosine", "peak_decay"):
        raise ValueError(f"Unsupported lr_schedule={args.lr_schedule!r}")
    total = max(1, int(total_steps) - 1)
    progress = float(step) / float(total)
    warmup_frac = float(np.clip(args.lr_warmup_frac, 0.0, 0.95))
    start = float(args.lr_start_mult)
    peak = float(args.lr_peak_mult)
    final = float(args.lr_final_mult)
    if warmup_frac > 0.0 and progress < warmup_frac:
        return start + (peak - start) * (progress / max(warmup_frac, 1e-8))
    tail = (progress - warmup_frac) / max(1.0 - warmup_frac, 1e-8)
    cosine = 0.5 * (1.0 + math.cos(math.pi * float(np.clip(tail, 0.0, 1.0))))
    return final + (peak - final) * cosine


def _apply_lr_schedule(opt: torch.optim.Optimizer, args, step: int, total_steps: int) -> float:
    mult = _lr_schedule_multiplier(args, step, total_steps)
    for group in opt.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        group["lr"] = base_lr * mult
    return mult


def _resize_cached_tensor(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(x) or x.ndim < 4:
        return x
    if tuple(x.shape[-2:]) == tuple(size):
        return x.detach()
    return F.interpolate(x.detach(), size=size, mode="area").detach()


def _downsample_frontend_cache(cache: dict, size: tuple[int, int]) -> dict:
    h, w = int(size[0]), int(size[1])
    packed_size = (max(1, h // 2), max(1, w // 2))
    out = {}
    for key, value in cache.items():
        if key == "den" and isinstance(value, dict):
            out[key] = {
                den_key: _resize_cached_tensor(den_value, packed_size)
                for den_key, den_value in value.items()
            }
        elif key in ("raw_denoised_full", "raw_variance_full", "raw_confidence", "linear_rgb"):
            out[key] = _resize_cached_tensor(value, (h, w))
        else:
            out[key] = value.detach() if torch.is_tensor(value) else value
    return out


def load_raw_denoiser(args, device: torch.device):
    ckpt = str(args.denoiser_ckpt).strip()
    if not ckpt:
        print("[denoiser] disabled")
        return None, {}
    raw_denoiser, den_meta = build_full4_denoiser_from_checkpoint(
        ckpt,
        device=str(device),
        trainable=args.denoiser_trainable,
    )
    print(f"[denoiser] loaded {ckpt}")
    if str(args.pattern).upper() != "GBRG":
        raise ValueError("The reviewer package Full4 denoiser expects the GBRG Bayer pattern.")
    return raw_denoiser, den_meta


def train_single(args) -> int:
    timing_info = {}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda_ok = torch.cuda.is_available()
    device = torch.device(args.device if cuda_ok or not str(args.device).startswith("cuda") else "cpu")
    print(
        f"[device] requested={args.device} using={device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cihx_info = _apply_cihx_metadata(args)
    if cihx_info.get("enabled"):
        print("[cihx]", cihx_info, flush=True)

    raw_denoiser, den_meta = load_raw_denoiser(args, device)
    roma_model = None
    if str(args.alignment_mode).lower() == "roma":
        print(f"[roma] variant={args.roma_variant} coarse={args.roma_coarse_res} upsample={args.roma_upsample_res}")
        roma_model = build_roma_model(
            device=device,
            variant=args.roma_variant,
            coarse_res=args.roma_coarse_res,
            upsample_res=args.roma_upsample_res,
        )
    if not args.disable_local_flow:
        raise ValueError("run is frozen for the RoMa-only experiment; keep --disable_local_flow enabled.")

    raw = load_hs_raw(
        args.hs_raw,
        args.raw_h,
        args.raw_w,
        raw_dtype=args.raw_dtype,
        little_endian=args.little_endian,
        black_level=args.black_level,
        white_level=args.white_level,
        use_12in16=True,
    ).to(device)
    phone = load_rgb(args.phone_rgb).to(device)
    exposure_t = torch.tensor(args.exposure_t, dtype=torch.float32, device=device)

    den_exp_min_log2 = math.log2(args.denoiser_condition_min_us + 1e-12)
    den_exp_max_log2 = math.log2(args.denoiser_condition_max_us + 1e-12)
    gtm_luminance_only = bool(args.gtm_luminance_only or args.tone_hue_preserving)
    ltm_luminance_only = bool(args.ltm_luminance_only or args.tone_hue_preserving)
    isp = ResidualISPV4(
        pattern=args.pattern,
        raw_denoiser=raw_denoiser,
        denoiser_exp_min_log2=den_exp_min_log2,
        denoiser_exp_max_log2=den_exp_max_log2,
        use_raw_denoiser=args.use_raw_denoiser,
        use_denoiser_clean_prior=args.use_denoiser_clean_prior,
        use_denoiser_uncertainty=args.use_denoiser_uncertainty,
        uncertainty_confidence_mode=args.uncertainty_confidence_mode,
        uncertainty_temperature=args.uncertainty_temperature,
        ltm_noise_aware=args.ltm_noise_aware,
        ltm_reliability_power=args.ltm_reliability_power,
        ltm_structure_power=args.ltm_structure_power,
        ltm_highlight_guard=args.ltm_highlight_guard,
        ltm_min_gate=args.ltm_min_gate,
        disable_ltm_dark_boost=args.disable_ltm_dark_boost,
        ltm_black_guard_min=args.ltm_black_guard_min,
        ltm_black_threshold=args.ltm_black_threshold,
        ltm_black_softness=args.ltm_black_softness,
        ltm_use_rgb_black_guard=args.ltm_use_rgb_black_guard,
        ltm_use_raw_black_guard=args.ltm_use_raw_black_guard,
        ltm_raw_black_threshold=args.ltm_raw_black_threshold,
        ltm_raw_black_softness=args.ltm_raw_black_softness,
        lut_black_anchor=args.lut_black_anchor,
        lut_near_black_fade=args.lut_near_black_fade,
        lut_near_black_low=args.lut_near_black_low,
        lut_near_black_high=args.lut_near_black_high,
        disable_scene_adaptive_ltm=args.disable_scene_adaptive_ltm,
        disable_ltm_highlight_compression=args.disable_ltm_highlight_compression,
        ltm_fixed_thresholds=args.ltm_fixed_thresholds,
        ltm_fixed_blend_strength=args.ltm_fixed_blend_strength,
        gtm_luminance_only=gtm_luminance_only,
        ltm_luminance_only=ltm_luminance_only,
    ).to(device)
    _apply_scene_adaptive_ltm_init(isp, args.scene_adaptive_ltm_init)

    if str(args.init_isp_ckpt).strip():
        init_ckpt_path = Path(args.init_isp_ckpt).expanduser()
        if not init_ckpt_path.is_absolute():
            init_ckpt_path = Path.cwd() / init_ckpt_path
        init_ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        init_state = init_ckpt.get("isp", init_ckpt)
        incompatible = isp.load_state_dict(init_state, strict=False)
        isp = isp.to(device)
        print(
            f"[isp-init] loaded {init_ckpt_path} "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
            flush=True,
        )

    denoiser_cache_info = {"enabled": False}
    if bool(args.cache_frozen_denoiser) and str(args.denoiser_trainable).lower() in ("none", "false", "0", "off") and isp.use_raw_denoiser:
        denoiser_cache_info = isp.cache_raw_frontend(raw, exposure_t)
        print("[denoiser-cache]", denoiser_cache_info)

    with torch.no_grad():
        frontend_cache = isp._cached_frontend() if hasattr(isp, "_cached_frontend") else None
        if frontend_cache is not None:
            src_lin_full = frontend_cache["linear_rgb"].clamp(0, 1)
        else:
            raw_for_init = isp.denoise_raw(raw, exposure_t) if isp.use_raw_denoiser else raw
            src_lin_full = demosaic_malvar2004(raw_for_init, pattern=args.pattern).clamp(0, 1)
    save_intermediate = bool(args.save_intermediate_images or args.debug)
    if save_intermediate:
        save_rgb(str(out_dir / "raw_demosaic.png"), src_lin_full)

    fixed_phone_path = args.fixed_aligned_phone.strip()
    fixed_mask_path = args.fixed_valid_mask.strip()
    if bool(fixed_phone_path) != bool(fixed_mask_path):
        raise ValueError("--fixed_aligned_phone and --fixed_valid_mask must be provided together")
    if fixed_phone_path:
        aligned_phone = load_rgb(fixed_phone_path).to(device)
        valid_mask = load_mask(fixed_mask_path).to(device)
        if aligned_phone.shape[-2:] != src_lin_full.shape[-2:]:
            aligned_phone = F.interpolate(aligned_phone, size=src_lin_full.shape[-2:], mode="bilinear", align_corners=False)
        if valid_mask.shape[-2:] != src_lin_full.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, size=src_lin_full.shape[-2:], mode="bilinear", align_corners=False)
        align_info = {
            "mode": "fixed_aligned_phone",
            "fixed_aligned_phone": fixed_phone_path,
            "fixed_valid_mask": fixed_mask_path,
            "valid_ratio": float(valid_mask.detach().mean().item()),
        }
    else:
        init_net = InitISPNet().to(device)
        init_from_unaligned_phone(
            init_net,
            src_lin_full,
            phone,
            wb_mode=args.init_wb_mode,
            wb_stat=args.init_wb_stat,
            wb_min=args.init_wb_min,
            wb_max=args.init_wb_max,
        )
        optimize_init_net(init_net, src_lin_full, phone, steps=args.init_net_steps, lr=args.init_net_lr, down=args.init_net_down)
        with torch.no_grad():
            init_reference = init_net(src_lin_full).clamp(0, 1)
        if save_intermediate:
            save_rgb(str(out_dir / "init_net_reference.png"), init_reference)
        aligned_phone, valid_mask, align_info = refresh_alignment(
            reference_t=init_reference,
            phone_t=phone,
            optic=None,
            out_dir=out_dir,
            device=device,
            align_max_side=args.align_max_side,
            do_local_flow=False,
            fb_alpha=args.flow_fb_alpha,
            fb_beta=args.flow_fb_beta,
            align_space=args.align_space,
            flow_iters=args.flow_iters,
            debug=args.debug,
            alignment_mode=args.alignment_mode,
            roma_model=roma_model,
            roma_certainty_threshold=args.roma_certainty_threshold,
            roma_certainty_power=args.roma_certainty_power,
        )
    print("[align]", align_info)
    save_rgb(str(out_dir / "phone_aligned.png"), aligned_phone)
    save_mask(str(out_dir / "valid_mask.png"), valid_mask)
    with (out_dir / "alignment_info.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(align_info), f, indent=2, ensure_ascii=False)

    base_supervision_mask_full = None
    structure_consistency_info = {"enabled": False}

    with torch.no_grad():
        src_lin_small = F.interpolate(src_lin_full, size=(args.down, args.down), mode="area")
        tgt_small = F.interpolate(aligned_phone, size=(args.down, args.down), mode="area")
        raw_init_valid = (
            F.interpolate(
                ((raw.detach() > float(args.raw_dark_threshold)) & (raw.detach() < float(args.raw_saturation_threshold))).float(),
                size=(args.down, args.down),
                mode="area",
            )
            > 0.995
        ).float()
        mask_small = (F.interpolate(valid_mask, size=(args.down, args.down), mode="area").clamp(0, 1) * raw_init_valid).clamp(0, 1)
        init_exposure_wb_base(
            isp,
            src_lin_small,
            tgt_small,
            mask_small,
            wb_mode=args.init_wb_mode,
            wb_stat=args.init_wb_stat,
            wb_min=args.init_wb_min,
            wb_max=args.init_wb_max,
        )
        print("[init] exposure =", float(isp.exposure().detach().item()))
        print("[init] log2_s0 =", float(isp.log2_s0.detach().item()))
        print("[init] wb0_rb =", isp.wb0_rb.detach().cpu().numpy().tolist())

    stage_schedule = parse_isp_stage_schedule(args.isp_stage_schedule, args.steps)
    total_steps = int(sum(steps for _, steps in stage_schedule))
    print("[stage] schedule =", ", ".join(f"{name}:{steps}" for name, steps in stage_schedule))
    with (out_dir / "stage_schedule.json").open("w", encoding="utf-8") as f:
        json.dump({"schedule": stage_schedule, "total_steps": total_steps, "loss_output_key": args.stage_loss_output_key}, f, indent=2)

    loss_static_cache = None
    loss_static_cache_info = {"enabled": False}
    if bool(args.cache_static_loss):
        loss_static_cache = build_loss_static_cache(
            raw=raw,
            tgt_srgb_full=aligned_phone,
            valid_mask_full=valid_mask,
            src_lin_full=src_lin_full,
            down=args.down,
            raw_dark_threshold=args.raw_dark_threshold,
            raw_saturation_threshold=args.raw_saturation_threshold,
            texture_percentile=args.texture_percentile,
            phone_saturation_threshold=args.phone_saturation_threshold,
            base_supervision_mask_full=base_supervision_mask_full,
        )
        loss_static_cache_info = {
            "enabled": True,
            "down": int(args.down),
            "base_mask_mean": float(loss_static_cache["base_mask"].detach().mean()),
            "raw_valid_mean": float(loss_static_cache["raw_valid_mean"]),
        }
        print("[loss-cache]", loss_static_cache_info)

    full_frontend_cache = isp._cached_frontend() if hasattr(isp, "_cached_frontend") else None
    full_frontend_cache_info = dict(getattr(isp, "_raw_frontend_cache_info", {}))
    train_forward_cache_info = {"enabled": False}
    if bool(args.train_forward_downsample):
        if full_frontend_cache is None:
            raise ValueError("--train_forward_downsample requires a frozen raw frontend cache; keep --cache_frozen_denoiser enabled.")
        train_side = int(args.train_forward_size) if int(args.train_forward_size) > 0 else int(args.down)
        train_size = (train_side, train_side)
        train_cache = _downsample_frontend_cache(full_frontend_cache, train_size)
        isp._raw_frontend_cache = train_cache
        isp._raw_frontend_cache_info = {
            **full_frontend_cache_info,
            "train_forward_downsample": True,
            "train_forward_shape": list(train_size),
            "full_forward_shape": list(src_lin_full.shape[-2:]),
        }
        train_forward_cache_info = {
            "enabled": True,
            "train_forward_shape": list(train_size),
            "full_forward_shape": list(src_lin_full.shape[-2:]),
            "linear_rgb_mean": float(train_cache["linear_rgb"].detach().mean().cpu()),
        }
        print("[train-forward-cache]", train_forward_cache_info)

    denoiser_trainable_mask = capture_denoiser_trainable_mask(isp)
    stage_rows = []
    opt, active_parts, group_desc = build_optimizer_for_isp_stage(isp, args, stage_schedule[0][0], denoiser_trainable_mask)
    stage_idx = 0
    stage_end = stage_schedule[0][1]
    print(f"[stage:start] idx=0 name={stage_schedule[0][0]} active={sorted(active_parts)} groups={group_desc}")
    lr_schedule_info = {
        "mode": str(args.lr_schedule),
        "base_lr": float(args.lr),
        "start_mult": float(args.lr_start_mult),
        "peak_mult": float(args.lr_peak_mult),
        "final_mult": float(args.lr_final_mult),
        "warmup_frac": float(args.lr_warmup_frac),
        "enabled": str(args.lr_schedule).strip().lower() not in ("", "none", "off", "constant"),
    }
    if lr_schedule_info["enabled"]:
        print("[lr-schedule]", lr_schedule_info)

    param_trace_interval = max(0, int(args.param_trace_interval))
    param_trace_rows = []
    param_trace_prev = None
    if param_trace_interval > 0:
        row, param_trace_prev = _build_param_trace_row(0, "init", None, {}, isp, None)
        param_trace_rows.append(row)
        print(f"[param-trace] enabled interval={param_trace_interval} path={out_dir / 'param_trace.csv'}")

    pred = None
    last_lr_mult = _apply_lr_schedule(opt, args, 0, total_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    optimize_start = time.perf_counter()
    for it in range(total_steps):
        stage_name, _ = stage_schedule[stage_idx]
        last_lr_mult = _apply_lr_schedule(opt, args, it, total_steps)
        opt.zero_grad(set_to_none=True)
        loss, stats, pred = total_loss_residual_v4(
            raw=raw,
            exp_t=exposure_t,
            tgt_srgb_full=aligned_phone,
            valid_mask_full=valid_mask,
            isp=isp,
            src_lin_full=src_lin_full,
            down=args.down,
            patch=args.patch,
            lam_low=args.lam_low,
            lam_patch=args.lam_patch,
            lam_swd=args.lam_swd,
            swd_max_samples=args.swd_max_samples,
            color_loss_mode=args.color_loss_mode,
            lam_hist=args.lam_hist,
            hist_bins=args.hist_bins,
            hist_sigma=args.hist_sigma,
            lam_struct=args.lam_struct,
            lam_reg=args.lam_reg,
            lam_range=args.lam_range,
            lam_lut_smooth=args.lam_lut_smooth,
            lam_lut_mono=args.lam_lut_mono,
            raw_dark_threshold=args.raw_dark_threshold,
            raw_saturation_threshold=args.raw_saturation_threshold,
            texture_percentile=args.texture_percentile,
            phone_saturation_threshold=args.phone_saturation_threshold,
            lam_neutral=args.lam_neutral,
            uncertainty_mask_weight=args.uncertainty_mask_weight,
            uncertainty_mask_power=args.uncertainty_mask_power,
            uncertainty_loss_routing=args.uncertainty_loss_routing,
            lam_uncertainty_reg=args.lam_uncertainty_reg,
            base_supervision_mask_full=base_supervision_mask_full,
            loss_output_key=args.stage_loss_output_key,
            static_cache=loss_static_cache,
            ltm_ev_weight=args.ltm_ev_weight,
            ltm_ev_tv_weight=args.ltm_ev_tv_weight,
        )
        loss.backward()
        opt.step()
        step = it + 1
        if param_trace_interval > 0 and (step % param_trace_interval == 0 or step == total_steps or it == stage_end - 1):
            row, param_trace_prev = _build_param_trace_row(step, stage_name, loss, stats, isp, param_trace_prev)
            param_trace_rows.append(row)
        if it % 100 == 0 or it == total_steps - 1 or it == stage_end - 1:
            print(
                f"[{stage_name} {it:04d}/{total_steps - 1:04d}] loss={float(loss.detach()):.6f} "
                f"color={stats['color']:.4f} neutral={stats.get('neutral', 0.0):.4f} "
                f"mask={stats['mask_mean']:.3f} raw_valid={stats.get('raw_valid_mean', 1.0):.3f}"
            )
        if it == stage_end - 1:
            if bool(args.save_stage_snapshots):
                row = save_stage_snapshot(
                    out_dir,
                    stage_idx,
                    stage_name,
                    it,
                    isp,
                    raw,
                    exposure_t,
                    aligned_phone,
                    valid_mask,
                    save_images=bool(save_intermediate or args.save_tone_stages),
                )
                stage_rows.append(row)
                write_stage_summary(out_dir, stage_rows)
                print(f"[stage:end] idx={stage_idx} name={stage_name} psnr={row['psnr_rgb']:.3f} de00={row['deltaE00_mean']:.3f}")
            else:
                print(f"[stage:end] idx={stage_idx} name={stage_name}")
            if stage_idx + 1 < len(stage_schedule):
                stage_idx += 1
                _, next_steps = stage_schedule[stage_idx]
                stage_end = it + 1 + next_steps
                opt, active_parts, group_desc = build_optimizer_for_isp_stage(isp, args, stage_schedule[stage_idx][0], denoiser_trainable_mask)
                last_lr_mult = _apply_lr_schedule(opt, args, it + 1, total_steps)
                print(f"[stage:start] idx={stage_idx} name={stage_schedule[stage_idx][0]} active={sorted(active_parts)} groups={group_desc}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    optimize_seconds = time.perf_counter() - optimize_start
    timing_info.update(
        {
            "optimize_seconds": optimize_seconds,
            "adaptation_seconds": optimize_seconds,
            "optimize_steps": total_steps,
            "optimize_ms_per_step": optimize_seconds * 1000.0 / max(total_steps, 1),
            "adaptation_config": {
                "total_steps": int(total_steps),
                "schedule": stage_schedule,
                "ltm_controls": int(getattr(isp, "du_ltm", torch.empty(5)).numel()),
                "lut_smooth_weight": float(args.lam_lut_smooth),
                "lut_monotonic_weight": float(args.lam_lut_mono),
                "lut_black_anchor": bool(args.lut_black_anchor),
                "lut_near_black_fade": bool(args.lut_near_black_fade),
            },
            "train_forward_downsample": train_forward_cache_info,
            "lr_schedule": {**lr_schedule_info, "last_mult": float(last_lr_mult)},
        }
    )
    print(
        f"[timing] adaptation_seconds={optimize_seconds:.3f} "
        f"optimize_seconds={optimize_seconds:.3f} "
        f"ms_per_step={timing_info['optimize_ms_per_step']:.3f} "
        f"(excludes final save/metrics)"
    )

    if param_trace_rows:
        _write_param_trace_csv(out_dir / "param_trace.csv", param_trace_rows)
        print(f"[param-trace] rows={len(param_trace_rows)} saved={out_dir / 'param_trace.csv'}")

    if train_forward_cache_info.get("enabled"):
        isp._raw_frontend_cache = full_frontend_cache
        isp._raw_frontend_cache_info = full_frontend_cache_info
        print("[train-forward-cache] restored full-res frontend for final render/metrics")

    if device.type == "cuda":
        torch.cuda.synchronize()
    final_render_start = time.perf_counter()
    with torch.no_grad():
        dbg = isp.forward_debug(raw, exposure_t)
        final = dbg["final"].clamp(0, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing_info["final_fullres_render_seconds"] = time.perf_counter() - final_render_start

    save_metrics_start = time.perf_counter()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    save_rgb(str(out_path), final)
    if out_path.parent != out_dir:
        save_rgb(str(out_dir / "predicted_rgb.png"), final)

    metrics = compute_color_metrics(final, aligned_phone, valid_mask)
    metrics["alignment"] = align_info
    metrics["denoiser_cache"] = denoiser_cache_info
    metrics["loss_static_cache"] = loss_static_cache_info
    metrics["train_forward_downsample"] = train_forward_cache_info
    metrics["param_trace"] = {
        "enabled": bool(param_trace_rows),
        "interval": param_trace_interval,
        "path": str(out_dir / "param_trace.csv") if param_trace_rows else "",
    }
    metrics["structure_consistency_mask"] = structure_consistency_info
    metrics["isp_stage_schedule"] = stage_schedule
    metrics["isp_stage_total_steps"] = total_steps
    metrics["stage_loss_output_key"] = args.stage_loss_output_key
    metrics["swd_max_samples"] = int(args.swd_max_samples)
    metrics["lr_schedule"] = lr_schedule_info
    stage_pairs = {
        "after_ccm_vs_phone_hue_error": (dbg["after_ccm"].clamp(0, 1), aligned_phone),
        "after_gtm_vs_phone_hue_error": (dbg["after_gtm"].clamp(0, 1), aligned_phone),
        "after_ltm_vs_phone_hue_error": (dbg["after_ltm"].clamp(0, 1), aligned_phone),
        "gtm_hue_shift_after_ccm_to_after_gtm": (dbg["after_gtm"].clamp(0, 1), dbg["after_ccm"].clamp(0, 1)),
        "ltm_hue_shift_after_gtm_to_after_ltm": (dbg["after_ltm"].clamp(0, 1), dbg["after_gtm"].clamp(0, 1)),
        "tone_hue_shift_after_ccm_to_after_ltm": (dbg["after_ltm"].clamp(0, 1), dbg["after_ccm"].clamp(0, 1)),
    }
    for prefix, (pred_stage, tgt_stage) in stage_pairs.items():
        _add_prefixed_metrics(metrics, prefix, compute_hue_error_metrics(pred_stage, tgt_stage, valid_mask))
    metrics["timing"] = dict(timing_info)
    with (out_dir / "color_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)
    if save_intermediate:
        save_color_histogram(str(out_dir / "color_histogram_diff.png"), final, aligned_phone, valid_mask)
    if args.save_isp_ckpt.strip():
        torch.save({"isp": isp.state_dict(), "args": vars(args), "denoiser_meta": den_meta}, args.save_isp_ckpt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing_info["save_metrics_checkpoint_seconds"] = time.perf_counter() - save_metrics_start
    with (out_dir / "timing.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(timing_info), f, indent=2, ensure_ascii=False)
    print(
        f"[timing] final_fullres_render_seconds={timing_info['final_fullres_render_seconds']:.3f} "
        f"save_metrics_checkpoint_seconds={timing_info['save_metrics_checkpoint_seconds']:.3f}"
    )
    print(f"[done] saved output: {out_path}")
    return 0


def main() -> int:
    return train_single(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
