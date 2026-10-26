#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import demosaic_malvar2004, load_hs_raw, load_rgb, save_rgb
from .full4_denoiser import build_full4_denoiser_from_checkpoint
from .defaults import DEFAULT_DENOISER_CKPT
from .isp_scene_adaptive_ltm import ResidualISPV4SceneAdaptiveLTM as ResidualISPV4
from .metadata import read_cihx_metadata
from .metrics import compute_extended_metrics, resize_like
from .summary import image_tile, paste_tiles_grid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render selected PFV4 RAW frames with a trained run ISP checkpoint.")
    p.add_argument("--isp_ckpt", required=True, type=str)
    p.add_argument("--raw_dir", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--case", default="render", type=str)
    p.add_argument("--first_n", default=0, type=int, help="Restrict selection to the first N sorted RAW frames; 0 means all.")
    p.add_argument("--count", default=10, type=int)
    p.add_argument("--warmup", default=5, type=int, help="Un-timed ISP forwards before measuring steady-state render speed.")
    p.add_argument("--stride", default=0, type=int, help="Fixed stride in selected frame list. If 0, use linspace.")
    p.add_argument("--start_index", default=0, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--denoiser_ckpt", default="", type=str)
    p.add_argument(
        "--preserve_external_denoiser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load ISP parameters but do not overwrite the explicitly loaded RAW denoiser from the ISP checkpoint.",
    )
    p.add_argument("--cihx", default="", type=str)
    p.add_argument("--raw_h", default=0, type=int)
    p.add_argument("--raw_w", default=0, type=int)
    p.add_argument("--exposure_t", default=0.0, type=float)
    p.add_argument(
        "--disable_scene_adaptive_ltm_override",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint Scene-adaptive LTM switch for an inference-only ablation.",
    )
    p.add_argument(
        "--crop_xywh",
        default="",
        type=str,
        help="Optional even-aligned RAW crop x,y,w,h applied before denoising/ISP.",
    )
    p.add_argument("--frame_numbers_txt", default="", type=str)
    p.add_argument("--frame_numbers_input", default="", type=str)
    p.add_argument("--built_in_dir", default="", type=str)
    p.add_argument("--metrics_max_side", default=512, type=int)
    p.add_argument("--save_raw_demosaic_vis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_denoised_vis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_builtin_vis", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compute_builtin_metrics", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--match_train_raw_brightness", action="store_true")
    p.add_argument("--brightness_reference_raw", default="", type=str)
    p.add_argument("--brightness_match_stat", default="mean", choices=("mean", "p95", "mean_p95"))
    p.add_argument("--brightness_scale_min", default=0.5, type=float)
    p.add_argument("--brightness_scale_max", default=1.5, type=float)
    return p


def _bool_arg(args: dict, key: str, default: bool) -> bool:
    value = args.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return root / p


def _find_cihx(raw_dir: Path, cihx_arg: str, root: Path) -> Path | None:
    if str(cihx_arg).strip():
        return _resolve_path(cihx_arg, root)
    files = sorted(raw_dir.glob("*.cihx")) + sorted(raw_dir.glob("*.CIHX"))
    return files[0] if files else None


def _save_percentile_rgb(path: Path, rgb: torch.Tensor, lo_p: float = 1.0, hi_p: float = 99.5) -> None:
    arr = rgb.detach().float().cpu().clamp_min(0)[0].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(arr, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        lo, hi = float(arr.min()), float(arr.max() + 1e-12)
    vis = np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((vis * 255.0 + 0.5).astype(np.uint8)).save(path)


def _raw_brightness_stat(raw: torch.Tensor, stat: str) -> float:
    x = raw.detach().float().clamp(0, 1)
    mean = float(x.mean().item())
    if stat == "mean":
        return mean
    p95 = float(torch.quantile(x.flatten(), 0.95).item())
    if stat == "p95":
        return p95
    if stat == "mean_p95":
        return math.sqrt(max(mean, 1e-12) * max(p95, 1e-12))
    raise ValueError(f"Unsupported brightness stat: {stat}")


def _path_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def _image_files(path: Path) -> list[Path]:
    return sorted(
        [p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")],
        key=lambda p: (_path_number(p), p.name),
    )


def _load_frame_numbers(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    numbers: list[int] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("selected_frame_numbers"):
            in_block = True
            numbers.extend(int(x) for x in re.findall(r"\d{5,}", stripped))
            continue
        if in_block:
            found = re.findall(r"\d{5,}", stripped)
            if found:
                numbers.extend(int(x) for x in found)
            elif stripped.endswith(":"):
                break
    if not numbers:
        numbers = [int(x) for x in re.findall(r"\d{5,}", text)]
    return numbers


def _select_frames(raw_paths: list[Path], first_n: int, count: int, stride: int, start_index: int) -> tuple[list[Path], list[int], str]:
    if first_n > 0:
        raw_paths = raw_paths[:first_n]
    if not raw_paths:
        raise RuntimeError("No RAW frames found.")
    if count <= 0:
        raise ValueError("--count must be positive.")

    start_index = max(0, int(start_index))
    if stride > 0:
        idx = [start_index + i * int(stride) for i in range(count)]
        idx = [i for i in idx if i < len(raw_paths)]
        rule = f"start_index={start_index}, fixed_stride={stride}, count={count}, first_n={first_n or 'all'}"
    else:
        usable = raw_paths[start_index:]
        if count >= len(usable):
            idx = list(range(start_index, len(raw_paths)))
        else:
            idx = [start_index + int(round(x)) for x in np.linspace(0, len(usable) - 1, count)]
        rule = f"start_index={start_index}, linspace_count={count}, first_n={first_n or 'all'}"
    selected = [raw_paths[i] for i in idx]
    return selected, idx, rule


def _select_frames_from_numbers(raw_paths: list[Path], frame_numbers: list[int]) -> tuple[list[Path], list[int], str]:
    lut = {_path_number(p): (i, p) for i, p in enumerate(raw_paths)}
    selected = []
    indices = []
    missing = []
    for number in frame_numbers:
        item = lut.get(int(number))
        if item is None:
            missing.append(int(number))
            continue
        idx, path = item
        selected.append(path)
        indices.append(idx)
    if not selected:
        raise RuntimeError("None of the requested frame numbers exists in raw_dir.")
    rule = f"explicit_frame_numbers={len(selected)}"
    if missing:
        rule += f", missing={missing}"
    return selected, indices, rule


def _maybe_downsample_for_metrics(x: torch.Tensor, max_side: int) -> torch.Tensor:
    if int(max_side) <= 0:
        return x
    _, _, h, w = x.shape
    side = max(h, w)
    if side <= int(max_side):
        return x
    scale = float(max_side) / float(side)
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _save_horizontal_compare(path: Path, items: list[tuple[str, Path]], width: int = 260) -> None:
    tiles = [image_tile(label, p, width=width) for label, p in items if p.exists()]
    paste_tiles_grid(tiles, path, columns=len(tiles), gap=8, bg=(255, 255, 255))


def render_selected_frames(args: argparse.Namespace) -> dict:
    root = Path.cwd()
    out_dir = _resolve_path(args.out_dir, root)
    frames_dir = out_dir / "frames"
    vis_dir = out_dir / "visualizations"
    frames_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    ckpt_path = _resolve_path(args.isp_ckpt, root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    pattern = str(ckpt_args.get("pattern", "GBRG"))
    raw_dir = _resolve_path(args.raw_dir, root)
    cihx_path = _find_cihx(raw_dir, args.cihx, root)
    cihx_meta = read_cihx_metadata(cihx_path) if cihx_path is not None else None
    raw_h = int(args.raw_h or ckpt_args.get("raw_h", 0))
    raw_w = int(args.raw_w or ckpt_args.get("raw_w", 0))
    if cihx_meta is not None:
        # The input sequence metadata describes the file on disk. This may be
        # larger than the training crop recorded in the ISP checkpoint.
        raw_h = int(args.raw_h or cihx_meta["height"])
        raw_w = int(args.raw_w or cihx_meta["width"])
    if raw_h <= 0 or raw_w <= 0:
        raise ValueError("RAW dimensions are missing. Provide --raw_h and --raw_w.")
    raw_dtype = str(ckpt_args.get("raw_dtype", "uint16"))
    little_endian = _bool_arg(ckpt_args, "little_endian", True)
    black_level = float(ckpt_args.get("black_level", 0.0))
    white_level = float(ckpt_args.get("white_level", 4095.0))
    crop_xywh = None
    if str(args.crop_xywh).strip():
        vals = [int(v.strip()) for v in str(args.crop_xywh).split(",")]
        if len(vals) != 4:
            raise ValueError("--crop_xywh must be x,y,w,h")
        x, y, cw, ch = vals
        if min(x, y, cw, ch) < 0 or cw <= 0 or ch <= 0:
            raise ValueError(f"Invalid --crop_xywh: {vals}")
        if any(v % 2 for v in vals):
            raise ValueError(f"Bayer-preserving --crop_xywh values must all be even: {vals}")
        if x + cw > raw_w or y + ch > raw_h:
            raise ValueError(f"Crop {vals} exceeds RAW size {raw_w}x{raw_h}")
        crop_xywh = (x, y, cw, ch)
    exposure_t_us = float(args.exposure_t or ckpt_args.get("exposure_t", 0.0))
    if exposure_t_us <= 0 and cihx_meta is not None:
        exposure_t_us = float(cihx_meta["exposure_us"])
    if exposure_t_us <= 0:
        exposure_t_us = 100.0

    denoiser_ckpt = args.denoiser_ckpt.strip() or str(ckpt_args.get("denoiser_ckpt", ""))
    denoiser_path = _resolve_path(denoiser_ckpt, root) if denoiser_ckpt else DEFAULT_DENOISER_CKPT
    min_us = float(ckpt_args.get("denoiser_condition_min_us", 23.2))
    max_us = float(ckpt_args.get("denoiser_condition_max_us", 498.2))
    raw_denoiser, _ = build_full4_denoiser_from_checkpoint(
        str(denoiser_path),
        device=str(device),
        trainable="none",
    )
    isp = ResidualISPV4(
        pattern=pattern,
        raw_denoiser=raw_denoiser,
        denoiser_exp_min_log2=math.log2(min_us + 1e-12),
        denoiser_exp_max_log2=math.log2(max_us + 1e-12),
        use_raw_denoiser=_bool_arg(ckpt_args, "use_raw_denoiser", True),
        use_denoiser_clean_prior=_bool_arg(ckpt_args, "use_denoiser_clean_prior", True),
        use_denoiser_uncertainty=_bool_arg(ckpt_args, "use_denoiser_uncertainty", False),
        uncertainty_confidence_mode=str(ckpt_args.get("uncertainty_confidence_mode", "reliability")),
        uncertainty_temperature=float(ckpt_args.get("uncertainty_temperature", 4.0)),
        ltm_noise_aware=_bool_arg(ckpt_args, "ltm_noise_aware", True),
        ltm_reliability_power=float(ckpt_args.get("ltm_reliability_power", 1.0)),
        ltm_structure_power=float(ckpt_args.get("ltm_structure_power", 1.0)),
        ltm_highlight_guard=float(ckpt_args.get("ltm_highlight_guard", 0.5)),
        ltm_min_gate=float(ckpt_args.get("ltm_min_gate", 0.2)),
        disable_ltm_dark_boost=_bool_arg(ckpt_args, "disable_ltm_dark_boost", False),
        ltm_black_guard_min=float(ckpt_args.get("ltm_black_guard_min", 0.05)),
        ltm_black_threshold=float(ckpt_args.get("ltm_black_threshold", 0.03)),
        ltm_black_softness=float(ckpt_args.get("ltm_black_softness", 0.02)),
        ltm_use_rgb_black_guard=_bool_arg(ckpt_args, "ltm_use_rgb_black_guard", False),
        ltm_use_raw_black_guard=_bool_arg(ckpt_args, "ltm_use_raw_black_guard", True),
        ltm_raw_black_threshold=float(ckpt_args.get("ltm_raw_black_threshold", 0.015)),
        ltm_raw_black_softness=float(ckpt_args.get("ltm_raw_black_softness", 0.02)),
        lut_black_anchor=_bool_arg(ckpt_args, "lut_black_anchor", True),
        lut_near_black_fade=_bool_arg(ckpt_args, "lut_near_black_fade", True),
        lut_near_black_low=float(ckpt_args.get("lut_near_black_low", 0.030)),
        lut_near_black_high=float(ckpt_args.get("lut_near_black_high", 0.150)),
        # Accept the pre-rename key when rendering older checkpoints.
        disable_scene_adaptive_ltm=_bool_arg(
            ckpt_args,
            "disable_scene_adaptive_ltm",
            _bool_arg(ckpt_args, "disable_dynamic_ltm", False),
        ),
        disable_ltm_highlight_compression=_bool_arg(ckpt_args, "disable_ltm_highlight_compression", False),
        ltm_fixed_thresholds=_bool_arg(ckpt_args, "ltm_fixed_thresholds", False),
        ltm_fixed_blend_strength=float(ckpt_args.get("ltm_fixed_blend_strength", -1.0)),
        gtm_luminance_only=_bool_arg(ckpt_args, "gtm_luminance_only", False) or _bool_arg(ckpt_args, "tone_hue_preserving", False),
        ltm_luminance_only=_bool_arg(ckpt_args, "ltm_luminance_only", False) or _bool_arg(ckpt_args, "tone_hue_preserving", False),
    ).to(device)
    isp_state = ckpt["isp"]
    if args.preserve_external_denoiser:
        isp_state = {k: v for k, v in isp_state.items() if not k.startswith("raw_denoiser.")}
    incompatible = isp.load_state_dict(isp_state, strict=False)
    # Some checkpoints contain non-parameter LUT buffers. Loading a CPU
    # checkpoint can replace those buffers after the initial .to(device).
    isp = isp.to(device)
    if args.disable_scene_adaptive_ltm_override is not None:
        isp.disable_scene_adaptive_ltm = bool(args.disable_scene_adaptive_ltm_override)
    if args.preserve_external_denoiser:
        unexpected_non_denoiser = [k for k in incompatible.unexpected_keys if not k.startswith("raw_denoiser.")]
        if unexpected_non_denoiser:
            raise RuntimeError(f"Unexpected non-denoiser ISP keys: {unexpected_non_denoiser}")
    isp.eval()

    raw_dir = _resolve_path(args.raw_dir, root)
    raw_paths = sorted(raw_dir.glob("*.raww"))
    if args.frame_numbers_input.strip():
        frame_numbers_input = _resolve_path(args.frame_numbers_input, root)
        selected, selected_indices, selection_rule = _select_frames_from_numbers(raw_paths, _load_frame_numbers(frame_numbers_input))
    else:
        selected, selected_indices, selection_rule = _select_frames(raw_paths, args.first_n, args.count, args.stride, args.start_index)
    exp_t = torch.tensor(exposure_t_us, dtype=torch.float32, device=device)
    built_in_dir = _resolve_path(args.built_in_dir, root) if args.built_in_dir.strip() else None
    built_in_paths = _image_files(built_in_dir) if built_in_dir is not None and built_in_dir.exists() else []
    brightness_info = {"enabled": False}
    brightness_target = None
    if bool(args.match_train_raw_brightness):
        ref_raw_arg = args.brightness_reference_raw.strip() or str(ckpt_args.get("hs_raw", ""))
        if not ref_raw_arg:
            raise ValueError("--match_train_raw_brightness needs checkpoint args hs_raw or --brightness_reference_raw.")
        ref_raw_path = _resolve_path(ref_raw_arg, root)
        ref_raw = load_hs_raw(
            str(ref_raw_path),
            raw_h,
            raw_w,
            raw_dtype=raw_dtype,
            little_endian=little_endian,
            black_level=black_level,
            white_level=white_level,
            use_12in16=True,
        ).to(device)
        brightness_target = _raw_brightness_stat(ref_raw, args.brightness_match_stat)
        brightness_info = {
            "enabled": True,
            "reference_raw": str(ref_raw_path.relative_to(root)) if ref_raw_path.is_relative_to(root) else str(ref_raw_path),
            "stat": args.brightness_match_stat,
            "target": brightness_target,
            "scale_min": float(args.brightness_scale_min),
            "scale_max": float(args.brightness_scale_max),
        }

    rows = []
    isp_render_seconds = 0.0
    warmup_done = False
    warmup_count = max(0, int(args.warmup))
    with torch.inference_mode():
        for row_index, raw_path in enumerate(selected):
            label = raw_path.stem
            frame_dir = frames_dir / label
            frame_dir.mkdir(parents=True, exist_ok=True)
            raw = load_hs_raw(
                str(raw_path),
                raw_h,
                raw_w,
                raw_dtype=raw_dtype,
                little_endian=little_endian,
                black_level=black_level,
                white_level=white_level,
                use_12in16=True,
            ).to(device)
            raw_mean_original = float(raw.mean().item())
            if crop_xywh is not None:
                x, y, cw, ch = crop_xywh
                raw = raw[..., y : y + ch, x : x + cw]
            brightness_scale = 1.0
            raw_brightness_original = _raw_brightness_stat(raw, args.brightness_match_stat)
            if brightness_target is not None:
                brightness_scale = brightness_target / max(raw_brightness_original, 1e-12)
                brightness_scale = float(np.clip(brightness_scale, args.brightness_scale_min, args.brightness_scale_max))
                raw = (raw * brightness_scale).clamp(0, 1)

            # The first CUDA forward includes one-time kernel selection and
            # lazy initialization. Exclude it from the reported steady-state
            # ISP speed, matching the benchmark protocol used elsewhere.
            if not warmup_done:
                for _ in range(warmup_count):
                    if args.save_denoised_vis:
                        _ = isp.forward_debug(raw, exp_t)
                    else:
                        _ = isp(raw, exp_t)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warmup_done = True
            if device.type == "cuda":
                torch.cuda.synchronize()
            isp_start = time.perf_counter()
            if args.save_denoised_vis:
                dbg = isp.forward_debug(raw, exp_t)
                pred = dbg["final"].clamp(0, 1)
            else:
                dbg = {}
                pred = isp(raw, exp_t).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            isp_render_seconds += time.perf_counter() - isp_start

            raw_linear_path = frame_dir / "raw_demosaic_linear_nogamma.png"
            den_linear_path = frame_dir / "denoised_demosaic_linear_nogamma.png"
            pred_path = frame_dir / "predicted_rgb.png"
            raw_pct_path = frame_dir / "raw_demosaic_percentile_nogamma.png"
            den_pct_path = frame_dir / "denoised_demosaic_percentile_nogamma.png"
            save_rgb(str(pred_path), pred)
            raw_linear_rel = ""
            den_linear_rel = ""
            raw_pct_rel = ""
            den_pct_rel = ""
            if args.save_raw_demosaic_vis:
                raw_demo = demosaic_malvar2004(raw, pattern=pattern).clamp(0, 1)
                save_rgb(str(raw_linear_path), raw_demo)
                _save_percentile_rgb(raw_pct_path, raw_demo)
                raw_linear_rel = str(raw_linear_path.relative_to(root))
                raw_pct_rel = str(raw_pct_path.relative_to(root))
            if args.save_denoised_vis:
                den_demo = demosaic_malvar2004(dbg["raw_denoised_full"].clamp(0, 1), pattern=pattern).clamp(0, 1)
                save_rgb(str(den_linear_path), den_demo)
                _save_percentile_rgb(den_pct_path, den_demo)
                den_linear_rel = str(den_linear_path.relative_to(root))
                den_pct_rel = str(den_pct_path.relative_to(root))
            built_in_path = built_in_paths[selected_indices[row_index]] if built_in_paths and selected_indices[row_index] < len(built_in_paths) else None
            built_in_rel = ""
            pred_builtin_compare_rel = ""
            builtin_metrics = None
            if built_in_path is not None:
                ref = resize_like(load_rgb(str(built_in_path)).to(device), pred)
                built_out = frame_dir / "built_in_rgb.png"
                compare_out = frame_dir / "predicted_builtin_compare.png"
                if args.save_builtin_vis:
                    save_rgb(str(built_out), ref)
                    _save_horizontal_compare(compare_out, [("built-in", built_out), ("predicted", pred_path)], width=260)
                    built_in_rel = str(built_out.relative_to(root))
                    pred_builtin_compare_rel = str(compare_out.relative_to(root))
                if args.compute_builtin_metrics:
                    pred_m = _maybe_downsample_for_metrics(pred, args.metrics_max_side)
                    ref_m = _maybe_downsample_for_metrics(ref, args.metrics_max_side)
                    full_mask = torch.ones((1, 1, ref_m.shape[-2], ref_m.shape[-1]), device=ref_m.device, dtype=ref_m.dtype)
                    builtin_metrics = compute_extended_metrics(pred_m, ref_m, full_mask)
            rows.append(
                {
                    "label": label,
                    "raw": str(raw_path.relative_to(root)) if raw_path.is_relative_to(root) else str(raw_path),
                    "raw_demosaic_linear_nogamma": raw_linear_rel,
                    "denoised_demosaic_linear_nogamma": den_linear_rel,
                    "predicted_rgb": str(pred_path.relative_to(root)),
                    "built_in_source": str(built_in_path.relative_to(root)) if built_in_path is not None and built_in_path.is_relative_to(root) else (str(built_in_path) if built_in_path is not None else ""),
                    "built_in_rgb": built_in_rel,
                    "predicted_builtin_compare": pred_builtin_compare_rel,
                    "raw_mean": float(raw.mean()),
                    "raw_mean_original": raw_mean_original,
                    "brightness_stat_original": raw_brightness_original,
                    "brightness_scale": brightness_scale,
                    "pred_mean": float(pred.mean()),
                    "raw_demosaic_percentile_nogamma": raw_pct_rel,
                    "denoised_demosaic_percentile_nogamma": den_pct_rel,
                    "metrics_vs_built_in": builtin_metrics,
                }
            )

    pred_grid = vis_dir / f"{args.case}_predicted_rgb_grid.png"
    raw_grid = vis_dir / f"{args.case}_raw_demosaic_percentile_nogamma_grid.png"
    den_grid = vis_dir / f"{args.case}_denoised_demosaic_percentile_nogamma_grid.png"
    comparison = vis_dir / f"{args.case}_demosaic_predicted_comparison.png"
    built_grid = vis_dir / f"{args.case}_built_in_rgb_grid.png"
    pred_builtin_grid = vis_dir / f"{args.case}_predicted_builtin_comparison_grid.png"
    paste_tiles_grid([image_tile(r["label"], root / r["predicted_rgb"], width=220) for r in rows], pred_grid, columns=5)
    if args.save_raw_demosaic_vis:
        paste_tiles_grid([image_tile(r["label"], root / r["raw_demosaic_percentile_nogamma"], width=220) for r in rows if r["raw_demosaic_percentile_nogamma"]], raw_grid, columns=5)
    if args.save_denoised_vis:
        paste_tiles_grid([image_tile(r["label"], root / r["denoised_demosaic_percentile_nogamma"], width=220) for r in rows if r["denoised_demosaic_percentile_nogamma"]], den_grid, columns=5)
    if args.save_raw_demosaic_vis or args.save_denoised_vis:
        comp_tiles = []
        for r in rows:
            if r["raw_demosaic_percentile_nogamma"]:
                comp_tiles.append(image_tile(r["label"] + " raw", root / r["raw_demosaic_percentile_nogamma"], width=220))
            if r["denoised_demosaic_percentile_nogamma"]:
                comp_tiles.append(image_tile(r["label"] + " denoised", root / r["denoised_demosaic_percentile_nogamma"], width=220))
            comp_tiles.append(image_tile(r["label"] + " predicted", root / r["predicted_rgb"], width=220))
        paste_tiles_grid(comp_tiles, comparison, columns=3, bg=(255, 255, 255))
    if built_in_paths and args.save_builtin_vis:
        paste_tiles_grid([image_tile(r["label"], root / r["built_in_rgb"], width=220) for r in rows if r["built_in_rgb"]], built_grid, columns=5)
        paste_tiles_grid([image_tile(r["label"], root / r["predicted_builtin_compare"], width=560) for r in rows if r["predicted_builtin_compare"]], pred_builtin_grid, columns=2, bg=(255, 255, 255))
    metric_rows = [r["metrics_vs_built_in"] for r in rows if r.get("metrics_vs_built_in")]
    render_metrics = {}
    if metric_rows:
        render_metrics = {k: float(np.mean([m[k] for m in metric_rows if k in m])) for k in metric_rows[0].keys()}

    frame_numbers = [_path_number(p) for p in selected]
    frame_txt = _resolve_path(args.frame_numbers_txt, root) if args.frame_numbers_txt.strip() else out_dir / "frame_numbers.txt"
    frame_txt.parent.mkdir(parents=True, exist_ok=True)
    frame_txt.write_text(
        "\n".join(
            [
                f"case: {args.case}",
                f"raw_dir: {raw_dir.relative_to(root) if raw_dir.is_relative_to(root) else raw_dir}",
                f"isp_ckpt: {ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path}",
                f"selection_rule: {selection_rule}",
                f"selected_indices_zero_based: {selected_indices}",
                "selected_frame_numbers:",
                *[str(n) for n in frame_numbers],
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "case": args.case,
        "isp_ckpt": str(ckpt_path.relative_to(root)) if ckpt_path.is_relative_to(root) else str(ckpt_path),
        "raw_dir": str(raw_dir.relative_to(root)) if raw_dir.is_relative_to(root) else str(raw_dir),
        "total_frames": len(raw_paths),
        "selection_rule": selection_rule,
        "selected_indices_zero_based": selected_indices,
        "selected_frame_numbers": frame_numbers,
        "frame_numbers_txt": str(frame_txt.relative_to(root)),
        "brightness_match": brightness_info,
        "built_in_dir": str(built_in_dir.relative_to(root)) if built_in_dir is not None and built_in_dir.exists() and built_in_dir.is_relative_to(root) else (str(built_in_dir) if built_in_dir is not None else ""),
        "built_in_count": len(built_in_paths),
        "metrics_vs_built_in_mean": render_metrics,
        "exposure_gain": float(isp.exposure().detach().cpu().item()),
        "wb_r": float(isp.wb_gain()[0, 0, 0, 0].detach().cpu().item()),
        "wb_b": float(isp.wb_gain()[0, 2, 0, 0].detach().cpu().item()),
        "exposure_t_us": exposure_t_us,
        "raw": {"height": raw_h, "width": raw_w, "pattern": pattern, "white_level": white_level},
        "crop_xywh": list(crop_xywh) if crop_xywh is not None else None,
        "timing": {
            "isp_render_seconds": float(isp_render_seconds),
            "isp_render_ms_per_frame": float(isp_render_seconds * 1000.0 / max(len(selected), 1)),
            "isp_render_fps": float(len(selected) / max(isp_render_seconds, 1e-12)),
            "image_io_excluded": True,
            "warmup_for_speed_measurement": int(warmup_count),
        },
        "frames": rows,
        "grids": {
            "comparison": str(comparison.relative_to(root)) if comparison.exists() else "",
            "predicted": str(pred_grid.relative_to(root)),
            "raw_percentile_nogamma": str(raw_grid.relative_to(root)) if raw_grid.exists() else "",
            "denoised_percentile_nogamma": str(den_grid.relative_to(root)) if den_grid.exists() else "",
            "built_in": str(built_grid.relative_to(root)) if built_grid.exists() else "",
            "predicted_builtin_comparison": str(pred_builtin_grid.relative_to(root)) if pred_builtin_grid.exists() else "",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "# PFV4 frame render\n\n"
        f"- Case: `{args.case}`\n"
        f"- Total frames in dir: `{len(raw_paths)}`\n"
        f"- Selection: `{selection_rule}`\n"
        f"- Selected frame numbers: `{frame_numbers}`\n"
        f"- Frame number file: `{summary['frame_numbers_txt']}`\n"
        f"- Brightness match: `{brightness_info}`\n"
        f"- Built-in dir: `{summary['built_in_dir']}`\n"
        f"- Built-in metrics: disabled; built-in is visualization only\n"
        f"- ISP exposure gain: `{summary['exposure_gain']:.6f}`\n"
        f"- WB: `R {summary['wb_r']:.6f}, B {summary['wb_b']:.6f}`\n"
        f"- Camera exposure_t: `{exposure_t_us}us`\n"
        f"- RAW: `{raw_w}x{raw_h}`, `{pattern}`, 12-bit\n\n"
        f"- ISP render time (I/O excluded): `{summary['timing']['isp_render_seconds']:.6f}s`\n"
        f"- ISP render FPS: `{summary['timing']['isp_render_fps']:.3f}`\n\n"
        "## Grids\n"
        f"- comparison: `{summary['grids']['comparison']}`\n"
        f"- predicted: `{summary['grids']['predicted']}`\n"
        f"- raw_percentile_nogamma: `{summary['grids']['raw_percentile_nogamma']}`\n"
        f"- denoised_percentile_nogamma: `{summary['grids']['denoised_percentile_nogamma']}`\n"
        f"- built_in: `{summary['grids']['built_in']}`\n"
        f"- predicted_builtin_comparison: `{summary['grids']['predicted_builtin_comparison']}`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    summary = render_selected_frames(build_parser().parse_args())
    print(json.dumps({
        "frames": summary["selected_frame_numbers"],
        "frame_numbers_txt": summary["frame_numbers_txt"],
        "predicted_grid": summary["grids"]["predicted"],
        "built_in_grid": summary["grids"].get("built_in", ""),
        "predicted_builtin_comparison": summary["grids"].get("predicted_builtin_comparison", ""),
        "comparison": summary["grids"]["comparison"],
        "timing": summary["timing"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
