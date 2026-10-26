#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import *
from .common import _json_safe
from .isp import *
from .metrics import compute_color_metrics

# ============================================================
# Staged ISP optimization diagnostics
# ============================================================

def normalize_isp_stage_name(stage: str) -> str:
    name = str(stage).strip().lower().replace("-", "_")
    allowed = {"warmup_exp_wb_ccm", "joint"}
    if name not in allowed:
        raise ValueError(f"Unsupported ISP stage {stage!r}. Allowed: {sorted(allowed)}")
    return name


def parse_isp_stage_schedule(spec: str, default_steps: int):
    text = str(spec or "joint").strip()
    if text.lower() in ("", "joint"):
        return [("joint", int(default_steps))]
    stages = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError("Stage schedule entries must be formatted as name:steps, e.g. warmup_exp_wb_ccm:50,joint:250")
        name, steps_s = token.split(":", 1)
        name = normalize_isp_stage_name(name)
        steps = int(steps_s)
        if steps <= 0:
            raise ValueError(f"Stage {name} has non-positive steps: {steps}")
        stages.append((name, steps))
    if not stages:
        stages = [("joint", int(default_steps))]
    return stages


def isp_stage_active_parts(stage: str):
    stage = normalize_isp_stage_name(stage)
    if stage == "warmup_exp_wb_ccm":
        return {"du_s", "du_wb", "du_ccm"}
    return {"du_s", "du_wb", "du_ccm", "du_gtm", "du_sat", "ltm", "lut"}


def _ltm_params(isp):
    params = []
    if hasattr(isp, "du_ltm"):
        params.append(isp.du_ltm)
    if hasattr(isp, "ltm_maps"):
        params.append(isp.ltm_maps)
    return params


def capture_denoiser_trainable_mask(isp: ResidualISPV4):
    if isp.raw_denoiser is None:
        return {}
    return {id(p): bool(p.requires_grad) for p in isp.raw_denoiser.parameters()}


def set_isp_trainable_stage(isp: ResidualISPV4, stage: str, denoiser_trainable_mask=None, allow_denoiser=False, freeze_scene_adaptive_ltm=False):
    active = isp_stage_active_parts(stage)
    for name in ("du_s", "du_wb", "du_ccm", "du_gtm", "du_sat"):
        getattr(isp, name).requires_grad_(name in active)
    for p in _ltm_params(isp):
        # Used for the fixed-LTM control: the Scene-adaptive LTM scene parameters are
        # present in the forward pass but remain at their initialized values.
        p.requires_grad_("ltm" in active and not bool(freeze_scene_adaptive_ltm))
    if isp.lut is not None:
        isp.lut.lut.requires_grad_("lut" in active)
    if isp.raw_denoiser is not None:
        mask = denoiser_trainable_mask or {}
        for p in isp.raw_denoiser.parameters():
            p.requires_grad_(bool(allow_denoiser and mask.get(id(p), False)))
    return active


def _stage_trainable_count(params):
    return int(sum(p.numel() for p in params if p.requires_grad))


def build_optimizer_for_isp_stage(isp: ResidualISPV4, args, stage: str, denoiser_trainable_mask=None):
    allow_denoiser = (normalize_isp_stage_name(stage) == "joint" and args.denoiser_trainable != "none")
    active = set_isp_trainable_stage(
        isp, stage, denoiser_trainable_mask, allow_denoiser=allow_denoiser,
        freeze_scene_adaptive_ltm=bool(getattr(args, "freeze_scene_adaptive_ltm", False)),
    )
    groups = []

    def add_group(params, lr, name):
        trainable = [p for p in params if p.requires_grad]
        if trainable:
            groups.append({"params": trainable, "lr": lr, "base_lr": lr, "name": name})

    add_group([isp.du_s, isp.du_wb], args.lr, "exposure_wb")
    add_group([isp.du_ccm, isp.du_gtm, isp.du_sat], args.lr * 0.5, "ccm_tone_sat")
    add_group(_ltm_params(isp), args.lr * 0.25, "ltm")
    if isp.lut is not None:
        lut_lr_mult = 0.5 if bool(getattr(isp.lut, "chroma_only", False)) else 0.1
        add_group([isp.lut.lut], args.lr * lut_lr_mult, "lut")
    if isp.raw_denoiser is not None and allow_denoiser:
        add_group([p for p in isp.raw_denoiser.parameters()], args.denoiser_lr, "raw_denoiser")
    if not groups:
        raise RuntimeError(f"No trainable parameters for ISP stage {stage}")
    group_desc = ", ".join(f"{g.get('name', '?')}:{_stage_trainable_count(g['params'])}@{g['lr']:.2g}" for g in groups)
    return torch.optim.Adam(groups, betas=(0.9, 0.99)), active, group_desc


def _safe_stage_label(stage_index: int, stage_name: str, end_iter: int):
    return f"{int(stage_index):02d}_{stage_name}_end{int(end_iter):04d}"


def _tensor_to_pil_rgb(rgb01: torch.Tensor):
    x = rgb01.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def save_tensor_montage(path: Path, items, tile_w=260):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    tiles = []
    for label, tensor in items:
        img = _tensor_to_pil_rgb(tensor)
        scale = float(tile_w) / max(1, img.width)
        thumb = img.resize((tile_w, max(1, int(round(img.height * scale)))), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, thumb.height + 30), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((6, 7), str(label), fill=(0, 0, 0), font=font)
        tile.paste(thumb, (0, 30))
        tiles.append(tile)
    canvas = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    canvas.save(path)


def _lut_delta_stats(isp: ResidualISPV4):
    if isp.lut is None:
        return {}
    lut = isp.lut.lut.detach()
    if bool(getattr(isp.lut, "chroma_only", False)):
        delta_scale = float(getattr(isp.lut, "delta_scale", 1.0))
        delta = torch.tanh(lut) * delta_scale
        return {
            "lut_kind": str(getattr(isp.lut, "lut_kind", "chroma")),
            "lut_delta_abs_mean": float(delta.abs().mean()),
            "lut_delta_abs_max": float(delta.abs().max()),
            "lut_min": float(lut.min()),
            "lut_max": float(lut.max()),
            "lut_params": int(lut.numel()),
        }
    grid = torch.linspace(0.0, 1.0, lut.shape[0], device=lut.device, dtype=lut.dtype)
    r, g, b = torch.meshgrid(grid, grid, grid, indexing="ij")
    identity = torch.stack([r, g, b], dim=-1)
    delta = (lut - identity).abs()
    return {
        "lut_kind": str(getattr(isp.lut, "lut_kind", "rgb")),
        "lut_delta_abs_mean": float(delta.mean()),
        "lut_delta_abs_max": float(delta.max()),
        "lut_min": float(lut.min()),
        "lut_max": float(lut.max()),
        "lut_params": int(lut.numel()),
    }


def isp_parameter_snapshot(isp: ResidualISPV4):
    with torch.no_grad():
        a, b, c = isp.gtm_params(isp.du_gtm.dtype, isp.du_gtm.device)
        ltm_param = isp.du_ltm if hasattr(isp, "du_ltm") else isp.ltm_maps
        snap = {
            "exposure": float(isp.exposure().detach().flatten()[0]),
            "wb_gain_rgb": [float(v) for v in isp.wb_gain().detach().flatten()],
            "ccm": isp.ccm().detach().cpu().numpy().tolist(),
            "gtm_abc": [float(a), float(b), float(c)],
            "saturation": float(isp.saturation().detach().flatten()[0]),
            "du_s": float(isp.du_s.detach().flatten()[0]),
            "du_wb": isp.du_wb.detach().cpu().numpy().tolist(),
            "du_ccm_abs_mean": float(isp.du_ccm.detach().abs().mean()),
            "du_ccm_abs_max": float(isp.du_ccm.detach().abs().max()),
            "du_gtm": isp.du_gtm.detach().cpu().numpy().tolist(),
            "du_sat": float(isp.du_sat.detach().flatten()[0]),
            "ltm_maps_abs_mean": float(ltm_param.detach().abs().mean()),
            "ltm_maps_abs_max": float(ltm_param.detach().abs().max()),
        }
        if hasattr(isp, "scene_adaptive_ltm_params"):
            snap["scene_adaptive_ltm"] = {
                k: float(v.detach().flatten()[0])
                for k, v in isp.scene_adaptive_ltm_params().items()
            }
        snap.update(_lut_delta_stats(isp))
    return snap


def neutral_region_diagnostics(dbg: dict, aligned_phone: torch.Tensor, valid_mask: torch.Tensor):
    with torch.no_grad():
        lab = srgb_to_lab_fast(aligned_phone.clamp(0, 1))
        chroma = lab[:, 1:3].pow(2).sum(1, keepdim=True).sqrt()
        wp = aligned_phone.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        y = (srgb_to_linear(aligned_phone.clamp(0, 1)) * wp).sum(1, keepdim=True)
        neutral = ((valid_mask > 0.2) & (chroma < 8.0) & (y > 0.05) & (y < 0.85)).float()
        if int(neutral.sum().item()) < 1024:
            neutral = (valid_mask > 0.2).float()

        def channel_mean_and_ratio(x):
            m = neutral
            denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            mean = (x.clamp(0, 1) * m).sum(dim=(2, 3), keepdim=True) / denom
            r, g, b = mean[:, 0:1], mean[:, 1:2], mean[:, 2:3]
            return {
                "mean_rgb": [float(v) for v in mean.detach().flatten()],
                "r_over_g": float((r / (g + 1e-6)).detach().flatten()[0]),
                "b_over_g": float((b / (g + 1e-6)).detach().flatten()[0]),
            }

        return {
            "neutral_mask_mean": float(neutral.mean()),
            "after_exposure": channel_mean_and_ratio(dbg["after_exposure"]),
            "after_wb": channel_mean_and_ratio(dbg["after_wb"]),
            "aligned_phone": channel_mean_and_ratio(aligned_phone),
        }


def save_stage_snapshot(out_dir: Path, stage_index: int, stage_name: str, end_iter: int, isp: ResidualISPV4, raw: torch.Tensor, exp_t: torch.Tensor, aligned_phone: torch.Tensor, valid_mask: torch.Tensor, save_images: bool = False):
    label = _safe_stage_label(stage_index, stage_name, end_iter)
    stage_dir = out_dir / "stage_snapshots" / label
    stage_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        dbg = isp.forward_debug(raw, exp_t)
        outputs = [
            ("aligned phone", aligned_phone),
            ("linear raw", dbg["linear_rgb"]),
            ("after exposure", dbg["after_exposure"]),
            ("after wb", dbg["after_wb"]),
            ("after ccm", dbg["after_ccm"]),
            ("after gtm", dbg["after_gtm"]),
            ("after ltm", dbg["after_ltm"]),
            ("final", dbg["final"]),
        ]
        if save_images:
            for name, tensor in outputs[1:]:
                save_rgb(str(stage_dir / f"{name.replace(' ', '_')}.png"), tensor.clamp(0, 1))
            save_rgb(str(stage_dir / "aligned_phone.png"), aligned_phone)
            save_tensor_montage(stage_dir / "module_outputs_montage.png", outputs, tile_w=220)

            deltas = [
                ("awb abs x4", (dbg["after_wb"] - dbg["after_exposure"]).abs() * 4.0),
                ("ccm abs x4", (dbg["after_ccm"] - dbg["after_wb"]).abs() * 4.0),
                ("gtm abs x4", (dbg["after_gtm"] - dbg["after_ccm"].clamp(0, 1)).abs() * 4.0),
                ("ltm abs x4", (dbg["after_ltm"] - dbg["after_gtm"]).abs() * 4.0),
                ("lut abs x8", (dbg["after_lut"] - dbg["before_lut"]).abs() * 8.0),
            ]
            for name, tensor in deltas:
                save_rgb(str(stage_dir / f"delta_{name.replace(' ', '_')}.png"), tensor.clamp(0, 1))
            save_tensor_montage(stage_dir / "module_delta_montage.png", deltas, tile_w=220)

        metrics = compute_color_metrics(dbg["final"].clamp(0, 1), aligned_phone, valid_mask)
        params = isp_parameter_snapshot(isp)
        neutral = neutral_region_diagnostics(dbg, aligned_phone, valid_mask)
        summary = {
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            "end_iter": int(end_iter),
            "metrics_final_vs_aligned_phone": metrics,
            "isp_params": params,
            "neutral_region_diagnostics": neutral,
        }
        with (stage_dir / "stage_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
        row = {
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            "end_iter": int(end_iter),
            "psnr_rgb": metrics["psnr_rgb"],
            "deltaE00_mean": metrics["deltaE00_mean"],
            "rgb_mae": metrics["rgb_mae"],
            "hue_error_mean_deg": metrics["hue_error_mean_deg"],
            "exposure": params["exposure"],
            "wb_r": params["wb_gain_rgb"][0],
            "wb_g": params["wb_gain_rgb"][1],
            "wb_b": params["wb_gain_rgb"][2],
            "neutral_after_exp_r_over_g": neutral["after_exposure"]["r_over_g"],
            "neutral_after_exp_b_over_g": neutral["after_exposure"]["b_over_g"],
            "neutral_after_wb_r_over_g": neutral["after_wb"]["r_over_g"],
            "neutral_after_wb_b_over_g": neutral["after_wb"]["b_over_g"],
            "neutral_phone_r_over_g": neutral["aligned_phone"]["r_over_g"],
            "neutral_phone_b_over_g": neutral["aligned_phone"]["b_over_g"],
            "ltm_maps_abs_mean": params["ltm_maps_abs_mean"],
            "lut_delta_abs_mean": params.get("lut_delta_abs_mean", 0.0),
            "snapshot_dir": str(stage_dir),
        }
    return row


def write_stage_summary(out_dir: Path, rows):
    if not rows:
        return
    with (out_dir / "stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(rows), f, indent=2, ensure_ascii=False)
    fieldnames = list(rows[0].keys())
    with (out_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
