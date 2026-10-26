#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def image_tile(label: str, path: Path, width=300):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = width / max(1, img.width)
    thumb = img.resize((width, max(1, int(round(img.height * scale)))), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, thumb.height + 34), (245, 245, 245))
    draw = ImageDraw.Draw(tile)
    draw.text((7, 8), label, fill=(0, 0, 0), font=load_font(15))
    tile.paste(thumb, (0, 34))
    return tile


def paste_tiles_grid(tiles, out_path: Path, columns: int, gap: int = 10, bg=(24, 24, 26)):
    if not tiles:
        return
    columns = max(1, min(int(columns), len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)
    canvas = Image.new("RGB", (columns * cell_w + (columns + 1) * gap, rows * cell_h + (rows + 1) * gap), bg)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        canvas.paste(tile, (gap + c * (cell_w + gap), gap + r * (cell_h + gap)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_case_compare(case, out_dir: Path):
    items = [
        ("reference RGB", case.phone),
        ("reference aligned", out_dir / "phone_aligned.png"),
        ("predicted RGB", out_dir / "predicted_rgb.png"),
        ("RoMa certainty", out_dir / "roma_certainty.png"),
        ("valid mask", out_dir / "valid_mask.png"),
    ]
    tiles = [image_tile(label, path, width=320) for label, path in items if path.exists()]
    paste_tiles_grid(tiles, out_dir / "compare_alignment_predicted.png", columns=len(tiles), bg=(255, 255, 255))


def summarize_cases(cases, out_root: Path):
    vis_dir = out_root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pred_tiles = []
    compare_tiles = []
    per_place_tiles = {}
    for case in cases:
        out_dir = out_root / case.place / case.folder
        pred = out_dir / "predicted_rgb.png"
        if not pred.exists():
            continue
        save_case_compare(case, out_dir)
        metrics_path = out_dir / "color_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        align = metrics.get("alignment", {})
        feat = align.get("feature_info", {})
        label = f"{case.place} {case.folder} fps, exp {case.exposure_us:.1f} us"
        tile = image_tile(label, pred, width=260)
        pred_tiles.append(tile)
        per_place_tiles.setdefault(case.place, []).append(tile)
        compare = out_dir / "compare_alignment_predicted.png"
        if compare.exists():
            compare_tiles.append(image_tile(label, compare, width=900))
        rows.append(
            {
                "case": case.case_id,
                "frame_rate": case.frame_rate,
                "exposure_us": case.exposure_us,
                "reference_type": case.reference_type,
                "raw": str(case.raw),
                "reference_rgb": str(case.phone),
                "predicted_rgb": str(pred),
                "compare": str(compare),
                "psnr_rgb": metrics.get("psnr_rgb", ""),
                "deltaE00_mean": metrics.get("deltaE00_mean", ""),
                "rgb_mae": metrics.get("rgb_mae", ""),
                "hue_error_mean_deg": metrics.get("hue_error_mean_deg", ""),
                "mask_mean": metrics.get("mask_mean", ""),
                "alignment_mode": align.get("mode", ""),
                "edge_ncc": align.get("edge_ncc", ""),
                "roma_certainty_mean": feat.get("certainty_mean", ""),
                "valid_ratio": align.get("valid_ratio", ""),
            }
        )
    if rows:
        with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# PFV4 RoMa-only ISP results",
            "",
            "| Case | FPS | Exposure us | PSNR | DeltaE00 | RGB MAE | Hue deg | Mask | Edge NCC | RoMa Cert | Valid | Predicted RGB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            def fnum(key, nd=3):
                return f"{float(r[key]):.{nd}f}" if r[key] != "" and r[key] is not None else ""
            lines.append(
                f"| {r['case']} | {r['frame_rate']} | {float(r['exposure_us']):.1f} | "
                f"{fnum('psnr_rgb')} | {fnum('deltaE00_mean')} | {fnum('rgb_mae', 4)} | "
                f"{fnum('hue_error_mean_deg')} | {fnum('mask_mean')} | {fnum('edge_ncc')} | "
                f"{fnum('roma_certainty_mean')} | {fnum('valid_ratio')} | {r['predicted_rgb']} |"
            )
        lines.extend(
            [
                "",
                f"Final predicted RGB grid: {vis_dir / 'final_predicted_rgb_grid.png'}",
                f"Alignment/predicted compare grid: {vis_dir / 'alignment_predicted_compare_grid.png'}",
            ]
        )
        (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paste_tiles_grid(pred_tiles, vis_dir / "final_predicted_rgb_grid.png", columns=5)
    paste_tiles_grid(compare_tiles, vis_dir / "alignment_predicted_compare_grid.png", columns=1)
    for place, tiles in per_place_tiles.items():
        paste_tiles_grid(tiles, vis_dir / f"{place}_final_predicted_rgb_grid.png", columns=3)
    return rows
