#!/usr/bin/env python3
"""Run one bundled example adaptation and render its five reviewer frames."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def save_builtin_ours_grid(
    built_in_paths: list[Path],
    ours_paths: list[Path],
    frame_numbers: list[int],
    out_path: Path,
) -> None:
    """Save five built-in frames on row one and five adapted frames on row two."""
    if len(built_in_paths) != 5 or len(ours_paths) != 5 or len(frame_numbers) != 5:
        raise ValueError("The reviewer grid must contain exactly five frames per row.")

    tile_w = 512
    source_size = Image.open(built_in_paths[0]).size
    if source_size[0] <= 0 or source_size[1] <= 0:
        raise ValueError(f"Invalid reviewer frame size: {source_size}")
    tile_h = max(1, round(tile_w * source_size[1] / source_size[0]))
    label_h = 42
    row_label_w = 150
    gap = 8
    panels: list[list[Image.Image]] = []
    for paths in (built_in_paths, ours_paths):
        row = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            if image.size != source_size:
                raise ValueError(f"Reviewer frames must share one size, got {image.size} from {path}")
            if image.size != (tile_w, tile_h):
                image = image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            row.append(image)
        panels.append(row)
    canvas_w = row_label_w + 5 * tile_w + 6 * gap
    canvas_h = 2 * (label_h + tile_h) + 3 * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "black")
    draw = ImageDraw.Draw(canvas)
    row_font = _font(32)
    frame_font = _font(22)
    for row_index, (row_name, row_images) in enumerate(zip(("built-in", "ours"), panels)):
        y0 = gap + row_index * (label_h + tile_h + gap)
        row_box = draw.textbbox((0, 0), row_name, font=row_font)
        draw.text(
            (gap, y0 + label_h + (tile_h - (row_box[3] - row_box[1])) / 2 - row_box[1]),
            row_name,
            fill="white",
            font=row_font,
        )
        for col, image in enumerate(row_images):
            x0 = row_label_w + gap + col * (tile_w + gap)
            frame_label = f"{frame_numbers[col]:06d}"
            draw.text((x0 + 4, y0 + 5), frame_label, fill="white", font=frame_font)
            canvas.paste(image, (x0, y0 + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _clean_output_images(out_dir: Path, keep: set[Path]) -> None:
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix.lower() in image_suffixes and path not in keep:
            path.unlink()


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument(
        "--video3",
        action="store_true",
        help="Adapt the bundled video3 example for 300 steps.",
    )
    scene_group.add_argument(
        "--video30",
        action="store_true",
        help="Adapt video30 from the default ISP initialization for 300 steps.",
    )
    scene_group.add_argument(
        "--video30-from-video3",
        action="store_true",
        help="Adapt video30 from the bundled video3 ISP checkpoint for 100 steps at lr=5e-3.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("ADAPTATION_DEVICE", "cuda:0"),
        help="PyTorch device (default: cuda:0; CPU fallback is automatic).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the adaptation checkpoint, timing, and final grids.",
    )
    args = parser.parse_args()

    transfer_video30 = bool(args.video30_from_video3)
    scratch_video30 = bool(args.video30)
    example_name = "video30" if (transfer_video30 or scratch_video30) else "video3"
    example_dir = root / "example" / example_name
    metadata = json.loads((example_dir / "metadata.json").read_text(encoding="utf-8"))
    if transfer_video30:
        default_output_name = "video30_from_video3_100steps"
    elif scratch_video30:
        default_output_name = "video30_from_scratch_300steps"
    else:
        default_output_name = example_name
    out_dir = args.output_dir if args.output_dir is not None else root / "example" / "output" / default_output_name
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = example_dir / metadata["raw_file"]
    phone = example_dir / metadata["phone_reference"]
    denoiser = root / "run" / "checkpoints" / "step50000.pth"
    adaptation_command = [
        sys.executable, "-u", "-m", "run.train_single",
        "--hs_raw", str(raw), "--raw_h", str(metadata["height"]),
        "--raw_w", str(metadata["width"]), "--exposure_t", str(metadata["exposure_us"]),
        "--phone_rgb", str(phone), "--denoiser_ckpt", str(denoiser),
        "--device", str(args.device), "--out_dir", str(out_dir),
        "--out", str(out_dir / "predicted_rgb.png"),
        "--save_isp_ckpt", str(out_dir / "isp_checkpoint.pth"),
    ]
    if transfer_video30:
        init_ckpt = root / "example" / "output" / "video3" / "isp_checkpoint.pth"
        if not init_ckpt.exists():
            raise FileNotFoundError(
                f"The video3 checkpoint is missing: {init_ckpt}. "
                "Run `python run_adaptation.py` once first."
            )
        adaptation_command.extend(
            [
                "--init_isp_ckpt", str(init_ckpt),
                "--steps", "100",
                "--isp_stage_schedule", "joint:100",
                "--lr", "5e-3",
                "--lr_schedule", "constant",
            ]
        )
    elif scratch_video30:
        adaptation_command.extend(
            [
                "--steps", "300",
            ]
        )
    print("[adaptation command]", " ".join(adaptation_command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(adaptation_command, cwd=root, env=env, check=True)

    render_raw_dir = example_dir / metadata["render_raw_dir"]
    render_built_in_dir = example_dir / metadata["render_built_in_dir"]
    render_work = out_dir / "_render_video5"
    if render_work.exists():
        shutil.rmtree(render_work)
    render_command = [
        sys.executable, "-u", "-m", "run.render_frames",
        "--isp_ckpt", str(out_dir / "isp_checkpoint.pth"),
        "--raw_dir", str(render_raw_dir), "--built_in_dir", str(render_built_in_dir),
        "--out_dir", str(render_work), "--case", f"{example_name}_5_frames", "--count", "5",
        "--raw_h", str(metadata["height"]), "--raw_w", str(metadata["width"]),
        "--exposure_t", str(metadata["exposure_us"]), "--denoiser_ckpt", str(denoiser),
        "--device", str(args.device), "--no-save_raw_demosaic_vis", "--no-save_denoised_vis",
    ]
    print("[render command]", " ".join(render_command), flush=True)
    subprocess.run(render_command, cwd=root, env=env, check=True)
    render_summary = json.loads((render_work / "summary.json").read_text(encoding="utf-8"))
    render_timing = render_summary["timing"]

    raw_paths = sorted(render_raw_dir.glob("*.raww"))
    frame_numbers = list(metadata["render_frame_numbers"])
    if len(raw_paths) != 5:
        raise RuntimeError(f"Expected five bundled render RAW frames, found {len(raw_paths)}")
    built_in_paths = [render_work / "frames" / path.stem / "built_in_rgb.png" for path in raw_paths]
    ours_paths = [render_work / "frames" / path.stem / "predicted_rgb.png" for path in raw_paths]
    if not all(path.exists() for path in built_in_paths + ours_paths):
        raise RuntimeError("The five-frame renderer did not produce all built-in and ours images.")

    grid_path = out_dir / "built-in_vs_ours.png"
    save_builtin_ours_grid(built_in_paths, ours_paths, frame_numbers, grid_path)
    phone_path = out_dir / "phone.png"
    # Apply the source JPEG orientation before writing PNG; PNG does not
    # preserve the EXIF orientation tag used by the bundled video30 phone.
    with Image.open(phone) as phone_image:
        ImageOps.exif_transpose(phone_image).convert("RGB").save(phone_path)
    shutil.rmtree(render_work)
    _clean_output_images(out_dir, {grid_path, phone_path})

    selection = {
        "source_total_frames": int(metadata["render_source_total_frames"]),
        "selected_frame_numbers": frame_numbers,
        "render_raw_dir": str(render_raw_dir.relative_to(root)),
        "render_built_in_dir": str(render_built_in_dir.relative_to(root)),
        "isp_render_seconds": render_timing["isp_render_seconds"],
        "isp_render_ms_per_frame": render_timing["isp_render_ms_per_frame"],
        "isp_render_fps": render_timing["isp_render_fps"],
        "image_io_excluded": bool(render_timing["image_io_excluded"]),
        "warmup_for_speed_measurement": int(render_timing.get("warmup_for_speed_measurement", 0)),
    }
    (out_dir / "render_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    timing = json.loads((out_dir / "timing.json").read_text(encoding="utf-8"))
    print(f"[result] grid={grid_path}", flush=True)
    print(f"[result] phone={phone_path}", flush=True)
    print(f"[result] adaptation_seconds={timing['adaptation_seconds']:.3f}", flush=True)
    print(f"[result] isp_render_seconds={render_timing['isp_render_seconds']:.6f}", flush=True)
    print(f"[result] isp_render_fps={render_timing['isp_render_fps']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
