#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PFV4Case:
    place: str
    folder: str
    raw: Path
    phone: Path
    reference_type: str
    width: int
    height: int
    exposure_us: float
    frame_rate: Optional[float]
    cihx: Path
    sensor_xpos: int = 0
    sensor_ypos: int = 0

    @property
    def case_id(self) -> str:
        return f"{self.place}/{self.folder}"


def _xml_text(root: ET.Element, path: str, default: Optional[str] = None) -> Optional[str]:
    node = root.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def read_cihx_metadata(path: Path) -> Dict[str, object]:
    data = path.read_bytes()
    start = data.find(b"<?xml")
    if start < 0:
        raise ValueError(f"No XML payload found in {path}")
    end = data.find(b"</cih>", start)
    if end < 0:
        raise ValueError(f"No </cih> XML terminator found in {path}")
    xml_bytes = data[start:end + len(b"</cih>")]
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    width = int(_xml_text(root, "./imageDataInfo/resolution/width"))
    height = int(_xml_text(root, "./imageDataInfo/resolution/height"))
    bit_depth = int(_xml_text(root, "./imageDataInfo/effectiveBit/depth", "12"))
    effective_side = str(_xml_text(root, "./imageDataInfo/effectiveBit/side", "Lower"))
    shutter_ns = float(_xml_text(root, "./recordInfo/shutterSpeedNsec"))
    frame_rate_text = _xml_text(root, "./recordInfo/recordRate", None)
    frame_rate = None if frame_rate_text is None else float(frame_rate_text)
    # Photron CIHX records the physical sensor origin of the acquired ROI.
    # These are deliberately not inferred from display/crop settings.
    sensor_xpos = int(_xml_text(root, "./imageDataInfo/segmentPos/sensorXpos", "0"))
    sensor_ypos = int(_xml_text(root, "./imageDataInfo/segmentPos/sensorYpos", "0"))
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "effective_side": effective_side,
        "exposure_us": shutter_ns / 1000.0,
        "frame_rate": frame_rate,
        "sensor_xpos": sensor_xpos,
        "sensor_ypos": sensor_ypos,
    }


def frame_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def resolve_reference_image(place_dir: Path, folder_name: str, reference_mode: str = "phone", reference_name: str = "") -> Tuple[Path, str]:
    if str(reference_name).strip():
        p = place_dir / str(reference_name).strip()
        if p.exists():
            return p, p.stem
        raise FileNotFoundError(f"Missing requested reference image {p}")
    for name in ("phone.jpg", "phone.jpeg", "phone.png", "Phone.jpg", "Phone.jpeg", "Phone.png"):
        p = place_dir / name
        if p.exists():
            return p, "phone"
    if str(reference_mode).lower() == "phone":
        raise FileNotFoundError(f"Missing phone GT reference in {place_dir}")
    camera_dir = place_dir / "Camera_1"
    if camera_dir.exists():
        for suffix in (".png", ".jpg", ".jpeg"):
            p = camera_dir / f"{folder_name}{suffix}"
            if p.exists():
                return p, "camera_1_exact"
    raise FileNotFoundError(f"Missing phone reference in {place_dir}")


def discover_cases(root: Path, places: Sequence[str], selected: Sequence[str] = (), reference_mode: str = "phone", reference_name: str = "") -> List[PFV4Case]:
    selected_set = {x.strip() for x in selected if str(x).strip()}
    cases: List[PFV4Case] = []
    for place in places:
        place_dir = root / place
        if not place_dir.exists():
            raise FileNotFoundError(place_dir)
        folders = sorted([p for p in place_dir.iterdir() if p.is_dir() and p.name.lower() != "camera_1"], key=lambda p: p.name)
        for folder in folders:
            keys = {place, folder.name, f"{place}/{folder.name}", f"{place}_{folder.name}"}
            if selected_set and not keys.intersection(selected_set):
                continue
            raws = sorted(folder.glob("*.raww"), key=lambda p: (frame_number(p), p.name))
            if not raws:
                continue
            cihx_files = sorted(folder.glob("*.cihx"))
            if not cihx_files:
                raise FileNotFoundError(f"Missing .cihx metadata in {folder}")
            meta = read_cihx_metadata(cihx_files[0])
            reference, ref_type = resolve_reference_image(place_dir, folder.name, reference_mode=reference_mode, reference_name=reference_name)
            cases.append(PFV4Case(
                place=place,
                folder=folder.name,
                raw=raws[0],
                phone=reference,
                reference_type=ref_type,
                width=int(meta["width"]),
                height=int(meta["height"]),
                exposure_us=float(meta["exposure_us"]),
                frame_rate=meta.get("frame_rate"),
                cihx=cihx_files[0],
                sensor_xpos=int(meta.get("sensor_xpos", 0)),
                sensor_ypos=int(meta.get("sensor_ypos", 0)),
            ))
    if not cases:
        raise RuntimeError(f"No PFV4 cases found under {root}")
    return cases


def parse_list(text: str):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def schedule_steps(schedule: str) -> int:
    total = 0
    for token in parse_list(schedule):
        if ":" not in token:
            raise ValueError(f"Bad schedule entry: {token}")
        total += int(token.split(":", 1)[1])
    return total
