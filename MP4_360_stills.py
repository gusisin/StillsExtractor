#!/usr/bin/env python3
"""Standalone Windows-friendly GUI: 360 equirect video to cubemap PNG stills per sample.

Extracts pinhole cubemap faces (f/r/b/l/u/d) for use with Splatter, Postshot, COLMAP, etc.
No COLMAP or training - image prep only.
"""

from __future__ import annotations

import copy
import io
import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any

try:
    from PIL import Image, ImageChops, ImageDraw
    from PIL.ImageTk import PhotoImage
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow") from None

from training_recommendations import plan_from_planned_stills

# --- Cubemap (matches Splatter / ffmpeg v360 conventions) -------------------

EQUIRECT_FACE_FOV_DEG = 90
EQUIRECT_DEFAULT_TILE_MAX_WIDTH = 1920
FACE_PREVIEW_THUMB_SIZE = 160
FACE_MASK_EDITOR_SIZE = 512
EQUIRECT_MASK_EDITOR_WIDTH = 1024
EQUIRECT_MASK_RASTER_WIDTH = 2048

# Normalized polygon rings per face: list of (x, y) with 0..1 coords.
FaceMaskPolygons = list[list[tuple[float, float]]]
FaceMaskMap = dict[str, FaceMaskPolygons]

_STILL_FACE_RE = re.compile(r"^(.+)-(\d{6})_([frblud])\.png$", re.IGNORECASE)

EQUIRECT_CUBE_FACES: tuple[tuple[str, int, int], ...] = (
    ("f", 180, 0),
    ("r", -90, 0),  # yaw=270 fails in ffmpeg v360
    ("b", 0, 0),
    ("l", 90, 0),
    ("u", 0, 90),
    ("d", 0, -90),
)

FACE_LABELS = {
    "f": "front",
    "r": "right",
    "b": "back",
    "l": "left",
    "u": "up",
    "d": "down",
}

CUBEMAP_FACE_ORDER = "frblud"

FACE_SET_PRESETS: dict[str, tuple[str, frozenset[str]]] = {
    "all": ("All faces (f, r, b, l, u, d)", frozenset("frblud")),
    "horizontal": ("Horizontal only (f, r, b, l)", frozenset("frbl")),
    "horizontal_down": ("Horizontal + down (f, r, b, l, d)", frozenset("frbld")),
}


def _faces_in_set(face_set: str) -> frozenset[str]:
    entry = FACE_SET_PRESETS.get(face_set)
    return entry[1] if entry else FACE_SET_PRESETS["all"][1]


def _face_jobs_for_set(face_set: str) -> list[tuple[str, int, int]]:
    allowed = _faces_in_set(face_set)
    return _face_jobs_for_faces(allowed)


def _face_jobs_for_faces(allowed: frozenset[str]) -> list[tuple[str, int, int]]:
    return [job for job in EQUIRECT_CUBE_FACES if job[0] in allowed]


def _face_count_for_set(face_set: str) -> int:
    return len(_face_jobs_for_set(face_set))


def _face_count_for_faces(allowed: frozenset[str]) -> int:
    return len(_face_jobs_for_faces(allowed))


def _selected_faces_label(faces: frozenset[str]) -> str:
    for key in FACE_SET_PRESETS:
        preset_faces = _faces_in_set(key)
        if faces == preset_faces:
            return FACE_SET_PRESETS[key][0]
    ordered = [face for face in CUBEMAP_FACE_ORDER if face in faces]
    if not ordered:
        return "No faces selected"
    labels = [f"{FACE_LABELS.get(face, face)} ({face})" for face in ordered]
    return "Custom: " + ", ".join(labels)


def _normalize_v360_angle(deg: int) -> int:
    d = int(deg) % 360
    return d - 360 if d > 180 else d


def _effective_max_width(max_width: int) -> int:
    w = int(max_width)
    return w if w > 0 else EQUIRECT_DEFAULT_TILE_MAX_WIDTH


def _equirect_input_normalize_vf() -> str:
    """Square pixels and 2:1 equirect canvas before v360 (fixes 16:9 mezzanine / SAR)."""
    return (
        "setsar=1,"
        "scale=2*ih:ih:force_original_aspect_ratio=decrease,"
        "pad=2*ih:ih:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def _equirect_square_face_scale_vf(tile_w: int) -> str:
    """Guarantee square cubemap face output after v360."""
    return (
        f"scale={tile_w}:{tile_w}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_w},"
        "setsar=1"
    )


def _equirect_v360_rectilinear_vf(yaw: int, pitch: int, tile_w: int) -> str:
    yaw = _normalize_v360_angle(yaw)
    pitch = _normalize_v360_angle(pitch)
    h_fov = EQUIRECT_FACE_FOV_DEG
    return (
        f"v360=input=equirect:output=rectilinear:yaw={yaw}:pitch={pitch}:roll=0:"
        f"ih_fov=360:iv_fov=180:h_fov={h_fov}:v_fov={h_fov}:w={tile_w}:h={tile_w}"
    )


def _build_equirect_face_vf(yaw: int, pitch: int, max_width: int) -> str:
    tile_w = _effective_max_width(max_width)
    return (
        f"{_equirect_input_normalize_vf()},"
        f"{_equirect_v360_rectilinear_vf(yaw, pitch, tile_w)},"
        f"{_equirect_square_face_scale_vf(tile_w)}"
    )


def _build_equirect_vf(sample_fps: float, yaw: int, pitch: int, max_width: int) -> str:
    tile_w = _effective_max_width(max_width)
    return (
        f"{_equirect_input_normalize_vf()},"
        f"fps={sample_fps:.6f},"
        f"{_equirect_v360_rectilinear_vf(yaw, pitch, tile_w)},"
        f"{_equirect_square_face_scale_vf(tile_w)}"
    )


def _ffmpeg_single_frame_png(
    src: Path,
    vf: str,
    time_sec: float = 0.0,
) -> bytes | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
    ]
    if time_sec > 0:
        cmd.extend(["-ss", f"{time_sec:.6f}"])
    cmd.extend(
        [
            "-i",
            str(src),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
    )
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _exclude_mask_from_polygons(
    size: tuple[int, int],
    polygons: FaceMaskPolygons,
) -> Image.Image:
    """L mode: 255 inside exclude polygons, 0 elsewhere."""
    w, h = size
    exclude = Image.new("L", (w, h), 0)
    if not polygons:
        return exclude
    draw = ImageDraw.Draw(exclude)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = [(x * w, y * h) for x, y in poly]
        draw.polygon(pts, fill=255)
    return exclude


def _apply_exclude_mask_to_image(
    image: Image.Image,
    exclude_mask: Image.Image,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    rgb = image.convert("RGB")
    black = Image.new("RGB", rgb.size, fill)
    return Image.composite(black, rgb, exclude_mask)


def _render_face_preview_with_mask_overlay(
    image: Image.Image,
    polygons: FaceMaskPolygons,
    extra_exclude: Image.Image | None = None,
) -> Image.Image:
    rgb = image.convert("RGB")
    exclude = _exclude_mask_from_polygons(rgb.size, polygons)
    if extra_exclude is not None:
        if extra_exclude.size != rgb.size:
            extra_exclude = extra_exclude.resize(rgb.size, Image.Resampling.NEAREST)
        exclude = ImageChops.lighter(exclude, extra_exclude)
    if exclude.getextrema()[1] == 0:
        return rgb
    overlay = Image.new("RGB", rgb.size, (200, 50, 50))
    return Image.composite(overlay, rgb, exclude)


def _face_masks_active(face_masks: FaceMaskMap) -> bool:
    return any(polygons for polygons in face_masks.values())


def _masks_active(face_masks: FaceMaskMap, equirect_polygons: FaceMaskPolygons) -> bool:
    return _face_masks_active(face_masks) or bool(equirect_polygons)


def _rasterize_equirect_exclude(
    polygons: FaceMaskPolygons,
    width: int = EQUIRECT_MASK_RASTER_WIDTH,
) -> Image.Image:
    height = width // 2
    return _exclude_mask_from_polygons((width, height), polygons)


def _ffmpeg_project_equirect_mask_to_face(
    equirect_exclude: Image.Image,
    yaw: int,
    pitch: int,
    face_size: int,
) -> Image.Image | None:
    """Project L-mode equirect exclude mask (255 inside exclude) to a square face exclude mask."""
    vf = _build_equirect_face_vf(yaw, pitch, face_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / "eq_mask.png"
        out_path = tmp / "face_mask.png"
        rgb = Image.new("RGB", equirect_exclude.size, (0, 0, 0))
        rgb.paste((255, 255, 255), mask=equirect_exclude)
        rgb.save(in_path, format="PNG")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-y",
            "-i",
            str(in_path),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not out_path.is_file():
            return None
        with Image.open(out_path) as im:
            gray = im.convert("L")
            return gray.point(lambda px: 255 if px > 32 else 0)


def _build_equirect_exclude_templates(
    equirect_polygons: FaceMaskPolygons,
    face_size: int,
) -> dict[str, Image.Image]:
    if not equirect_polygons:
        return {}
    eq_raster = _rasterize_equirect_exclude(equirect_polygons)
    tile_w = _effective_max_width(face_size)
    templates: dict[str, Image.Image] = {}
    for face, yaw, pitch in EQUIRECT_CUBE_FACES:
        projected = _ffmpeg_project_equirect_mask_to_face(eq_raster, yaw, pitch, tile_w)
        if projected is not None:
            templates[face] = projected
    return templates


def _combine_exclude_masks(
    size: tuple[int, int],
    face_polygons: FaceMaskPolygons,
    equirect_template: Image.Image | None,
) -> Image.Image:
    exclude = _exclude_mask_from_polygons(size, face_polygons)
    if equirect_template is not None:
        tmpl = (
            equirect_template
            if equirect_template.size == size
            else equirect_template.resize(size, Image.Resampling.NEAREST)
        )
        exclude = ImageChops.lighter(exclude, tmpl)
    return exclude


def _face_from_still_path(path: Path) -> str | None:
    match = _STILL_FACE_RE.match(path.name)
    return match.group(3).lower() if match else None


def _mask_sidecar_dir(video: Path) -> Path:
    """Mask definitions and companion PNGs live beside the source video, not in stills output."""
    return video.parent / f"{video.stem}_masks"


def _masks_json_path(video: Path) -> Path:
    return _mask_sidecar_dir(video) / "face_masks.json"


def _app_root() -> Path:
    return Path(__file__).resolve().parent


def _user_masks_dir() -> Path:
    path = _app_root() / "usermasks"
    path.mkdir(exist_ok=True)
    return path


def _sanitize_user_mask_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (raw or "").strip()).strip("-_")
    return cleaned[:64] if cleaned else "mask"


def _list_user_mask_names() -> list[str]:
    return sorted(p.stem for p in _user_masks_dir().glob("*.json"))


def _user_mask_path(name: str) -> Path:
    return _user_masks_dir() / f"{_sanitize_user_mask_name(name)}.json"


def _parse_masks_payload(data: dict[str, Any]) -> tuple[FaceMaskMap, FaceMaskPolygons, float]:
    faces_raw = data.get("faces") or {}
    face_masks: FaceMaskMap = {face: [] for face in CUBEMAP_FACE_ORDER}
    for face, polys in faces_raw.items():
        if face in face_masks and isinstance(polys, list):
            face_masks[face] = [
                [(float(pt[0]), float(pt[1])) for pt in poly]
                for poly in polys
                if isinstance(poly, list) and len(poly) >= 3
            ]
    equirect_raw = data.get("equirect") or []
    equirect_polygons: FaceMaskPolygons = [
        [(float(pt[0]), float(pt[1])) for pt in poly]
        for poly in equirect_raw
        if isinstance(poly, list) and len(poly) >= 3
    ]
    preview_time = float(data.get("preview_time_sec") or 0.0)
    return face_masks, equirect_polygons, preview_time


def _load_masks_json(path: Path) -> tuple[FaceMaskMap, FaceMaskPolygons, float] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _parse_masks_payload(data)


def _apply_face_masks_batch(
    paths: list[Path],
    face_masks: FaceMaskMap,
    equirect_polygons: FaceMaskPolygons,
    *,
    face_size: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Apply exclude polygons to stills. Returns (ok_stills, fail_stills)."""
    if not paths or not _masks_active(face_masks, equirect_polygons):
        return 0, 0
    equirect_templates = _build_equirect_exclude_templates(equirect_polygons, face_size)
    ok = fail = 0
    total = len(paths)
    for idx, path in enumerate(paths, 1):
        face = _face_from_still_path(path)
        if not face:
            if on_progress:
                on_progress(idx, total)
            continue
        face_polygons = face_masks.get(face, [])
        equirect_tmpl = equirect_templates.get(face)
        if not face_polygons and equirect_tmpl is None:
            if on_progress:
                on_progress(idx, total)
            continue
        try:
            with Image.open(path) as im:
                exclude = _combine_exclude_masks(im.size, face_polygons, equirect_tmpl)
                if exclude.getextrema()[1] == 0:
                    if on_progress:
                        on_progress(idx, total)
                    continue
                masked = _apply_exclude_mask_to_image(im, exclude)
                _save_still_png(masked, path)
            ok += 1
        except OSError:
            fail += 1
        if on_progress:
            on_progress(idx, total)
    return ok, fail


def _save_masks_json(
    path: Path,
    face_masks: FaceMaskMap,
    equirect_polygons: FaceMaskPolygons,
    preview_time_sec: float,
    *,
    video: Path | None = None,
    output_prefix: str | None = None,
    library_name: str | None = None,
    description: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 2,
        "preview_time_sec": preview_time_sec,
        "note": "Normalized 0-1 vertices; polygon interior excluded (black fill in stills).",
        "equirect": [[list(pt) for pt in poly] for poly in equirect_polygons],
        "faces": {
            face: [[list(pt) for pt in poly] for poly in polygons]
            for face, polygons in face_masks.items()
            if polygons
        },
    }
    if video is not None:
        payload["source_video"] = str(video)
    if output_prefix:
        payload["output_prefix"] = output_prefix
    if library_name:
        payload["library_name"] = library_name
    if description:
        payload["description"] = description
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_user_mask_library(
    name: str,
    face_masks: FaceMaskMap,
    equirect_polygons: FaceMaskPolygons,
    preview_time_sec: float,
    *,
    description: str | None = None,
) -> Path:
    safe_name = _sanitize_user_mask_name(name)
    path = _user_mask_path(safe_name)
    _save_masks_json(
        path,
        face_masks,
        equirect_polygons,
        preview_time_sec,
        library_name=safe_name,
        description=description,
    )
    return path


def _generate_face_previews(
    src: Path,
    size: int = FACE_MASK_EDITOR_SIZE,
    time_sec: float = 0.0,
) -> dict[str, Image.Image]:
    out: dict[str, Image.Image] = {}
    for face, yaw, pitch in EQUIRECT_CUBE_FACES:
        vf = _build_equirect_face_vf(yaw, pitch, size)
        data = _ffmpeg_single_frame_png(src, vf, time_sec)
        if not data:
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                out[face] = im.copy()
        except OSError:
            continue
    return out


def _generate_equirect_preview(
    src: Path,
    width: int = EQUIRECT_MASK_EDITOR_WIDTH,
    time_sec: float = 0.0,
) -> Image.Image | None:
    height = width // 2
    vf = f"{_equirect_input_normalize_vf()},scale={width}:{height}"
    data = _ffmpeg_single_frame_png(src, vf, time_sec)
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.copy()
    except OSError:
        return None


def _ffmpeg_extract_popen(
    src: Path,
    pattern: Path,
    vf: str,
    start_number: int,
) -> subprocess.Popen[Any]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-an",
        "-start_number",
        str(start_number),
        str(pattern),
    ]
    # Do not PIPE ffmpeg output - long runs fill the OS pipe buffer and ffmpeg blocks
    # after the last frame is written, which looks like a hang between face passes.
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _count_prefix_pngs(out_dir: Path, prefix: str) -> int:
    return len(list(out_dir.glob(f"{prefix}-*.png")))


def _count_face_pngs(out_dir: Path, prefix: str, face: str) -> int:
    return len(list(out_dir.glob(f"{prefix}-*_{face}.png")))


def _list_still_pngs(out_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        path
        for path in out_dir.glob(f"{prefix}-*.png")
        if _STILL_FACE_RE.match(path.name)
    )


def _save_still_png(image: Image.Image, path: Path) -> None:
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    tmp = path.with_name(f"{path.name}.tmp")
    rgb.save(tmp, format="PNG", pnginfo=PngInfo(), compress_level=6)
    tmp.replace(path)


def _strip_png_metadata(
    paths: list[Path],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    if not paths:
        return 0, 0
    total = len(paths)
    magick = shutil.which("magick")
    if magick:
        ok = fail = 0
        chunk_size = 40
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start : start + chunk_size]
            result = subprocess.run(
                [magick, "mogrify", "-strip", *[str(p) for p in chunk]],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                ok += len(chunk)
            else:
                for path in chunk:
                    try:
                        with Image.open(path) as im:
                            _save_still_png(im, path)
                        ok += 1
                    except OSError:
                        fail += 1
            if on_progress:
                on_progress(ok + fail, total)
        return ok, fail

    ok = fail = 0
    for path in paths:
        try:
            with Image.open(path) as im:
                _save_still_png(im, path)
            ok += 1
        except OSError:
            fail += 1
        if on_progress:
            on_progress(ok + fail, total)
    return ok, fail


def _probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "ffprobe failed")
    data = json.loads(result.stdout or "{}")
    duration = float(data.get("format", {}).get("duration") or 0.0)
    video = None
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video":
            video = stream
            break
    if not video:
        raise RuntimeError("No video stream found")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    aspect = width / height if height else 0.0
    sar = str(video.get("sample_aspect_ratio") or "1:1")
    dar = str(video.get("display_aspect_ratio") or "")
    hints: list[str] = []
    tags = video.get("tags") or {}
    for key in ("projection", "spherical", "com.apple.quicktime.projection"):
        val = str(tags.get(key) or "").lower()
        if "equirect" in val or "360" in val:
            hints.append("spherical_metadata")
    for entry in video.get("side_data_list") or []:
        if isinstance(entry, dict):
            sd = str(entry.get("side_data_type") or entry.get("type") or "").lower()
            if "spherical" in sd or "360" in sd:
                hints.append("spherical_metadata")
    if 1.85 <= aspect <= 2.15:
        hints.append("aspect_ratio_2_1")
    looks_equirect = "spherical_metadata" in hints or (
        "aspect_ratio_2_1" in hints and aspect >= 1.95
    )
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "aspect": aspect,
        "sample_aspect_ratio": sar,
        "display_aspect_ratio": dar,
        "hints": hints,
        "looks_equirect": looks_equirect,
    }


def _sanitize_prefix(raw: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (raw or "").strip()).strip("-_")
    if cleaned:
        return cleaned[:64]
    cleaned_fb = re.sub(r"[^a-zA-Z0-9_-]+", "-", fallback).strip("-_")
    return cleaned_fb or "stills"


def _estimate_sample_count(duration: float, interval_sec: float) -> int:
    """Match ffmpeg fps filter: samples at t=0, interval, 2*interval, while t < duration."""
    if duration <= 0 or interval_sec <= 0:
        return 0
    n = int(duration / interval_sec + 1e-6)
    if duration > n * interval_sec + 1e-6:
        n += 1
    return max(1, n)


# Outdoor cubemap PNGs ~2.4-3 B/px; use 2.75 for headroom.
_PNG_BYTES_PER_PIXEL = 2.75
_DISK_HEADROOM_FACTOR = 1.10


def _estimate_output_bytes(total_png: int, face_width: int) -> int:
    if total_png <= 0:
        return 0
    side = _effective_max_width(face_width)
    bytes_per_face = max(int(side * side * _PNG_BYTES_PER_PIXEL), 64 * 1024)
    return total_png * bytes_per_face


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.0f} MB"
    return f"{num_bytes / 1024:.0f} KB"


def _volume_for_disk_check(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    parent = path.parent
    while not parent.exists():
        if parent.parent == parent:
            return Path.cwd()
        parent = parent.parent
    return parent


def _disk_free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(_volume_for_disk_check(path)).free)
    except OSError:
        return -1


def _disk_space_check(out_dir: Path, required_bytes: int) -> tuple[bool, str]:
    free = _disk_free_bytes(out_dir)
    if free < 0:
        return True, "Could not read free disk space for the output location."
    need = int(required_bytes * _DISK_HEADROOM_FACTOR)
    if free >= need:
        return True, f"Disk: ~{_format_bytes(need)} needed, {_format_bytes(free)} free."
    return (
        False,
        f"Not enough disk space on the output drive.\n\n"
        f"Estimated need: ~{_format_bytes(need)} (includes {_DISK_HEADROOM_FACTOR:.0%} headroom)\n"
        f"Available: {_format_bytes(free)}\n\n"
        f"Choose another output folder, increase sample interval, lower face max width, "
        f"or choose a smaller face set (e.g. horizontal only).",
    )


# --- Face mask polygon editor ------------------------------------------------


class FaceMaskEditor(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        face: str,
        image: Image.Image,
        polygons: FaceMaskPolygons,
        on_save: Callable[[FaceMaskPolygons], None],
    ) -> None:
        super().__init__(parent)
        label = FACE_LABELS.get(face, face)
        self.title(f"Mask editor — {label} ({face})")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._size = FACE_MASK_EDITOR_SIZE
        self._image = image.copy()
        if self._image.size != (self._size, self._size):
            self._image = self._image.resize((self._size, self._size), Image.Resampling.LANCZOS)
        self._polygons: FaceMaskPolygons = [list(poly) for poly in polygons]
        self._current: list[tuple[float, float]] = []
        self._on_save = on_save
        self._photo: PhotoImage | None = None

        ttk.Label(
            self,
            text=(
                "Click to place vertices around regions to exclude (filled black in stills). "
                "Finish each polygon, then Save. Mask applies to all time samples on this face."
            ),
            wraplength=FACE_MASK_EDITOR_SIZE + 20,
        ).pack(padx=10, pady=(10, 4))

        self._canvas = tk.Canvas(
            self,
            width=self._size,
            height=self._size,
            highlightthickness=1,
            highlightbackground="#cccccc",
            cursor="crosshair",
        )
        self._canvas.pack(padx=10, pady=4)
        self._canvas.bind("<Button-1>", self._on_click)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(btn_row, text="Finish polygon", command=self._finish_polygon).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Clear all", command=self._clear_all).pack(side=tk.LEFT)

        ok_row = ttk.Frame(self)
        ok_row.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(ok_row, text="Save", command=self._save).pack(side=tk.LEFT)
        ttk.Button(ok_row, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=6)

        self._redraw()

    def _norm(self, x: int, y: int) -> tuple[float, float]:
        return (max(0.0, min(1.0, x / self._size)), max(0.0, min(1.0, y / self._size)))

    def _on_click(self, event: tk.Event) -> None:
        self._current.append(self._norm(event.x, event.y))
        self._redraw()

    def _finish_polygon(self) -> None:
        if len(self._current) < 3:
            messagebox.showwarning("Polygon", "Need at least 3 points to close a polygon.", parent=self)
            return
        self._polygons.append(list(self._current))
        self._current = []
        self._redraw()

    def _undo(self) -> None:
        if self._current:
            self._current.pop()
        elif self._polygons:
            self._polygons.pop()
        self._redraw()

    def _clear_all(self) -> None:
        self._current = []
        self._polygons = []
        self._redraw()

    def _redraw(self) -> None:
        preview_polys = list(self._polygons)
        if len(self._current) >= 3:
            preview_polys.append(list(self._current))
        vis = _render_face_preview_with_mask_overlay(self._image, preview_polys)
        if self._current:
            draw = ImageDraw.Draw(vis)
            pts = [(x * self._size, y * self._size) for x, y in self._current]
            for px, py in pts:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 220, 0))
            if len(pts) >= 2:
                draw.line(pts, fill=(255, 220, 0), width=2)
        self._photo = PhotoImage(vis)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _save(self) -> None:
        if len(self._current) >= 3:
            self._polygons.append(list(self._current))
            self._current = []
        self._on_save(self._polygons)
        self.destroy()


class EquirectMaskEditor(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        image: Image.Image,
        polygons: FaceMaskPolygons,
        on_save: Callable[[FaceMaskPolygons], None],
    ) -> None:
        super().__init__(parent)
        self.title("Mask editor — equirectangular (all faces)")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._width = EQUIRECT_MASK_EDITOR_WIDTH
        self._height = EQUIRECT_MASK_EDITOR_WIDTH // 2
        self._image = image.copy()
        if self._image.size != (self._width, self._height):
            self._image = self._image.resize(
                (self._width, self._height),
                Image.Resampling.LANCZOS,
            )
        self._polygons: FaceMaskPolygons = [list(poly) for poly in polygons]
        self._current: list[tuple[float, float]] = []
        self._on_save = on_save
        self._photo: PhotoImage | None = None

        ttk.Label(
            self,
            text=(
                "Draw on the full 360° frame once — excluded regions project to every cubemap face. "
                "Combine with per-face masks if needed. Applies to all time samples."
            ),
            wraplength=self._width + 20,
        ).pack(padx=10, pady=(10, 4))

        self._canvas = tk.Canvas(
            self,
            width=self._width,
            height=self._height,
            highlightthickness=1,
            highlightbackground="#cccccc",
            cursor="crosshair",
        )
        self._canvas.pack(padx=10, pady=4)
        self._canvas.bind("<Button-1>", self._on_click)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(btn_row, text="Finish polygon", command=self._finish_polygon).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Clear all", command=self._clear_all).pack(side=tk.LEFT)

        ok_row = ttk.Frame(self)
        ok_row.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(ok_row, text="Save", command=self._save).pack(side=tk.LEFT)
        ttk.Button(ok_row, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=6)

        self._redraw()

    def _norm(self, x: int, y: int) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, x / self._width)),
            max(0.0, min(1.0, y / self._height)),
        )

    def _on_click(self, event: tk.Event) -> None:
        self._current.append(self._norm(event.x, event.y))
        self._redraw()

    def _finish_polygon(self) -> None:
        if len(self._current) < 3:
            messagebox.showwarning("Polygon", "Need at least 3 points to close a polygon.", parent=self)
            return
        self._polygons.append(list(self._current))
        self._current = []
        self._redraw()

    def _undo(self) -> None:
        if self._current:
            self._current.pop()
        elif self._polygons:
            self._polygons.pop()
        self._redraw()

    def _clear_all(self) -> None:
        self._current = []
        self._polygons = []
        self._redraw()

    def _redraw(self) -> None:
        preview_polys = list(self._polygons)
        if len(self._current) >= 3:
            preview_polys.append(list(self._current))
        vis = _render_face_preview_with_mask_overlay(self._image, preview_polys)
        if self._current:
            draw = ImageDraw.Draw(vis)
            pts = [(x * self._width, y * self._height) for x, y in self._current]
            for px, py in pts:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 220, 0))
            if len(pts) >= 2:
                draw.line(pts, fill=(255, 220, 0), width=2)
        self._photo = PhotoImage(vis)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _save(self) -> None:
        if len(self._current) >= 3:
            self._polygons.append(list(self._current))
            self._current = []
        self._on_save(self._polygons)
        self.destroy()


# --- GUI ---------------------------------------------------------------------


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("360 Cubemap Stills Extractor")
        self.minsize(640, 520)
        self.geometry("760x720")

        self._worker: threading.Thread | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._progress_queue: queue.Queue[tuple[int, int, str]] = queue.Queue()
        self._preview_queue: queue.Queue[tuple[int, dict[str, Image.Image], Image.Image | None]] = queue.Queue()
        self._poll_id: str | None = None
        self._preview_poll_id: str | None = None
        self._preview_gen = 0
        self._applying_face_preset = False

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.prefix = tk.StringVar(value="stills")
        self.interval_sec = tk.DoubleVar(value=5.0)
        self.max_width = tk.IntVar(value=0)
        self.face_set = tk.StringVar(value="all")
        self._face_vars: dict[str, tk.BooleanVar] = {
            face: tk.BooleanVar(value=True) for face in CUBEMAP_FACE_ORDER
        }
        self._preview_photos: dict[str, PhotoImage] = {}
        self._preview_thumb_labels: dict[str, ttk.Label] = {}
        self._face_preview_images: dict[str, Image.Image] = {}
        self._face_masks: FaceMaskMap = {face: [] for face in CUBEMAP_FACE_ORDER}
        self._equirect_preview_image: Image.Image | None = None
        self._equirect_mask_polygons: FaceMaskPolygons = []
        self._face_equirect_exclude_previews: dict[str, Image.Image] = {}
        self._mask_preview_time_loaded = -1.0
        self._preview_video_path = ""
        self.mask_preview_sec = tk.DoubleVar(value=0.0)
        self.training_priority = tk.StringVar(value="balanced")
        self.estimate_text = tk.StringVar(value="Select a video to see sample estimate.")
        self.training_recommendation = tk.StringVar(
            value="Select a video to see Splatter training splat estimates."
        )

        self._build_ui()
        self._bind_events()
        self._refresh_user_mask_combo()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        scroll_container = ttk.Frame(outer)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = ttk.Frame(self._scroll_canvas)
        self._scroll_window = self._scroll_canvas.create_window((0, 0), window=frm, anchor=tk.NW)

        def _on_frm_configure(_event: tk.Event | None = None) -> None:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

        frm.bind("<Configure>", _on_frm_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event: tk.Event) -> None:
            if event.delta:
                self._scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

        self.bind_all("<MouseWheel>", _on_mousewheel)

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.BOTH, expand=False, pady=(8, 0))

        ttk.Label(frm, text="360 equirect video to cubemap PNG stills per time sample").pack(
            anchor=tk.W, pady=(0, 8)
        )

        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Video file", width=14).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.video_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row1, text="Browse...", command=self._browse_video).pack(side=tk.LEFT)

        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="Output folder", width=14).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row2, text="Browse...", command=self._browse_output).pack(side=tk.LEFT)

        row3 = ttk.Frame(frm)
        row3.pack(fill=tk.X, **pad)
        ttk.Label(row3, text="File prefix", width=14).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.prefix, width=24).pack(side=tk.LEFT)
        ttk.Label(row3, text="  e.g. lake -> lake-000001_f.png").pack(side=tk.LEFT)

        row4 = ttk.Frame(frm)
        row4.pack(fill=tk.X, **pad)
        ttk.Label(row4, text="Sample every (s)", width=14).pack(side=tk.LEFT)
        ttk.Spinbox(
            row4,
            textvariable=self.interval_sec,
            from_=0.1,
            to=3600.0,
            increment=0.5,
            width=8,
        ).pack(side=tk.LEFT)
        ttk.Label(row4, text="  seconds between time samples").pack(side=tk.LEFT, padx=(8, 0))

        row5 = ttk.Frame(frm)
        row5.pack(fill=tk.X, **pad)
        ttk.Label(row5, text="Face max width", width=14).pack(side=tk.LEFT)
        ttk.Spinbox(row5, textvariable=self.max_width, from_=0, to=8192, increment=256, width=8).pack(
            side=tk.LEFT
        )
        ttk.Label(row5, text="  0 = default 1920 px per face").pack(side=tk.LEFT, padx=(8, 0))

        row6 = ttk.Frame(frm)
        row6.pack(fill=tk.X, **pad)
        ttk.Label(row6, text="Cubemap faces", width=14).pack(side=tk.LEFT, anchor=tk.N)
        face_col = ttk.Frame(row6)
        face_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        preset_row = ttk.Frame(face_col)
        preset_row.pack(fill=tk.X)
        ttk.Label(preset_row, text="Quick presets:").pack(side=tk.LEFT)
        for key, short_label in (
            ("all", "All"),
            ("horizontal", "Horizontal"),
            ("horizontal_down", "Horiz. + down"),
        ):
            ttk.Button(
                preset_row,
                text=short_label,
                width=12,
                command=lambda k=key: self._apply_face_preset(k),
            ).pack(side=tk.LEFT, padx=(6, 0))

        preview_frm = ttk.LabelFrame(face_col, text="Preview — check faces to extract")
        preview_frm.pack(fill=tk.X, pady=(6, 0))
        preview_time_row = ttk.Frame(preview_frm)
        preview_time_row.pack(fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(preview_time_row, text="Mask reference time (s)").pack(side=tk.LEFT)
        ttk.Spinbox(
            preview_time_row,
            textvariable=self.mask_preview_sec,
            from_=0.0,
            to=86400.0,
            increment=0.5,
            width=8,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            preview_time_row,
            text="Refresh previews",
            command=self._refresh_mask_previews,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(
            preview_time_row,
            text="Masks use this frame; changing time clears masks.",
            foreground="#666666",
        ).pack(side=tk.LEFT, padx=(6, 0))
        self._preview_status = ttk.Label(
            preview_frm,
            text="Select a video to load direction previews.",
            foreground="#666666",
        )
        self._preview_status.pack(anchor=tk.W, padx=6, pady=(4, 2))

        grid = ttk.Frame(preview_frm)
        grid.pack(padx=6, pady=(0, 6))
        preview_order = ("f", "r", "b", "l", "u", "d")
        for idx, face in enumerate(preview_order):
            cell = ttk.Frame(grid)
            cell.grid(row=idx // 3, column=idx % 3, padx=6, pady=4)
            thumb = ttk.Label(
                cell,
                text="—",
                width=FACE_PREVIEW_THUMB_SIZE // 8,
                anchor=tk.CENTER,
            )
            thumb.pack()
            self._preview_thumb_labels[face] = thumb
            label = FACE_LABELS.get(face, face)
            ttk.Checkbutton(
                cell,
                text=f"{label} ({face})",
                variable=self._face_vars[face],
                command=self._on_face_toggle,
            ).pack()
            ttk.Button(
                cell,
                text="Mask…",
                width=8,
                command=lambda f=face: self._open_mask_editor(f),
            ).pack(pady=(2, 0))

        self._mask_toggle_btn = ttk.Button(
            face_col,
            text="▼ Masking — exclude regions",
            command=self._toggle_mask_section,
        )
        self._mask_toggle_btn.pack(anchor=tk.W, pady=(8, 0))

        self._mask_body = ttk.Frame(face_col)
        self._mask_body.pack(fill=tk.X, pady=(4, 0))

        eq_row = ttk.Frame(self._mask_body)
        eq_row.pack(fill=tk.X, padx=2, pady=(2, 2))
        ttk.Button(eq_row, text="Edit equirect mask…", command=self._open_equirect_mask_editor).pack(
            side=tk.LEFT
        )
        ttk.Button(eq_row, text="Clear equirect mask", command=self._clear_equirect_mask).pack(
            side=tk.LEFT, padx=6
        )
        mask_row = ttk.Frame(self._mask_body)
        mask_row.pack(fill=tk.X, padx=2, pady=4)
        ttk.Label(mask_row, text="Edit face").pack(side=tk.LEFT)
        self._mask_face_combo = ttk.Combobox(
            mask_row,
            values=[f"{FACE_LABELS[f]} ({f})" for f in CUBEMAP_FACE_ORDER],
            state="readonly",
            width=16,
        )
        self._mask_face_combo.set(f"{FACE_LABELS['f']} (f)")
        self._mask_face_combo.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(mask_row, text="Edit mask…", command=self._open_mask_editor).pack(side=tk.LEFT, padx=6)
        ttk.Button(mask_row, text="Clear face", command=self._clear_face_mask).pack(side=tk.LEFT)
        ttk.Button(mask_row, text="Clear all masks", command=self._clear_all_masks).pack(side=tk.LEFT, padx=6)

        library_row = ttk.Frame(self._mask_body)
        library_row.pack(fill=tk.X, padx=2, pady=(6, 2))
        ttk.Label(library_row, text="Saved masks").pack(side=tk.LEFT)
        self._user_mask_combo = ttk.Combobox(library_row, state="readonly", width=24)
        self._user_mask_combo.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(library_row, text="Load", command=self._load_user_mask).pack(side=tk.LEFT, padx=4)
        ttk.Button(library_row, text="Save current…", command=self._save_user_mask).pack(side=tk.LEFT)
        ttk.Button(library_row, text="Delete", command=self._delete_user_mask).pack(side=tk.LEFT, padx=4)

        ttk.Label(
            self._mask_body,
            text=(
                "Reusable masks live in usermasks/ (polygon JSON, not PNG). "
                "Per-video copy still saved beside source on extract."
            ),
            foreground="#666666",
            wraplength=520,
        ).pack(anchor=tk.W, padx=2, pady=(0, 2))
        ttk.Label(
            self._mask_body,
            text=(
                "Excluded regions are filled black on stills only — no separate mask image files."
            ),
            foreground="#666666",
            wraplength=520,
        ).pack(anchor=tk.W, padx=2, pady=(0, 2))

        ttk.Label(
            face_col,
            text=(
                "Uncheck a direction to skip it in the extract (e.g. handheld operator behind the camera). "
                "Horizontal + down is typical for outdoor drone scans."
            ),
            foreground="#666666",
            wraplength=520,
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Label(frm, textvariable=self.estimate_text, wraplength=680).pack(
            anchor=tk.W, padx=10, pady=(8, 4)
        )

        train_frm = ttk.LabelFrame(frm, text="Splatter training estimate (after COLMAP in main app)")
        train_frm.pack(fill=tk.X, padx=10, pady=(4, 4))
        prio_row = ttk.Frame(train_frm)
        prio_row.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(prio_row, text="Priority").pack(side=tk.LEFT)
        for value, label in (("speed", "Speed"), ("balanced", "Balanced"), ("quality", "Quality")):
            ttk.Radiobutton(
                prio_row,
                text=label,
                value=value,
                variable=self.training_priority,
                command=self._refresh_training_recommendation,
            ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(
            train_frm,
            textvariable=self.training_recommendation,
            wraplength=660,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(0, 8))

        self.warn_label = ttk.Label(frm, text="", foreground="#c07000", wraplength=680)
        self.warn_label.pack(anchor=tk.W, padx=10, pady=(0, 4))

        self.progress_text = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.progress_text).pack(anchor=tk.W, pady=(0, 2))

        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 6))

        btn_row = ttk.Frame(footer)
        btn_row.pack(fill=tk.X)
        self.extract_btn = ttk.Button(btn_row, text="Extract cubemap stills", command=self._start_extract)
        self.extract_btn.pack(side=tk.LEFT)

        self._log_toggle_btn = ttk.Button(
            footer,
            text="▶ Log",
            command=self._toggle_log_section,
        )
        self._log_toggle_btn.pack(anchor=tk.W, pady=(8, 0))

        self._log_body = ttk.Frame(footer)
        self.log = scrolledtext.ScrolledText(self._log_body, height=8, state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _toggle_log_section(self, expand: bool | None = None) -> None:
        if expand is None:
            expand = not self._log_body.winfo_ismapped()
        if expand:
            self._log_body.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
            self._log_toggle_btn.configure(text="▼ Log")
        else:
            self._log_body.pack_forget()
            self._log_toggle_btn.configure(text="▶ Log")

    def _toggle_mask_section(self) -> None:
        if self._mask_body.winfo_ismapped():
            self._mask_body.pack_forget()
            self._mask_toggle_btn.configure(text="▶ Masking — exclude regions")
        else:
            self._mask_body.pack(fill=tk.X, pady=(4, 0))
            self._mask_toggle_btn.configure(text="▼ Masking — exclude regions")
        if hasattr(self, "_scroll_canvas"):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _bind_events(self) -> None:
        self.video_path.trace_add("write", lambda *_: self._refresh_probe())
        self.interval_sec.trace_add("write", lambda *_: self._refresh_estimate())
        self.prefix.trace_add("write", lambda *_: self._refresh_estimate())
        self.max_width.trace_add("write", lambda *_: self._refresh_estimate())
        self.output_dir.trace_add("write", lambda *_: self._refresh_estimate())
        self.training_priority.trace_add("write", lambda *_: self._refresh_training_recommendation())

    def _get_selected_faces(self) -> frozenset[str]:
        return frozenset(face for face, var in self._face_vars.items() if var.get())

    def _apply_face_preset(self, preset_key: str) -> None:
        if preset_key not in FACE_SET_PRESETS:
            preset_key = "all"
        self.face_set.set(preset_key)
        allowed = _faces_in_set(preset_key)
        self._applying_face_preset = True
        for face, var in self._face_vars.items():
            var.set(face in allowed)
        self._applying_face_preset = False
        self._refresh_estimate()

    def _on_face_toggle(self) -> None:
        if self._applying_face_preset:
            return
        selected = self._get_selected_faces()
        for key in FACE_SET_PRESETS:
            if selected == _faces_in_set(key):
                self.face_set.set(key)
                break
        self._refresh_estimate()

    def _face_from_combo_label(self, text: str) -> str:
        if ")" in text and "(" in text:
            return text.rsplit("(", 1)[1].rstrip(")").lower()
        return "f"

    def _open_mask_editor(self, face: str | None = None) -> None:
        if face is None:
            face = self._face_from_combo_label(self._mask_face_combo.get())
        if face not in self._face_preview_images:
            messagebox.showwarning(
                "Mask",
                "Load face previews first (select a video and wait for previews).",
            )
            return

        def on_save(polygons: FaceMaskPolygons) -> None:
            self._face_masks[face] = polygons
            self._update_face_thumb(face)

        FaceMaskEditor(
            self,
            face,
            self._face_preview_images[face],
            self._face_masks.get(face, []),
            on_save,
        )

    def _clear_face_mask(self) -> None:
        face = self._face_from_combo_label(self._mask_face_combo.get())
        self._face_masks[face] = []
        self._update_face_thumb(face)

    def _clear_equirect_mask(self) -> None:
        self._equirect_mask_polygons = []
        self._face_equirect_exclude_previews.clear()
        for face in self._face_preview_images:
            self._update_face_thumb(face)

    def _clear_all_masks(self) -> None:
        for face in CUBEMAP_FACE_ORDER:
            self._face_masks[face] = []
        self._equirect_mask_polygons = []
        self._face_equirect_exclude_previews.clear()
        for face in self._face_preview_images:
            self._update_face_thumb(face)

    def _rebuild_equirect_face_excludes(self) -> None:
        self._face_equirect_exclude_previews.clear()
        if not self._equirect_mask_polygons:
            return
        templates = _build_equirect_exclude_templates(
            self._equirect_mask_polygons,
            FACE_MASK_EDITOR_SIZE,
        )
        self._face_equirect_exclude_previews = templates

    def _open_equirect_mask_editor(self) -> None:
        if self._equirect_preview_image is None:
            messagebox.showwarning(
                "Mask",
                "Load previews first (select a video and wait for previews).",
            )
            return

        def on_save(polygons: FaceMaskPolygons) -> None:
            self._equirect_mask_polygons = polygons
            self._rebuild_equirect_face_excludes()
            for face in self._face_preview_images:
                self._update_face_thumb(face)

        EquirectMaskEditor(
            self,
            self._equirect_preview_image,
            self._equirect_mask_polygons,
            on_save,
        )

    def _update_face_thumb(self, face: str) -> None:
        label = self._preview_thumb_labels.get(face)
        source = self._face_preview_images.get(face)
        if label is None or source is None:
            return
        thumb = source.resize(
            (FACE_PREVIEW_THUMB_SIZE, FACE_PREVIEW_THUMB_SIZE),
            Image.Resampling.LANCZOS,
        )
        eq_exclude = self._face_equirect_exclude_previews.get(face)
        if eq_exclude is not None:
            eq_exclude = eq_exclude.resize(
                (FACE_PREVIEW_THUMB_SIZE, FACE_PREVIEW_THUMB_SIZE),
                Image.Resampling.NEAREST,
            )
        vis = _render_face_preview_with_mask_overlay(
            thumb,
            self._face_masks.get(face, []),
            eq_exclude,
        )
        photo = PhotoImage(vis)
        self._preview_photos[face] = photo
        label.configure(image=photo, text="")

    def _apply_loaded_masks(
        self,
        face_masks: FaceMaskMap,
        equirect_polygons: FaceMaskPolygons,
    ) -> None:
        self._face_masks = face_masks
        self._equirect_mask_polygons = equirect_polygons
        self._rebuild_equirect_face_excludes()
        for face in self._face_preview_images:
            self._update_face_thumb(face)

    def _refresh_user_mask_combo(self, select: str | None = None) -> None:
        names = _list_user_mask_names()
        self._user_mask_combo["values"] = names
        if select and select in names:
            self._user_mask_combo.set(select)
        elif names:
            self._user_mask_combo.set(names[0])
        else:
            self._user_mask_combo.set("")

    def _load_user_mask(self) -> None:
        name = self._user_mask_combo.get().strip()
        if not name:
            messagebox.showwarning("Saved masks", "No saved mask selected. Save one first or pick from the list.")
            return
        loaded = _load_masks_json(_user_mask_path(name))
        if loaded is None:
            messagebox.showerror("Saved masks", f"Could not read mask '{name}'.")
            return
        face_masks, equirect, _ = loaded
        if not _masks_active(face_masks, equirect):
            messagebox.showwarning("Saved masks", f"Mask '{name}' is empty.")
            return
        if not self._face_preview_images:
            messagebox.showwarning(
                "Saved masks",
                "Load video previews first so mask overlays can be shown.",
            )
        self._apply_loaded_masks(face_masks, equirect)
        self._preview_status.configure(text=f"Loaded saved mask '{name}'.")

    def _save_user_mask(self) -> None:
        if not _masks_active(self._face_masks, self._equirect_mask_polygons):
            messagebox.showwarning(
                "Save mask",
                "Draw a mask first (equirect or per-face), then save to the library.",
            )
            return
        raw_name = simpledialog.askstring(
            "Save mask to library",
            "Name for this mask (e.g. drone_props):",
            parent=self,
        )
        if not raw_name:
            return
        safe_name = _sanitize_user_mask_name(raw_name)
        if safe_name != raw_name.strip():
            messagebox.showinfo(
                "Save mask",
                f"Saved as '{safe_name}' (sanitized file name).",
            )
        existing = _user_mask_path(safe_name)
        if existing.is_file():
            ok = messagebox.askyesno(
                "Overwrite mask",
                f"Mask '{safe_name}' already exists. Replace it?",
            )
            if not ok:
                return
        description = simpledialog.askstring(
            "Save mask to library",
            "Short description (optional, e.g. DJI props on up/down):",
            parent=self,
        )
        path = _save_user_mask_library(
            safe_name,
            copy.deepcopy(self._face_masks),
            copy.deepcopy(self._equirect_mask_polygons),
            self._mask_preview_time_loaded if self._mask_preview_time_loaded >= 0 else 0.0,
            description=(description or "").strip() or None,
        )
        self._refresh_user_mask_combo(select=safe_name)
        messagebox.showinfo("Save mask", f"Saved to {path}")

    def _delete_user_mask(self) -> None:
        name = self._user_mask_combo.get().strip()
        if not name:
            messagebox.showwarning("Saved masks", "No saved mask selected.")
            return
        if not messagebox.askyesno("Delete mask", f"Delete saved mask '{name}' permanently?"):
            return
        path = _user_mask_path(name)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            messagebox.showerror("Delete mask", f"Could not delete: {exc}")
            return
        self._refresh_user_mask_combo()

    def _try_load_masks_from_sidecar(self, video: Path) -> None:
        loaded = _load_masks_json(_masks_json_path(video))
        if loaded is None:
            return
        face_masks, equirect, preview_time = loaded
        if not _masks_active(face_masks, equirect):
            return
        self._apply_loaded_masks(face_masks, equirect)
        if abs(preview_time - self._mask_preview_time_loaded) > 0.01:
            self._preview_status.configure(
                text=(
                    f"{self._preview_status.get()} "
                    f"(loaded saved masks from t={preview_time:g}s)."
                )
            )

    def _clear_face_previews(self) -> None:
        self._face_preview_images.clear()
        self._equirect_preview_image = None
        self._face_equirect_exclude_previews.clear()
        for face in CUBEMAP_FACE_ORDER:
            self._preview_photos.pop(face, None)
            label = self._preview_thumb_labels.get(face)
            if label is not None:
                label.configure(image="", text="—")

    def _mask_preview_time_sec(self) -> float:
        try:
            return max(0.0, float(self.mask_preview_sec.get()))
        except tk.TclError:
            return 0.0

    def _refresh_mask_previews(self) -> None:
        path = Path(self.video_path.get().strip()).expanduser()
        if not path.is_file():
            messagebox.showwarning("Video", "Select a valid video file first.")
            return
        time_sec = self._mask_preview_time_sec()
        if time_sec != self._mask_preview_time_loaded:
            self._face_masks = {face: [] for face in CUBEMAP_FACE_ORDER}
            self._equirect_mask_polygons = []
            self._face_equirect_exclude_previews.clear()
        self._mask_preview_time_loaded = time_sec
        self._start_face_preview_worker(path, time_sec)

    def _start_face_preview_worker(self, path: Path, time_sec: float = 0.0) -> None:
        if shutil.which("ffmpeg") is None:
            self._clear_face_previews()
            self._preview_status.configure(text="ffmpeg not found — direction previews unavailable.")
            return
        path_key = str(path.resolve())
        if path_key != self._preview_video_path:
            self._face_masks = {face: [] for face in CUBEMAP_FACE_ORDER}
            self._equirect_mask_polygons = []
            self._face_equirect_exclude_previews.clear()
            self._mask_preview_time_loaded = -1.0
        self._preview_video_path = path_key
        self._preview_gen += 1
        gen = self._preview_gen
        self._clear_face_previews()
        self._preview_status.configure(
            text=f"Generating previews at t={time_sec:g}s..."
        )
        threading.Thread(
            target=self._face_preview_worker,
            args=(gen, path, time_sec),
            daemon=True,
        ).start()
        if self._preview_poll_id is None:
            self._preview_poll_id = self.after(100, self._poll_face_previews)

    def _face_preview_worker(self, gen: int, path: Path, time_sec: float) -> None:
        try:
            previews = _generate_face_previews(path, time_sec=time_sec)
            equirect = _generate_equirect_preview(path, time_sec=time_sec)
        except Exception:
            previews = {}
            equirect = None
        self._preview_queue.put((gen, previews, equirect))

    def _poll_face_previews(self) -> None:
        self._preview_poll_id = None
        applied = False
        while True:
            try:
                gen, previews, equirect = self._preview_queue.get_nowait()
            except queue.Empty:
                break
            if gen != self._preview_gen:
                continue
            self._apply_face_previews(previews, equirect)
            applied = True
        if not applied:
            self._preview_poll_id = self.after(100, self._poll_face_previews)

    def _apply_face_previews(
        self,
        previews: dict[str, Image.Image],
        equirect: Image.Image | None,
    ) -> None:
        self._face_preview_images = dict(previews)
        self._equirect_preview_image = equirect
        if self._equirect_mask_polygons:
            self._rebuild_equirect_face_excludes()
        missing = 0
        for face in CUBEMAP_FACE_ORDER:
            if face not in previews:
                missing += 1
                label = self._preview_thumb_labels.get(face)
                if label is not None:
                    label.configure(image="", text="?")
        for face in previews:
            self._update_face_thumb(face)
        video = Path(self.video_path.get().strip()).expanduser()
        if video.is_file():
            self._try_load_masks_from_sidecar(video)
        time_sec = self._mask_preview_time_loaded
        if previews:
            note = f"Preview at t={time_sec:g}s ({len(previews)} face(s))."
            if equirect is None:
                note += " Equirect preview failed."
            if missing:
                note += f" {missing} face preview(s) failed."
            self._preview_status.configure(text=note)
        else:
            self._preview_status.configure(text="Could not generate face previews from this video.")

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select 360 video",
            filetypes=[
                ("Video", "*.mp4 *.mov *.mkv *.webm *.avi"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)
            stem = Path(path).stem
            if self.prefix.get().strip() in ("", "stills"):
                self.prefix.set(_sanitize_prefix(stem, stem))

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _flush_log_queue(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, line)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

    def _flush_progress_queue(self) -> None:
        while True:
            try:
                current, total, message = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self.progress.configure(maximum=max(total, 1))
            self.progress["value"] = min(current, total)
            self.progress_text.set(message)

    def _refresh_probe(self) -> None:
        path = Path(self.video_path.get().strip()).expanduser()
        if not path.is_file():
            self.warn_label.configure(text="")
            self.estimate_text.set("Select a video to see sample estimate.")
            self.training_recommendation.set("Select a video to see Splatter training splat estimates.")
            self._clear_face_previews()
            self._face_masks = {face: [] for face in CUBEMAP_FACE_ORDER}
            self._equirect_mask_polygons = []
            self._mask_preview_time_loaded = -1.0
            self._preview_status.configure(text="Select a video to load direction previews.")
            return
        if shutil.which("ffprobe") is None:
            self.warn_label.configure(text="ffprobe not found in PATH.")
            return
        try:
            info = _probe_video(path)
        except Exception as exc:
            self.warn_label.configure(text=f"Could not read video: {exc}")
            return

        w, h, dur = info["width"], info["height"], info["duration"]
        sar = info.get("sample_aspect_ratio") or "1:1"
        if info["looks_equirect"]:
            sar_note = ""
            if sar not in ("1:1", "1/1", "N/A", ""):
                sar_note = f" SAR {sar} (will be normalized)."
            self.warn_label.configure(
                text=(
                    f"Video looks like stitched equirectangular "
                    f"({w}x{h}, {dur:.1f}s).{sar_note}"
                )
            )
        else:
            self.warn_label.configure(
                text=(
                    f"Warning: {w}x{h} (aspect {info['aspect']:.2f}) does not look like standard "
                    f"2:1 equirect 360. Export may be wrong projection - cubemap faces will still "
                    f"be generated but may not align for photogrammetry."
                )
            )
        self._refresh_estimate(info)
        time_sec = self._mask_preview_time_sec()
        if self._mask_preview_time_loaded < 0:
            self._mask_preview_time_loaded = time_sec
        self._start_face_preview_worker(path, time_sec)

    def _refresh_estimate(self, info: dict | None = None) -> None:
        if info is None:
            path = Path(self.video_path.get().strip()).expanduser()
            if not path.is_file() or shutil.which("ffprobe") is None:
                return
            try:
                info = _probe_video(path)
            except Exception:
                return
        try:
            interval = float(self.interval_sec.get())
        except tk.TclError:
            interval = 0.0
        samples = _estimate_sample_count(float(info["duration"]), interval)
        selected_faces = self._get_selected_faces()
        faces = _face_count_for_faces(selected_faces)
        face_desc = _selected_faces_label(selected_faces)
        prefix = _sanitize_prefix(self.prefix.get(), Path(self.video_path.get()).stem)
        total = samples * faces
        tile_w = _effective_max_width(int(self.max_width.get() or 0))
        est_bytes = _estimate_output_bytes(total, int(self.max_width.get() or 0))
        disk_note = ""
        out_raw = self.output_dir.get().strip()
        if out_raw:
            free = _disk_free_bytes(Path(out_raw))
            if free >= 0:
                need = int(est_bytes * _DISK_HEADROOM_FACTOR)
                disk_note = f" - est. ~{_format_bytes(est_bytes)}, need ~{_format_bytes(need)}, free {_format_bytes(free)}"
        self.estimate_text.set(
            f"~{samples} time sample(s) x {faces} face(s) = ~{total} PNGs "
            f"({prefix}-000001_f.png ...), {tile_w}px per face, {face_desc}.{disk_note}"
        )
        self._refresh_training_recommendation(info)

    def _refresh_training_recommendation(self, info: dict | None = None) -> None:
        if info is None:
            path = Path(self.video_path.get().strip()).expanduser()
            if not path.is_file() or shutil.which("ffprobe") is None:
                return
            try:
                info = _probe_video(path)
            except Exception:
                return
        try:
            interval = float(self.interval_sec.get())
        except tk.TclError:
            interval = 0.0
        if interval <= 0:
            self.training_recommendation.set("Set a valid sample interval to see training estimates.")
            return
        samples = _estimate_sample_count(float(info["duration"]), interval)
        selected_faces = self._get_selected_faces()
        total = samples * _face_count_for_faces(selected_faces)
        priority_raw = self.training_priority.get()
        priority = priority_raw if priority_raw in ("speed", "balanced", "quality") else "balanced"
        _, text = plan_from_planned_stills(
            total,
            int(self.max_width.get() or 0),
            priority,
        )
        self.training_recommendation.set(text)

    def _start_extract(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("Busy", "Extraction is already running.")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror("Missing ffmpeg", "ffmpeg was not found in PATH.")
            return

        video = Path(self.video_path.get().strip()).expanduser()
        out_dir = Path(self.output_dir.get().strip()).expanduser()
        if not video.is_file():
            messagebox.showerror("Video", "Select a valid video file.")
            return
        if not out_dir:
            messagebox.showerror("Output", "Select an output folder.")
            return

        try:
            interval = float(self.interval_sec.get())
        except tk.TclError:
            messagebox.showerror("Interval", "Sample interval must be a number.")
            return
        if interval <= 0:
            messagebox.showerror("Interval", "Sample interval must be greater than 0.")
            return

        prefix = _sanitize_prefix(self.prefix.get(), video.stem)
        max_w = int(self.max_width.get() or 0)
        selected_faces = self._get_selected_faces()
        if not selected_faces:
            messagebox.showerror("Faces", "Select at least one cubemap face to extract.")
            return

        try:
            info = _probe_video(video)
        except Exception as exc:
            messagebox.showerror("Video", f"Could not probe video:\n{exc}")
            return

        if not info["looks_equirect"]:
            ok = messagebox.askyesno(
                "360 warning",
                "This file does not look like a standard 2:1 equirectangular 360 video.\n\n"
                f"Size: {info['width']}x{info['height']} (aspect {info['aspect']:.2f})\n\n"
                "Cubemap extraction assumes equirect input. Continue anyway?",
                icon=messagebox.WARNING,
            )
            if not ok:
                return

        samples = _estimate_sample_count(info["duration"], interval)
        faces = _face_count_for_faces(selected_faces)
        total_png = samples * faces
        existing = list(out_dir.glob(f"{prefix}-*.png")) if out_dir.exists() else []
        if existing:
            ok = messagebox.askyesno(
                "Existing files",
                f"Found {len(existing)} existing PNG(s) matching '{prefix}-*.png' in the output folder.\n"
                "New files may overwrite same indices. Continue?",
            )
            if not ok:
                return

        est_bytes = _estimate_output_bytes(total_png, max_w)
        disk_ok, disk_msg = _disk_space_check(out_dir, est_bytes)
        if not disk_ok:
            messagebox.showerror("Not enough disk space", disk_msg)
            return
        if est_bytes > 0 and _disk_free_bytes(out_dir) < 0:
            ok = messagebox.askyesno(
                "Disk space unknown",
                f"{disk_msg}\n\nContinue without a free-space check?",
                icon=messagebox.WARNING,
            )
            if not ok:
                return

        if total_png > 2000:
            ok = messagebox.askyesno(
                "Large output",
                f"This will create approximately {total_png} PNG files.\nContinue?",
            )
            if not ok:
                return

        self.extract_btn.configure(state=tk.DISABLED)
        self._toggle_log_section(expand=True)
        self.progress.configure(mode="determinate", maximum=max(total_png, 1), value=0)
        self.progress_text.set("Starting extraction...")
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

        self._log(f"[INFO] {disk_msg}\n")
        self._log(f"[INFO] Estimated output size: ~{_format_bytes(est_bytes)}\n")
        priority_raw = self.training_priority.get()
        priority = priority_raw if priority_raw in ("speed", "balanced", "quality") else "balanced"
        _, train_hint = plan_from_planned_stills(total_png, max_w, priority)
        self._log(f"[INFO] Splatter training ({priority}): {train_hint.replace(chr(10), ' | ')}\n\n")

        face_masks = copy.deepcopy(self._face_masks)
        equirect_masks = copy.deepcopy(self._equirect_mask_polygons)
        preview_time = self._mask_preview_time_loaded
        if _masks_active(face_masks, equirect_masks):
            parts: list[str] = []
            if equirect_masks:
                parts.append("equirect")
            face_parts = [f for f in CUBEMAP_FACE_ORDER if face_masks.get(f)]
            if face_parts:
                parts.append(f"per-face ({', '.join(face_parts)})")
            self._log(
                f"[INFO] Masks at t={preview_time:g}s: "
                f"{'; '.join(parts)} — applied to all time samples.\n"
            )
            self._log(f"[INFO] Mask definitions → {_mask_sidecar_dir(video) / 'face_masks.json'}\n")

        self._worker = threading.Thread(
            target=self._extract_worker,
            args=(
                video,
                out_dir,
                prefix,
                interval,
                max_w,
                selected_faces,
                total_png,
                face_masks,
                equirect_masks,
                preview_time,
            ),
            daemon=True,
        )
        self._worker.start()
        self._poll_id = self.after(100, self._poll_worker)

    def _poll_worker(self) -> None:
        self._flush_log_queue()
        self._flush_progress_queue()
        if self._worker and self._worker.is_alive():
            self._poll_id = self.after(100, self._poll_worker)
            return
        self._flush_progress_queue()
        self.extract_btn.configure(state=tk.NORMAL)
        self._flush_log_queue()

    def _extract_worker(
        self,
        video: Path,
        out_dir: Path,
        prefix: str,
        interval_sec: float,
        max_width: int,
        selected_faces: frozenset[str],
        total_png: int,
        face_masks: FaceMaskMap,
        equirect_masks: FaceMaskPolygons,
        mask_preview_time_sec: float,
    ) -> None:
        def report(current: int, message: str) -> None:
            self._progress_queue.put((current, total_png, message))

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_fps = 1.0 / interval_sec
            face_jobs = _face_jobs_for_faces(selected_faces)
            if not face_jobs:
                self._log("[ERROR] No cubemap faces selected.\n")
                report(0, "No faces selected.")
                return

            self._log(f"[INFO] Video: {video}\n")
            self._log(f"[INFO] Output: {out_dir}\n")
            self._log(f"[INFO] Prefix: {prefix}\n")
            self._log(f"[INFO] Faces: {_selected_faces_label(selected_faces)}\n")
            self._log(
                f"[INFO] Sample every {interval_sec:g}s (fps={sample_fps:.6f}), "
                f"faces: {', '.join(f[0] for f in face_jobs)}\n"
            )
            self._log(f"[INFO] Face width: {_effective_max_width(max_width)} px\n\n")

            report(0, f"Writing file 0 of {total_png}...")

            frame_index = 1
            n_faces = len(face_jobs)
            samples_per_face: int | None = None
            for face_idx, (face, yaw, pitch) in enumerate(face_jobs, 1):
                suffix = f"_{face}.png"
                pattern = out_dir / f"{prefix}-%06d{suffix}"
                vf = _build_equirect_vf(sample_fps, yaw, pitch, max_width)
                label = FACE_LABELS.get(face, face)
                self._log(f"[INFO] Extracting face {face} ({label}) [{face_idx}/{n_faces}]...\n")
                self._log(
                    f"$ ffmpeg -hide_banner -nostats -loglevel error -y -i {video!r} "
                    f"-vf {vf!r} -pix_fmt rgb24 -an -start_number {frame_index} {pattern.name!r}\n"
                )
                proc = _ffmpeg_extract_popen(video, pattern, vf, frame_index)
                while proc.poll() is None:
                    face_n = _count_face_pngs(out_dir, prefix, face)
                    overall = sum(_count_face_pngs(out_dir, prefix, f) for f, _, _ in face_jobs[:face_idx])
                    report(
                        overall,
                        f"Writing file {overall} of {total_png} "
                        f"(face {face_idx}/{n_faces} {label}: {face_n} frame(s))...",
                    )
                    time.sleep(0.25)
                code = int(proc.wait())
                face_n = _count_face_pngs(out_dir, prefix, face)
                if samples_per_face is None:
                    samples_per_face = face_n
                    actual_total = samples_per_face * n_faces
                    if actual_total != total_png:
                        self._log(
                            f"[INFO] ffmpeg produced {samples_per_face} sample(s) per face "
                            f"(estimated {total_png // n_faces}); total {actual_total} PNG(s).\n"
                        )
                        total_png = actual_total
                elif face_n != samples_per_face:
                    self._log(
                        f"[WARN] Face {face} produced {face_n} frame(s), "
                        f"expected {samples_per_face} (same as first face).\n"
                    )
                if code != 0:
                    self._log(f"[ERROR] ffmpeg failed for face {face} (exit {code}).\n")
                    report(_count_prefix_pngs(out_dir, prefix), f"Failed on face {face}.")
                    return
                overall = sum(_count_face_pngs(out_dir, prefix, f) for f, _, _ in face_jobs[:face_idx])
                report(
                    overall,
                    f"Writing file {overall} of {total_png} "
                    f"(face {face_idx}/{n_faces} {label}: done, {face_n} frame(s))...",
                )
                self._log(f"[INFO] Face {face} ({label}) complete - {face_n} PNG(s).\n")

            still_pngs = _list_still_pngs(out_dir, prefix)
            if _masks_active(face_masks, equirect_masks):
                self._log(f"\n[INFO] Applying masks ({len(still_pngs)} still(s))...\n")
                report(total_png, f"Applying masks 0 of {len(still_pngs)}...")

                def mask_progress(done: int, mask_total: int) -> None:
                    report(total_png, f"Applying masks {done} of {mask_total}...")

                mask_ok, mask_fail = _apply_face_masks_batch(
                    still_pngs,
                    face_masks,
                    equirect_masks,
                    face_size=max_width,
                    on_progress=mask_progress,
                )
                if mask_fail:
                    self._log(f"[WARN] Mask apply failed for {mask_fail} still(s).\n")
                else:
                    self._log(f"[INFO] Applied masks to {mask_ok} still(s) (excluded regions filled black).\n")
                masks_json = _masks_json_path(video)
                try:
                    _save_masks_json(
                        masks_json,
                        face_masks,
                        equirect_masks,
                        mask_preview_time_sec,
                        video=video,
                        output_prefix=prefix,
                    )
                    self._log(f"[INFO] Saved mask definitions to {masks_json}\n")
                except OSError as exc:
                    self._log(f"[WARN] Could not save mask JSON: {exc}\n")

            pngs = list(still_pngs)
            self._log(f"\n[INFO] Stripping PNG metadata ({len(pngs)} file(s))...\n")
            report(
                _count_prefix_pngs(out_dir, prefix),
                f"Stripping metadata 0 of {len(pngs)}...",
            )

            def strip_progress(done: int, strip_total: int) -> None:
                report(
                    total_png,
                    f"Stripping metadata {done} of {strip_total}...",
                )

            ok, fail = _strip_png_metadata(pngs, on_progress=strip_progress)
            if fail:
                self._log(f"[WARN] Metadata strip failed for {fail} file(s).\n")
            else:
                self._log(
                    f"[INFO] Stripped metadata from {ok} PNG(s) "
                    "(Postshot / gamma-squared compatible).\n"
                )
            report(total_png, f"Done - {len(pngs)} file(s) written.")
            self._log(f"\n[DONE] {len(pngs)} still(s) in {out_dir}\n")
        except Exception as exc:
            self._log(f"[ERROR] {exc}\n")
            report(_count_prefix_pngs(out_dir, prefix) if out_dir.exists() else 0, f"Error: {exc}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
