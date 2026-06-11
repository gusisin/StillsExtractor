#!/usr/bin/env python3
"""Standalone Windows-friendly GUI: 360 equirect video to cubemap PNG stills per sample.

Extracts pinhole cubemap faces (f/r/b/l/u/d) for use with Splatter, Postshot, COLMAP, etc.
No COLMAP or training - image prep only.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow") from None

from training_recommendations import plan_from_planned_stills

# --- Cubemap (matches Splatter / ffmpeg v360 conventions) -------------------

EQUIRECT_FACE_FOV_DEG = 90
EQUIRECT_DEFAULT_TILE_MAX_WIDTH = 1920

EQUIRECT_CUBE_FACES: tuple[tuple[str, int, int], ...] = (
    ("f", 0, 0),
    ("r", 90, 0),
    ("b", 180, 0),
    ("l", -90, 0),  # yaw=270 fails in ffmpeg v360
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


def _face_set_label(face_set: str) -> str:
    entry = FACE_SET_PRESETS.get(face_set)
    return entry[0] if entry else FACE_SET_PRESETS["all"][0]


def _faces_in_set(face_set: str) -> frozenset[str]:
    entry = FACE_SET_PRESETS.get(face_set)
    return entry[1] if entry else FACE_SET_PRESETS["all"][1]


def _face_jobs_for_set(face_set: str) -> list[tuple[str, int, int]]:
    allowed = _faces_in_set(face_set)
    return [job for job in EQUIRECT_CUBE_FACES if job[0] in allowed]


def _face_count_for_set(face_set: str) -> int:
    return len(_face_jobs_for_set(face_set))


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


def _build_equirect_vf(sample_fps: float, yaw: int, pitch: int, max_width: int) -> str:
    tile_w = _effective_max_width(max_width)
    yaw = _normalize_v360_angle(yaw)
    pitch = _normalize_v360_angle(pitch)
    h_fov = EQUIRECT_FACE_FOV_DEG
    return (
        f"{_equirect_input_normalize_vf()},"
        f"fps={sample_fps:.6f},"
        f"v360=input=equirect:output=rectilinear:yaw={yaw}:pitch={pitch}:roll=0:"
        f"ih_fov=360:iv_fov=180:h_fov={h_fov}:v_fov={h_fov}:w={tile_w}:h={tile_w},"
        f"{_equirect_square_face_scale_vf(tile_w)}"
    )


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


# --- GUI ---------------------------------------------------------------------


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("360 Cubemap Stills Extractor")
        self.minsize(640, 580)
        self.geometry("720x680")

        self._worker: threading.Thread | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._progress_queue: queue.Queue[tuple[int, int, str]] = queue.Queue()
        self._poll_id: str | None = None

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.prefix = tk.StringVar(value="stills")
        self.interval_sec = tk.DoubleVar(value=5.0)
        self.max_width = tk.IntVar(value=0)
        self.face_set = tk.StringVar(value="all")
        self.training_priority = tk.StringVar(value="balanced")
        self.estimate_text = tk.StringVar(value="Select a video to see sample estimate.")
        self.training_recommendation = tk.StringVar(
            value="Select a video to see Splatter training splat estimates."
        )

        self._build_ui()
        self._bind_events()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

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
        for key in ("all", "horizontal", "horizontal_down"):
            ttk.Radiobutton(
                face_col,
                text=FACE_SET_PRESETS[key][0],
                value=key,
                variable=self.face_set,
                command=self._refresh_estimate,
            ).pack(anchor=tk.W)
        ttk.Label(
            face_col,
            text="Horizontal + down is useful for drone footage (ground fills gaps in the horizon ring).",
            foreground="#666666",
            wraplength=520,
        ).pack(anchor=tk.W, pady=(2, 0))

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
        ttk.Label(frm, textvariable=self.progress_text).pack(anchor=tk.W, padx=10, pady=(4, 0))

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=8)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        self.extract_btn = ttk.Button(btn_row, text="Extract cubemap stills", command=self._start_extract)
        self.extract_btn.pack(side=tk.LEFT)

        ttk.Label(frm, text="Log").pack(anchor=tk.W, padx=10, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(frm, height=12, state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    def _bind_events(self) -> None:
        self.video_path.trace_add("write", lambda *_: self._refresh_probe())
        self.interval_sec.trace_add("write", lambda *_: self._refresh_estimate())
        self.face_set.trace_add("write", lambda *_: self._refresh_estimate())
        self.prefix.trace_add("write", lambda *_: self._refresh_estimate())
        self.max_width.trace_add("write", lambda *_: self._refresh_estimate())
        self.output_dir.trace_add("write", lambda *_: self._refresh_estimate())
        self.training_priority.trace_add("write", lambda *_: self._refresh_training_recommendation())

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
        face_set = self.face_set.get()
        if face_set not in FACE_SET_PRESETS:
            face_set = "all"
        faces = _face_count_for_set(face_set)
        face_desc = _face_set_label(face_set)
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
        face_set = self.face_set.get()
        if face_set not in FACE_SET_PRESETS:
            face_set = "all"
        total = samples * _face_count_for_set(face_set)
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
        face_set = self.face_set.get()
        if face_set not in FACE_SET_PRESETS:
            face_set = "all"

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
        faces = _face_count_for_set(face_set)
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

        self._worker = threading.Thread(
            target=self._extract_worker,
            args=(video, out_dir, prefix, interval, max_w, face_set, total_png),
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
        face_set: str,
        total_png: int,
    ) -> None:
        def report(current: int, message: str) -> None:
            self._progress_queue.put((current, total_png, message))

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_fps = 1.0 / interval_sec
            if face_set not in FACE_SET_PRESETS:
                face_set = "all"
            face_jobs = _face_jobs_for_set(face_set)

            self._log(f"[INFO] Video: {video}\n")
            self._log(f"[INFO] Output: {out_dir}\n")
            self._log(f"[INFO] Prefix: {prefix}\n")
            self._log(f"[INFO] Face set: {_face_set_label(face_set)}\n")
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

            pngs = sorted(out_dir.glob(f"{prefix}-*.png"))
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
