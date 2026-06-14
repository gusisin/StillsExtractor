# 360 Cubemap Stills Extractor

Standalone Windows-friendly tool that converts **360° equirectangular video** into **cubemap face PNGs** for Splatter, Postshot, COLMAP, and similar photogrammetry workflows.

Part of the [Grenade FPV](https://www.grenadefpv.com/) creator toolkit. Extracted from [Splatter](https://github.com/gusisin/Splatter) as a focused, dependency-light project.

## What it does

- Samples an equirect MP4/MOV at a fixed interval (e.g. every 2 seconds)
- Projects each sample into cubemap faces: **front, right, back, left, down** (optional **up**)
- Writes `{prefix}-{frame}_{face}.png` (e.g. `house-000001_f.png`)
- Outputs **square** pinhole faces (default 1920×1920) with correct 90°×90° FOV via ffmpeg `v360`
- Normalizes non–2:1 / SAR inputs at extract time (pad to 2:1 equirect before projection)
- **Face previews** from a reference frame — check/uncheck directions to skip whole faces (e.g. handheld operator behind camera)
- **Masking** — draw exclude polygons on the equirect frame or per-face; excluded regions are filled **black** on all time samples
- **Reusable mask library** — save/load masks from `usermasks/` (e.g. drone prop mask for repeat flights)
- Strips PNG metadata after extract (Postshot / gamma compatibility)
- Shows Splatter training hints based on planned still count

The **stills output folder** contains only cubemap PNGs (ready to copy into your training app). Mask definitions are **not** mixed into that folder.

## Requirements

| Tool | Required | Notes |
|------|----------|-------|
| Python 3.10+ | Yes | tkinter included with standard Windows Python |
| [ffmpeg](https://ffmpeg.org/) + ffprobe | Yes | Must be on `PATH` |
| Pillow | Yes | `pip install -r requirements.txt` |
| ImageMagick (`magick`) | No | Faster metadata strip if available |

## Installation

From PowerShell in the project folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
cd d:\StillsExtractor
.\install.ps1
.\.venv\Scripts\Activate.ps1
python MP4_360_stills.py
```

`install.ps1` creates `.venv`, installs Pillow, and checks for ffmpeg/ffprobe (and optional ImageMagick).

**Manual install** (same result):

```powershell
cd d:\StillsExtractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python MP4_360_stills.py
```

## Usage

### Basic extract

1. **Video file** — stitched equirectangular 360° footage (ideal: 2:1 aspect, e.g. 7680×3840)
2. **Output folder** — where cubemap PNGs are written (stills only)
3. **File prefix** — base name for output files
4. **Sample every (s)** — seconds between time samples
5. **Face max width** — `0` = default 1920 px per face
6. **Cubemap faces** — use previews and checkboxes; quick presets: All, Horizontal, Horiz. + down

7. Click **Extract cubemap stills**

### Face previews & selection

After selecting a video, the app loads **reference-frame previews** for all six faces. Uncheck a direction to skip it entirely in the extract.

- **Mask reference time (s)** — which video frame previews (and masks) are based on; use **Refresh previews** after changing
- Changing video or reference time clears in-session masks (saved library masks are unchanged)

### Masking

Collapsible **Masking** section:

| Action | Purpose |
|--------|---------|
| **Edit equirect mask…** | Draw on the full 360° frame once; mask projects to every cubemap face |
| **Edit mask…** (per face) | Fine-tune one direction (e.g. props on up/down only) |
| **Saved masks** | Load a mask from `usermasks/` (e.g. `drone_props`) |
| **Save current…** | Save polygons to the library for reuse on future footage |
| **Clear** | Remove masks from the current session |

Masks use **normalized polygon coordinates** (JSON, not PNG) so the same library mask works across resolutions and videos. Excluded areas are filled black on output stills.

**Where mask data is stored**

| Location | Contents |
|----------|----------|
| `usermasks/*.json` | Reusable library masks (gitignored; local to your machine) |
| `{video_folder}/{video_stem}_masks/face_masks.json` | Per-video copy written on extract (beside source footage) |
| Output stills folder | Cubemap PNGs only — no mask files |

### UI

- Dark theme, scrollable form, fixed footer with **Extract** and collapsible **Log**
- Header logo links to [grenadefpv.com](https://www.grenadefpv.com/)

## Equirect source tips

True 2:1 equirect sources work best. If you transcode from CineForm or other formats:

```powershell
ffmpeg -i input.mov -vf "scale=7680:3840:force_original_aspect_ratio=decrease,pad=7680:3840:(ow-iw)/2:(oh-ih)/2,setsar=1" -c:v libx265 -crf 18 -an out.mp4
```

The extract pipeline also normalizes SAR and pads to 2:1 before `v360`.

## Project layout

| Path | Purpose |
|------|---------|
| `MP4_360_stills.py` | Main tkinter GUI and ffmpeg pipeline |
| `training_recommendations.py` | Splatter training hints in the UI |
| `install.ps1` | Creates venv, installs deps, checks ffmpeg |
| `assets/logo.png` | Grenade FPV header branding |
| `usermasks/` | Saved reusable mask polygons (`*.json`, gitignored) |

## Verify after code changes

```powershell
python -m py_compile MP4_360_stills.py training_recommendations.py
```

For a quick extract check: confirm log `vf` includes `w=1920:h=1920` and output PNGs are square.

## License

Same as parent Splatter project.
