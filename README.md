# 360 Cubemap Stills Extractor

Standalone Windows-friendly tool that converts **360° equirectangular video** into **cubemap face PNGs** for Splatter, Postshot, COLMAP, and similar photogrammetry workflows.

Extracted from [Splatter](https://github.com/gusisin/Splatter) as a focused, dependency-light project.

## What it does

- Samples an equirect MP4/MOV at a fixed interval (e.g. every 2 seconds)
- Projects each sample into cubemap faces: **front, right, back, left, down** (optional up)
- Writes `{prefix}-{frame}_{face}.png` (e.g. `eqipano-000001_f.png`)
- Outputs **square** pinhole faces (default 1920×1920) with correct 90°×90° FOV
- Optionally strips PNG metadata for Postshot / gamma compatibility
- Shows Splatter training hints based on planned still count

## Requirements

| Tool | Required | Notes |
|------|----------|-------|
| Python 3.10+ | Yes | tkinter included with standard Windows Python |
| [ffmpeg](https://ffmpeg.org/) + ffprobe | Yes | Must be on `PATH` |
| Pillow | Yes | `pip install -r requirements.txt` |
| ImageMagick (`magick`) | No | Faster metadata strip if available |

## Quick start

```powershell
cd d:\StillsExtractor
.\install.ps1
.\.venv\Scripts\Activate.ps1
python MP4_360_stills.py
```

Or without the installer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python MP4_360_stills.py
```

## Usage

1. **Video file** — stitched equirectangular 360° footage (ideal: 2:1 aspect, e.g. 7680×3840)
2. **Output folder** — where PNG faces are written
3. **File prefix** — base name for output files
4. **Sample every (s)** — time between extracted frames
5. **Face max width** — `0` = default 1920 px per face
6. **Cubemap faces** — horizontal ring + down is typical for outdoor scans

## Equirect source tips

True 2:1 equirect sources work best. If you transcode from CineForm or other formats:

```powershell
ffmpeg -i input.mov -vf "scale=7680:3840:force_original_aspect_ratio=decrease,pad=7680:3840:(ow-iw)/2:(oh-ih)/2,setsar=1" -c:v libx265 -crf 18 -an out.mp4
```

The app also normalizes non–2:1 inputs at extract time.

## Files

| File | Purpose |
|------|---------|
| `MP4_360_stills.py` | Main tkinter GUI and ffmpeg pipeline |
| `training_recommendations.py` | Training preset hints shown in the UI |
| `install.ps1` | Creates venv, installs deps, checks ffmpeg |

## License

Same as parent Splatter project.
