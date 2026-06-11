"""Heuristic training settings from extracted stills and COLMAP sparse stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TrainingPriority = Literal["speed", "balanced", "quality"]

DEFAULT_FACE_WIDTH = 1920
DEFAULT_DOWNSAMPLE = 2
# Typical registered-image fraction for outdoor 360° cubemap COLMAP (lake ~62%).
DEFAULT_ASSUMED_REGISTRATION_FRACTION = 0.62


@dataclass(frozen=True)
class DatasetTrainingStats:
    stills_count: int
    registered_images: int | None
    colmap_points: int | None
    face_max_width: int

    @property
    def effective_views(self) -> int:
        if self.registered_images and self.registered_images > 0:
            return self.registered_images
        return max(self.stills_count, 1)

    @property
    def seed_splats(self) -> int:
        if self.colmap_points and self.colmap_points > 0:
            return self.colmap_points
        return max(self.effective_views * 80, 10_000)


@dataclass(frozen=True)
class TrainingPlan:
    priority: TrainingPriority
    target_splats: int
    mode: str
    config_name: str
    downsample: int
    iterations: int
    optimizer: str
    max_gaussians: int | None
    densify_end_iteration: int | None


def _clamp_splats(value: float) -> int:
    return int(max(50_000, min(round(value), 2_000_000)))


def estimate_target_splats(stats: DatasetTrainingStats, downsample: int) -> dict[TrainingPriority, int]:
    """Rough final Gaussian counts for each priority tier."""
    views = stats.effective_views
    seed = stats.seed_splats
    face_px = max(256, stats.face_max_width // max(int(downsample), 1))
    res_factor = max(0.6, min(1.8, face_px / 960.0))

    per_view = {
        "speed": 250,
        "balanced": 650,
        "quality": 1400,
    }
    out: dict[TrainingPriority, int] = {}
    for key, pv in per_view.items():
        from_views = views * pv * res_factor
        from_seed = seed * (3 if key == "speed" else 6 if key == "balanced" else 12)
        out[key] = _clamp_splats(max(from_views, from_seed))
    return out


def _pick_downsample(priority: TrainingPriority, stills_count: int) -> int:
    if priority == "quality":
        return 1 if stills_count <= 240 else 2
    if priority == "speed":
        if stills_count >= 360:
            return 3
        if stills_count >= 180:
            return 2
        return 2
    return 2 if stills_count >= 120 else 1


def build_training_plan(stats: DatasetTrainingStats, priority: TrainingPriority) -> TrainingPlan:
    targets = estimate_target_splats(stats, _pick_downsample(priority, stats.stills_count))
    target = targets[priority]
    downsample = _pick_downsample(priority, stats.stills_count)

    if priority == "speed":
        return TrainingPlan(
            priority=priority,
            target_splats=target,
            mode="3DGUT",
            config_name="apps/colmap_3dgut_mcmc.yaml",
            downsample=downsample,
            iterations=15_000,
            optimizer="selective_adam",
            max_gaussians=target,
            densify_end_iteration=None,
        )
    if priority == "quality":
        return TrainingPlan(
            priority=priority,
            target_splats=target,
            mode="3DGUT",
            config_name="apps/colmap_3dgut.yaml",
            downsample=downsample,
            iterations=30_000,
            optimizer="selective_adam",
            max_gaussians=None,
            densify_end_iteration=None,
        )
    return TrainingPlan(
        priority=priority,
        target_splats=target,
        mode="3DGUT",
        config_name="apps/colmap_3dgut.yaml",
        downsample=downsample,
        iterations=30_000,
        optimizer="selective_adam",
        max_gaussians=None,
        densify_end_iteration=12_000,
    )


def _read_sparse_stats(model_dir: Path) -> dict[str, int | None]:
    stats: dict[str, int | None] = {"images": None, "points": None}

    def read_count_bin(name: str) -> int | None:
        path = model_dir / name
        if not path.is_file():
            return None
        try:
            head = path.read_bytes()[:8]
        except OSError:
            return None
        if len(head) != 8:
            return None
        return int.from_bytes(head, byteorder="little", signed=False)

    stats["images"] = read_count_bin("images.bin")
    stats["points"] = read_count_bin("points3D.bin")
    return stats


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def gather_training_stats(
    data_path: str,
    *,
    splat_base_dir: str = "",
    splat_name: str = "",
) -> DatasetTrainingStats | None:
    """Collect still/COLMAP counts from manifest and/or the dataset on disk."""
    manifest: dict[str, Any] | None = None
    if splat_name and splat_base_dir:
        manifest = _load_manifest(Path(splat_base_dir).expanduser() / splat_name / "manifest.json")

    stills_count = 0
    registered: int | None = None
    points: int | None = None
    face_max_width = DEFAULT_FACE_WIDTH

    if manifest:
        stills_count = int(manifest.get("stills_count") or 0)
        settings = manifest.get("settings") if isinstance(manifest.get("settings"), dict) else {}
        raw_w = int(settings.get("max_width") or 0)
        face_max_width = raw_w if raw_w > 0 else DEFAULT_FACE_WIDTH
        colmap_stats = manifest.get("colmap_stats")
        if isinstance(colmap_stats, dict):
            stills_count = int(colmap_stats.get("stills_count") or stills_count)
            ri = colmap_stats.get("registered_images")
            pt = colmap_stats.get("points3d")
            registered = int(ri) if isinstance(ri, int) else None
            points = int(pt) if isinstance(pt, int) else None

    ds = Path(data_path).expanduser()
    sparse0 = ds / "sparse" / "0"
    if not sparse0.is_dir() and ds.name.lower() == "stills":
        sparse0 = ds.parent / "dataset" / "sparse" / "0"
    if sparse0.is_dir():
        live = _read_sparse_stats(sparse0)
        if isinstance(live.get("images"), int):
            registered = live["images"]
        if isinstance(live.get("points"), int):
            points = live["points"]

    if stills_count <= 0 and not registered:
        if manifest and manifest.get("stills_files"):
            stills_count = len(manifest["stills_files"])
        elif stills_count <= 0:
            return None

    if stills_count <= 0 and registered:
        stills_count = registered

    return DatasetTrainingStats(
        stills_count=max(stills_count, 0),
        registered_images=registered,
        points3d=points,
        face_max_width=face_max_width,
    )


def stats_from_planned_stills(
    stills_count: int,
    face_max_width: int,
    *,
    assume_registration_fraction: float | None = DEFAULT_ASSUMED_REGISTRATION_FRACTION,
) -> DatasetTrainingStats | None:
    """Build stats from a planned extraction (before COLMAP exists)."""
    if stills_count <= 0:
        return None
    registered: int | None = None
    if assume_registration_fraction is not None and 0 < assume_registration_fraction <= 1:
        registered = max(1, int(round(stills_count * assume_registration_fraction)))
    width = int(face_max_width) if int(face_max_width) > 0 else DEFAULT_FACE_WIDTH
    return DatasetTrainingStats(
        stills_count=stills_count,
        registered_images=registered,
        colmap_points=None,
        face_max_width=width,
    )


def plan_from_planned_stills(
    stills_count: int,
    face_max_width: int,
    priority: TrainingPriority,
) -> tuple[TrainingPlan | None, str]:
    stats = stats_from_planned_stills(stills_count, face_max_width)
    if stats is None:
        return None, "Splatter training: select a video to see suggested splat counts."
    plan = build_training_plan(stats, priority)
    return plan, format_recommendation_plaintext(stats, plan, pre_colmap=True)


def _recommendation_lines(
    stats: DatasetTrainingStats,
    plan: TrainingPlan,
    *,
    pre_colmap: bool,
) -> list[str]:
    targets = estimate_target_splats(stats, plan.downsample)
    reg = stats.registered_images
    if pre_colmap and reg:
        reg_note = f"~{reg:,} registered est. ({DEFAULT_ASSUMED_REGISTRATION_FRACTION:.0%} of stills)"
    elif reg:
        reg_note = f"{reg:,} registered"
        if stats.stills_count > 0:
            reg_note += f" ({reg / stats.stills_count * 100:.0f}% of stills)"
    else:
        reg_note = "registration unknown"

    lines = [
        f"Stills: {stats.stills_count:,} · COLMAP: {reg_note} · Seed est.: {stats.seed_splats:,}",
        f"Target splats ({plan.priority}): ~{plan.target_splats:,} "
        f"(speed ~{targets['speed']:,} · balanced ~{targets['balanced']:,} · quality ~{targets['quality']:,})",
        f"Splatter preset: {plan.config_name}, downsample={plan.downsample}, "
        f"{plan.iterations:,} iters, {plan.optimizer}",
    ]
    if plan.max_gaussians:
        lines.append(f"MCMC cap: strategy.add.max_n_gaussians={plan.max_gaussians:,}")
    elif plan.densify_end_iteration and plan.densify_end_iteration < 15_000:
        lines.append(f"Densify stop: iteration {plan.densify_end_iteration:,}")
    if pre_colmap:
        lines.append(
            "Pre-COLMAP estimate — load this session in Splatter Train tab after COLMAP for refined numbers."
        )
    else:
        lines.append(
            "Heuristic only — outdoor 360° cubemap scenes vary. Quality for final exports, Speed for previews."
        )
    return lines


def format_recommendation_plaintext(
    stats: DatasetTrainingStats,
    plan: TrainingPlan,
    *,
    pre_colmap: bool = False,
) -> str:
    return "\n".join(_recommendation_lines(stats, plan, pre_colmap=pre_colmap))


def format_recommendation_markdown(stats: DatasetTrainingStats, plan: TrainingPlan) -> str:
    targets = estimate_target_splats(stats, plan.downsample)
    reg = stats.registered_images
    reg_note = f"{reg:,} registered" if reg else "registration unknown"
    pct = ""
    if reg and stats.stills_count > 0:
        pct = f" ({reg / stats.stills_count * 100:.0f}% of stills)"

    lines = [
        "### Training recommendation",
        f"- **Stills:** {stats.stills_count:,} · **COLMAP:** {reg_note}{pct} · "
        f"**Seed points:** {stats.seed_splats:,}",
        f"- **Target splats ({plan.priority}):** ~{plan.target_splats:,} "
        f"(speed ~{targets['speed']:,} · balanced ~{targets['balanced']:,} · quality ~{targets['quality']:,})",
        f"- **Applied preset:** `{plan.config_name}`, downsample={plan.downsample}, "
        f"{plan.iterations:,} iterations, {plan.optimizer}",
    ]
    if plan.max_gaussians:
        lines.append(f"- **MCMC cap:** `strategy.add.max_n_gaussians={plan.max_gaussians:,}`")
    elif plan.densify_end_iteration and plan.densify_end_iteration < 15_000:
        lines.append(
            f"- **Densify stop (speed/balanced):** iteration {plan.densify_end_iteration:,}"
        )
    lines.append(
        "_Heuristic only — outdoor 360° cubemap scenes vary. Use **Quality** for final exports, **Speed** for previews._"
    )
    return "\n".join(lines)


def plan_for_session(
    data_path: str,
    priority: TrainingPriority,
    *,
    splat_base_dir: str = "",
    splat_name: str = "",
) -> tuple[TrainingPlan | None, str]:
    stats = gather_training_stats(data_path, splat_base_dir=splat_base_dir, splat_name=splat_name)
    if stats is None:
        return None, (
            "### Training recommendation\n"
            "Load a splat session with extracted stills (and COLMAP, if available) to see suggested splat counts."
        )
    plan = build_training_plan(stats, priority)
    return plan, format_recommendation_markdown(stats, plan)


def training_hydra_overrides(plan: TrainingPlan) -> list[str]:
    overrides: list[str] = []
    if plan.max_gaussians:
        overrides.append(f"strategy.add.max_n_gaussians={int(plan.max_gaussians)}")
    if plan.densify_end_iteration and "mcmc" not in plan.config_name.lower():
        overrides.append(f"strategy.densify.end_iteration={int(plan.densify_end_iteration)}")
    return overrides
