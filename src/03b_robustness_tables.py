"""STEP 3b - Phase 1 robustness extension (Tasks A and B).

Task A - nested narrow-strait AOI sensitivity
    Re-runs the identical Phase 1 metric definitions for a second, *pre-specified*
    AOI (``config/hormuz_narrow_strait.geojson``, fixed bbox 56.25-56.65 E /
    26.35-26.70 N) and writes ``outputs/tables/aoi_sensitivity.csv``.

    The primary AOI remains the primary specification no matter which AOI shows a
    larger effect; the narrow AOI exists only to test how much of the result
    depends on the AOI's size (a single Sentinel-1 scene covers the narrow box
    fully, so more scenes are usable per phase).

Task B - matching-definition and vessel-size robustness
    * ``outputs/tables/matching_definition_sensitivity.csv``
      matching rate under the primary definition
      (``matched_category != 'unmatched'``) and under the MMSI definition
      (``mmsi is not null``), by period and by month, for both AOIs.
    * ``outputs/tables/length_bin_matching.csv``
      counts, matching rate and (where the sample allows) bootstrap intervals by
      vessel-length bin and phase, with small-sample flags.

Requires steps 01 and 03 to have been run for every AOI listed in the config.

Usage
-----
    python src/01_prepare_sar.py --aoi hormuz_core
    python src/01_prepare_sar.py --aoi hormuz_narrow_strait
    python src/03_build_scene_metrics.py --aoi hormuz_core
    python src/03_build_scene_metrics.py --aoi hormuz_narrow_strait
    python src/03b_robustness_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    in_window,
    list_aois,
    load_config,
    primary_aoi_key,
    resolve_aoi,
    resolve_path,
    resolve_phases,
    resolve_pre_post,
    safe_divide,
    write_json,
)
from hormuz_stats import (  # noqa: E402
    bootstrap_pooled_rate,
    bootstrap_rate_difference,
    pooled_rate,
)

PHASE_ORDER = ["baseline", "acute_shock", "persistent_disruption"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 1 robustness tables (Tasks A and B)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None,
                        help="AOI keys to include (default: every configured AOI)")
    return parser.parse_args(argv)


def load_inputs(
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the scene metrics and AOI detections produced by steps 01/03."""
    processed = resolve_path(cfg["paths"]["processed_dir"])
    interim = resolve_path(cfg["paths"]["interim_dir"])
    metrics_path = aoi_scoped_path(processed / "sar_scene_metrics.parquet", aoi_key, cfg)
    detections_path = aoi_scoped_path(interim / "sar_detections_aoi.parquet", aoi_key, cfg)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing {metrics_path}. Run `python src/01_prepare_sar.py --aoi {aoi_key}` and "
            f"`python src/03_build_scene_metrics.py --aoi {aoi_key}` first."
        )
    metrics = pd.read_parquet(metrics_path)
    detections = pd.read_parquet(detections_path)
    # AOI detections may contain scenes that are unusable (low coverage or a
    # footprint-only platform); those must not enter the metric sample.
    keep = set(metrics.loc[metrics["usable_for_primary_metrics"], "scene_id"])
    detections = detections[detections["scene_id"].isin(keep)].copy()
    logger.info(
        "AOI %s: %d scenes (%d usable, %d intersecting), %d detections in usable scenes",
        aoi_key,
        len(metrics),
        int(metrics["usable_for_primary_metrics"].sum()),
        int(metrics["count_identity_ok"].notna().sum()),
        len(detections),
    )
    return metrics, detections


def period_mask(
    dates: pd.Series,
    cfg: dict[str, Any],
    year: int,
    phase: str,
) -> pd.Series:
    """Boolean mask for a configured phase window in a given year."""
    phases = resolve_phases(cfg, year)
    if phase not in phases:
        raise KeyError(f"unknown phase '{phase}'; configured phases are {list(phases)}")
    return in_window(dates, phases[phase])


# ---------------------------------------------------------------------------
# Task A - AOI sensitivity
# ---------------------------------------------------------------------------
def build_aoi_sensitivity(
    cfg: dict[str, Any],
    per_aoi: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    logger,
) -> pd.DataFrame:
    """Per-AOI, per-phase metrics plus the pre/post calendar-split change."""
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_scenes = int(cfg["stats"].get("min_scenes_per_period", 3))
    event_year = int(cfg["analysis"]["event_year"])
    rows: list[dict[str, Any]] = []

    for aoi_key, (metrics, _detections) in per_aoi.items():
        _key, definition = resolve_aoi(cfg, aoi_key)
        metrics = metrics.copy()
        metrics["d"] = pd.to_datetime(metrics["scene_date_parsed"])
        usable = metrics[metrics["usable_for_primary_metrics"]]

        # ---- per phase, 2026 vs the same calendar window in reference years
        for year_group, years in (("2026", [event_year]), ("reference_2024_2025", list(cfg["analysis"]["reference_years"]))):
            for phase in PHASE_ORDER:
                inside = pd.Series(False, index=metrics.index)
                usable_inside = pd.Series(False, index=usable.index)
                for year in years:
                    try:
                        inside |= period_mask(metrics["d"], cfg, year, phase)
                        usable_inside |= period_mask(usable["d"], cfg, year, phase)
                    except KeyError:
                        continue
                phase_scenes = metrics[inside]
                phase_usable = usable[usable_inside]
                interval = bootstrap_pooled_rate(
                    phase_usable["n_matched"].to_numpy(),
                    (phase_usable["n_matched"] + phase_usable["n_unmatched"]).to_numpy(),
                    iterations=iterations,
                    level=level,
                    seed=seed,
                    min_units_for_ci=min_scenes,
                )
                rows.append(
                    {
                        "aoi_key": aoi_key,
                        "aoi_role": definition.get("role"),
                        "aoi_label": definition.get("label", aoi_key),
                        "row_type": "phase",
                        "year_group": year_group,
                        "period": phase,
                        "n_scenes_intersecting": int(len(phase_scenes)),
                        "n_scenes_usable": int(len(phase_usable)),
                        "n_acquisition_weeks": int(phase_usable["week_key"].nunique()),
                        "n_detections": int(phase_usable["n_total"].sum()),
                        "n_detections_all_scenes": int(phase_scenes["n_total"].sum()),
                        "n_matched": int(phase_usable["n_matched"].sum()),
                        "n_unmatched": int(phase_usable["n_unmatched"].sum()),
                        "n_unknown": int(phase_usable["n_unknown"].sum()),
                        "median_coverage_fraction": float(phase_scenes["coverage_fraction"].median())
                        if len(phase_scenes) else np.nan,
                        "median_coverage_km2": float(phase_scenes["coverage_km2"].median())
                        if len(phase_scenes) else np.nan,
                        "matching_rate": interval.estimate,
                        "matching_rate_ci_low": interval.low,
                        "matching_rate_ci_high": interval.high,
                        "ci_computed": interval.ci_computed,
                        "ci_method": interval.method,
                        "delta_pre_post_pp": np.nan,
                        "delta_ci_low_pp": np.nan,
                        "delta_ci_high_pp": np.nan,
                        "median_sar_density": float(phase_usable["sar_density"].median())
                        if len(phase_usable) else np.nan,
                        "mean_sar_density": float(phase_usable["sar_density"].mean())
                        if len(phase_usable) else np.nan,
                        "coverage_weighted_sar_density": float(
                            safe_divide(phase_usable["n_total"].sum(), phase_usable["coverage_km2"].sum())
                            * float(cfg["sar"]["sar_density_scale_km2"])
                        )
                        if len(phase_usable) and phase_usable["coverage_km2"].sum() > 0
                        else np.nan,
                    }
                )

        # ---- pre/post calendar split (placebo years use the identical days)
        for year in cfg["analysis"]["years"]:
            split = resolve_pre_post(cfg, year)
            year_scenes = usable[usable["d"].dt.year == year]
            pre = year_scenes[in_window(year_scenes["d"], split["pre"])]
            post = year_scenes[in_window(year_scenes["d"], split["post"])]
            interval = bootstrap_rate_difference(
                pre["n_matched"].to_numpy(),
                (pre["n_matched"] + pre["n_unmatched"]).to_numpy(),
                post["n_matched"].to_numpy(),
                (post["n_matched"] + post["n_unmatched"]).to_numpy(),
                iterations=iterations,
                level=level,
                seed=seed,
                min_units_for_ci=min_scenes,
            )
            rows.append(
                {
                    "aoi_key": aoi_key,
                    "aoi_role": definition.get("role"),
                    "aoi_label": definition.get("label", aoi_key),
                    "row_type": "pre_post_split",
                    "year_group": str(year),
                    "period": "pre_post_delta",
                    "n_scenes_intersecting": int(len(year_scenes)),
                    "n_scenes_usable": int(len(pre) + len(post)),
                    "n_acquisition_weeks": int(
                        pd.concat([pre["week_key"], post["week_key"]]).nunique()
                    ),
                    "n_detections": int(pre["n_total"].sum() + post["n_total"].sum()),
                    "n_detections_all_scenes": np.nan,
                    "n_matched": int(pre["n_matched"].sum() + post["n_matched"].sum()),
                    "n_unmatched": int(pre["n_unmatched"].sum() + post["n_unmatched"].sum()),
                    "n_unknown": int(pre["n_unknown"].sum() + post["n_unknown"].sum()),
                    "median_coverage_fraction": float(year_scenes["coverage_fraction"].median())
                    if len(year_scenes) else np.nan,
                    "median_coverage_km2": float(year_scenes["coverage_km2"].median())
                    if len(year_scenes) else np.nan,
                    "matching_rate": np.nan,
                    "matching_rate_ci_low": np.nan,
                    "matching_rate_ci_high": np.nan,
                    "ci_computed": interval.ci_computed,
                    "ci_method": interval.method,
                    "delta_pre_post_pp": 100.0 * interval.estimate,
                    "delta_ci_low_pp": 100.0 * interval.low,
                    "delta_ci_high_pp": 100.0 * interval.high,
                    "median_sar_density": np.nan,
                    "mean_sar_density": np.nan,
                    "coverage_weighted_sar_density": np.nan,
                    "rate_pre": pooled_rate(
                        pre["n_matched"].to_numpy(),
                        (pre["n_matched"] + pre["n_unmatched"]).to_numpy(),
                    ),
                    "rate_post": pooled_rate(
                        post["n_matched"].to_numpy(),
                        (post["n_matched"] + post["n_unmatched"]).to_numpy(),
                    ),
                    "n_scenes_pre": int(len(pre)),
                    "n_scenes_post": int(len(post)),
                }
            )
    frame = pd.DataFrame(rows)
    logger.info(
        "aoi_sensitivity: %d rows across %d AOI(s)", len(frame), frame["aoi_key"].nunique()
    )
    return frame


# ---------------------------------------------------------------------------
# Task B - matching definition sensitivity
# ---------------------------------------------------------------------------
def build_matching_definition_sensitivity(
    cfg: dict[str, Any],
    per_aoi: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    logger,
) -> pd.DataFrame:
    """Primary (match label) vs secondary (MMSI) matching definitions."""
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_scenes = int(cfg["stats"].get("min_scenes_per_period", 3))
    event_year = int(cfg["analysis"]["event_year"])
    rows: list[dict[str, Any]] = []

    def add_period(
        aoi_key: str,
        definition: dict[str, Any],
        row_type: str,
        year_group: str,
        period: str,
        scenes: pd.DataFrame,
    ) -> None:
        """Append one row with both definitions side by side."""
        for label, numerator, denominator in (
            ("primary_matched_category", scenes["n_matched"], scenes["n_matched"] + scenes["n_unmatched"]),
            ("secondary_mmsi_notnull", scenes["n_matched_mmsi"], scenes["n_total"]),
        ):
            interval = bootstrap_pooled_rate(
                numerator.to_numpy(),
                denominator.to_numpy(),
                iterations=iterations,
                level=level,
                seed=seed,
                min_units_for_ci=min_scenes,
            )
            rows.append(
                {
                    "aoi_key": aoi_key,
                    "aoi_role": definition.get("role"),
                    "row_type": row_type,
                    "year_group": year_group,
                    "period": period,
                    "definition": label,
                    "n_scenes": int(len(scenes)),
                    "n_detections": int(scenes["n_total"].sum()),
                    "n_matched": int(numerator.sum()),
                    "n_unmatched_definition_specific": int(denominator.sum() - numerator.sum()),
                    "matching_rate": interval.estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "ci_computed": interval.ci_computed,
                    "delta_pre_post_pp": np.nan,
                    "delta_ci_low_pp": np.nan,
                    "delta_ci_high_pp": np.nan,
                }
            )

    for aoi_key, (metrics, _detections) in per_aoi.items():
        _key, definition = resolve_aoi(cfg, aoi_key)
        usable = metrics[metrics["usable_for_primary_metrics"]].copy()
        usable["d"] = pd.to_datetime(usable["scene_date_parsed"])

        for year_group, years in (
            ("2026", [event_year]),
            ("reference_2024_2025", list(cfg["analysis"]["reference_years"])),
        ):
            for phase in PHASE_ORDER:
                mask = pd.Series(False, index=usable.index)
                for year in years:
                    try:
                        mask |= period_mask(usable["d"], cfg, year, phase)
                    except KeyError:
                        continue
                add_period(aoi_key, definition, "phase", year_group, phase, usable[mask])

        # monthly detail: shows whether the two definitions move together in time
        for (year, month), group in usable.groupby([usable["d"].dt.year, usable["d"].dt.month]):
            add_period(
                aoi_key,
                definition,
                "month",
                str(year),
                f"{int(year)}-{int(month):02d}",
                group,
            )

        # pre/post deltas under both definitions
        for year in cfg["analysis"]["years"]:
            split = resolve_pre_post(cfg, year)
            year_scenes = usable[usable["d"].dt.year == year]
            pre = year_scenes[in_window(year_scenes["d"], split["pre"])]
            post = year_scenes[in_window(year_scenes["d"], split["post"])]
            for label, num_pre, den_pre, num_post, den_post in (
                (
                    "primary_matched_category",
                    pre["n_matched"],
                    pre["n_matched"] + pre["n_unmatched"],
                    post["n_matched"],
                    post["n_matched"] + post["n_unmatched"],
                ),
                (
                    "secondary_mmsi_notnull",
                    pre["n_matched_mmsi"],
                    pre["n_total"],
                    post["n_matched_mmsi"],
                    post["n_total"],
                ),
            ):
                interval = bootstrap_rate_difference(
                    num_pre.to_numpy(),
                    den_pre.to_numpy(),
                    num_post.to_numpy(),
                    den_post.to_numpy(),
                    iterations=iterations,
                    level=level,
                    seed=seed,
                    min_units_for_ci=min_scenes,
                )
                rows.append(
                    {
                        "aoi_key": aoi_key,
                        "aoi_role": definition.get("role"),
                        "row_type": "pre_post_delta",
                        "year_group": str(year),
                        "period": "pre_post_delta",
                        "definition": label,
                        "n_scenes": int(len(pre) + len(post)),
                        "n_detections": int(pre["n_total"].sum() + post["n_total"].sum()),
                        "n_matched": int(num_pre.sum() + num_post.sum()),
                        "n_unmatched_definition_specific": np.nan,
                        "matching_rate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "ci_computed": interval.ci_computed,
                        "delta_pre_post_pp": 100.0 * interval.estimate,
                        "delta_ci_low_pp": 100.0 * interval.low,
                        "delta_ci_high_pp": 100.0 * interval.high,
                    }
                )
    frame = pd.DataFrame(rows)
    logger.info("matching_definition_sensitivity: %d rows", len(frame))
    return frame


# ---------------------------------------------------------------------------
# Task B - length-bin matching
# ---------------------------------------------------------------------------
def build_detection_dispersion(
    cfg: dict[str, Any],
    per_aoi: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    logger,
) -> pd.DataFrame:
    """Per-scene spatial dispersion of detections, by AOI and month.

    Why this matters: a fall in detection counts can mean "fewer vessels" or
    "everything is still there but at one anchorage". The standard distance of a
    scene's detections (km) separates those cases, and it is computed from SAR
    alone, so it is available before any AIS data arrives.

    Column ``median_standard_distance_km`` is the median over scenes of the
    root-mean-square distance of that scene's detections from their centroid;
    a tight cluster of anchored vessels gives a small value.
    """
    rows: list[dict[str, Any]] = []
    scale = 111.32  # km per degree of latitude; longitude scaled by cos(lat)
    for aoi_key, (_metrics, detections) in per_aoi.items():
        _key, definition = resolve_aoi(cfg, aoi_key)
        frame = detections.copy()
        frame["year"] = pd.to_datetime(frame["timestamp"]).dt.year
        frame["month"] = pd.to_datetime(frame["timestamp"]).dt.month
        frame["cell"] = (
            (frame["lat"] / 0.01).round().astype(int).astype(str)
            + "_"
            + (frame["lon"] / 0.01).round().astype(int).astype(str)
        )
        for (year, month), group in frame.groupby(["year", "month"]):
            per_scene = []
            for _scene_id, scene_group in group.groupby("scene_id"):
                if len(scene_group) < 2:
                    per_scene.append(
                        {
                            "n": len(scene_group),
                            "standard_distance_km": 0.0,
                            "lon_span_km": 0.0,
                            "lat_span_km": 0.0,
                        }
                    )
                    continue
                mean_lat = float(scene_group["lat"].mean())
                x = (scene_group["lon"] - scene_group["lon"].mean()) * scale * np.cos(np.radians(mean_lat))
                y = (scene_group["lat"] - scene_group["lat"].mean()) * scale
                per_scene.append(
                    {
                        "n": len(scene_group),
                        "standard_distance_km": float(np.sqrt((x**2 + y**2).mean())),
                        "lon_span_km": float((scene_group["lon"].max() - scene_group["lon"].min()) * scale * np.cos(np.radians(mean_lat))),
                        "lat_span_km": float((scene_group["lat"].max() - scene_group["lat"].min()) * scale),
                    }
                )
            scene_frame = pd.DataFrame(per_scene)
            rows.append(
                {
                    "aoi_key": aoi_key,
                    "aoi_role": definition.get("role"),
                    "year": int(year),
                    "month": int(month),
                    "n_detections": int(len(group)),
                    "n_scenes": int(group["scene_id"].nunique()),
                    "n_distinct_0p01deg_cells": int(group["cell"].nunique()),
                    "detections_per_cell": float(len(group) / max(group["cell"].nunique(), 1)),
                    "median_standard_distance_km": float(scene_frame["standard_distance_km"].median()),
                    "median_scene_lon_span_km": float(scene_frame["lon_span_km"].median()),
                    "median_scene_lat_span_km": float(scene_frame["lat_span_km"].median()),
                    "n_scenes_with_ge5_detections": int((scene_frame["n"] >= 5).sum()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["aoi_key", "year", "month"])
    logger.info("detection_spatial_dispersion: %d rows", len(frame))
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task B - length-bin matching
# ---------------------------------------------------------------------------
def build_length_bin_matching(
    cfg: dict[str, Any],
    per_aoi: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    logger,
) -> pd.DataFrame:
    """Counts and matching rate by vessel-length bin and phase.

    Small cells are flagged explicitly (``small_sample``) and receive no
    bootstrap interval, because a rate computed from a handful of detections in a
    small AOI would otherwise look deceptively precise.
    """
    bins = [float(value) for value in cfg["sar"]["robustness_length_bins_m"]]
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_detections = int(cfg["sar"]["small_sample_min_detections"])
    event_year = int(cfg["analysis"]["event_year"])
    rows: list[dict[str, Any]] = []

    for aoi_key, (_metrics, detections) in per_aoi.items():
        _key, definition = resolve_aoi(cfg, aoi_key)
        frame = detections.copy()
        frame["d"] = pd.to_datetime(frame["timestamp"])
        frame["length_bin"] = pd.cut(frame["length_m"], bins=bins, right=False)
        # right=False makes the labels read as half-open [left, right) so that
        # "<50", "50-100", ">=300" match the requested interpretation.

        for year_group, years in (
            ("2026", [event_year]),
            ("reference_2024_2025", list(cfg["analysis"]["reference_years"])),
        ):
            for phase in PHASE_ORDER:
                mask = pd.Series(False, index=frame.index)
                for year in years:
                    try:
                        mask |= period_mask(frame["d"], cfg, year, phase)
                    except KeyError:
                        continue
                phase_frame = frame[mask]
                for interval_label, group in phase_frame.groupby("length_bin", observed=False):
                    if len(group) == 0:
                        continue
                    numerator = group["matched_flag"].to_numpy()
                    denominator = (group["matched_flag"] | group["unmatched_flag"]).to_numpy()
                    n_denominator = int(denominator.sum())
                    rate = float(numerator.sum() / n_denominator) if n_denominator else float("nan")
                    small = n_denominator < min_detections
                    boot = bootstrap_pooled_rate(
                        numerator,
                        denominator.astype(float),
                        iterations=iterations,
                        level=level,
                        seed=seed,
                        min_units_for_ci=min_detections,
                    )
                    rows.append(
                        {
                            "aoi_key": aoi_key,
                            "aoi_role": definition.get("role"),
                            "year_group": year_group,
                            "phase": phase,
                            "length_bin": f"[{int(interval_label.left)},{int(interval_label.right)})",
                            "length_bin_left_m": float(interval_label.left),
                            "length_bin_right_m": float(interval_label.right),
                            "n_total": int(len(group)),
                            "n_matched": int(numerator.sum()),
                            "n_unmatched": int(group["unmatched_flag"].sum()),
                            "n_unknown": int(group["unknown_flag"].sum()),
                            "matching_rate": rate,
                            "ci_low": boot.low,
                            "ci_high": boot.high,
                            "ci_computed": boot.ci_computed and not small,
                            "small_sample": bool(small),
                            "small_sample_threshold": min_detections,
                            "median_length_m": float(group["length_m"].median()),
                            "note": (
                                f"small sample: {n_denominator} classified detections "
                                f"(< {min_detections}); do not over-interpret"
                                if small
                                else ""
                            ),
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["aoi_key", "year_group", "phase", "length_bin_left_m"]
    )
    logger.info("length_bin_matching: %d rows", len(frame))
    return frame.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("03b_robustness_tables", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 3b - Phase 1 robustness extension (AOI + matching definition + size)")

    aoi_keys = args.aois or list_aois(cfg)
    primary = primary_aoi_key(cfg)
    if primary not in aoi_keys:
        logger.warning("primary AOI '%s' is not in the requested list %s", primary, aoi_keys)
    per_aoi: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for aoi_key in aoi_keys:
        per_aoi[aoi_key] = load_inputs(cfg, aoi_key, logger)

    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    aoi_sensitivity = build_aoi_sensitivity(cfg, per_aoi, logger)
    aoi_sensitivity.to_csv(tables / "aoi_sensitivity.csv", index=False)
    logger.info("wrote %s", tables / "aoi_sensitivity.csv")

    definition_sensitivity = build_matching_definition_sensitivity(cfg, per_aoi, logger)
    definition_sensitivity.to_csv(tables / "matching_definition_sensitivity.csv", index=False)
    logger.info("wrote %s", tables / "matching_definition_sensitivity.csv")

    length_bins = build_length_bin_matching(cfg, per_aoi, logger)
    length_bins.to_csv(tables / "length_bin_matching.csv", index=False)
    logger.info("wrote %s", tables / "length_bin_matching.csv")

    dispersion = build_detection_dispersion(cfg, per_aoi, logger)
    diagnostics_dir = resolve_path(cfg["paths"]["diagnostics_dir"])
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    dispersion.to_csv(diagnostics_dir / "detection_spatial_dispersion.csv", index=False)
    logger.info("wrote %s", diagnostics_dir / "detection_spatial_dispersion.csv")

    # Console summary: the AOI comparison is the point of Task A.
    summary_rows = []
    for aoi_key in aoi_keys:
        subset = aoi_sensitivity[
            (aoi_sensitivity["aoi_key"] == aoi_key)
            & (aoi_sensitivity["row_type"] == "pre_post_split")
        ]
        for row in subset.itertuples(index=False):
            summary_rows.append(
                {
                    "aoi": aoi_key,
                    "aoi_role": row.aoi_role,
                    "year": row.year_group,
                    "scenes_pre": getattr(row, "n_scenes_pre", None),
                    "scenes_post": getattr(row, "n_scenes_post", None),
                    "rate_pre": getattr(row, "rate_pre", None),
                    "rate_post": getattr(row, "rate_post", None),
                    "delta_pp": row.delta_pre_post_pp,
                    "ci_low": row.delta_ci_low_pp,
                    "ci_high": row.delta_ci_high_pp,
                }
            )
    console = pd.DataFrame(summary_rows)
    logger.info("pre/post matching-rate change by AOI:\n%s", console.round(3).to_string(index=False))

    diagnostics = resolve_path(cfg["paths"]["diagnostics_dir"])
    write_json(
        diagnostics / "robustness_tables_summary.json",
        {
            "aois": aoi_keys,
            "primary_aoi": primary,
            "n_rows_aoi_sensitivity": int(len(aoi_sensitivity)),
            "n_rows_definition_sensitivity": int(len(definition_sensitivity)),
            "n_rows_length_bins": int(len(length_bins)),
            "pre_post_by_aoi": console.to_dict(orient="records"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
