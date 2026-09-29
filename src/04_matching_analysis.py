"""STEP 4 (Phase 3) - formal matching-rate analysis.

Terminology (fixed for the rest of the project)
-----------------------------------------------
``PRIMARY AIS-SAR matching definition``
    ``matched_category != "unmatched"`` - a confirmed AIS-SAR match.
``MMSI-present diagnostic``
    ``mmsi is not null`` - **not** a confirmed match. A detection can carry a
    candidate MMSI while GFW still labels it ``unmatched`` (see the cross-tab this
    script writes). It is reported as a diagnostic only and must never be
    described as an alternative matching definition.

Outputs
-------
``outputs/tables/matching_phase_estimates.csv``
    Per AOI x year-group x phase: scene counts, detection counts, matched,
    unmatched, unknown, pooled rate, scene-mean rate, bootstrap CI.
``outputs/tables/placebo_results.csv``
    Formal pre/post calendar-split table per AOI and year (pooled rate,
    scene-mean rate, difference in percentage points, bootstrap CI).
``outputs/tables/matching_definition_crosstab.csv``
    ``matched_category`` x MMSI presence x ``matching_score`` distribution, by
    year and by event phase, including quantiles of ``matching_score`` for the
    ``unmatched + MMSI present`` group. Purpose: establish whether that group is
    a set of low-confidence candidate identities rather than accepted matches.
    No vessel-behaviour claim is derived from it.

Usage
-----
    python src/04_matching_analysis.py
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
    write_json,
)
from hormuz_stats import bootstrap_pooled_rate, bootstrap_rate_difference  # noqa: E402

PHASES = ("baseline", "acute_shock", "persistent_disruption")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Formal matching-rate analysis (Phase 3)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None,
                        help="AOI keys (default: the two Hormuz AOIs)")
    return parser.parse_args(argv)


def default_aois(cfg: dict[str, Any]) -> list[str]:
    """Hormuz AOIs plus any explicit sensitivity AOI, excluding corridors."""
    keys = []
    for key in list_aois(cfg):
        role = resolve_aoi(cfg, key)[1].get("role")
        if role == "technical_reference_corridor":
            continue
        keys.append(key)
    return keys


def load_scene_metrics(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load the AOI's scene metrics, restricted to the primary analysis sample."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet", aoi_key, cfg
    )
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run step 01 and 03 for AOI {aoi_key} first")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    return frame[frame["usable_for_primary_metrics"]].copy()


def load_detections(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load AOI detections, keeping only scenes in the primary analysis sample."""
    metrics = load_scene_metrics(cfg, aoi_key, logger)
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet", aoi_key, cfg
    )
    frame = pd.read_parquet(path)
    frame = frame[frame["scene_id"].isin(set(metrics["scene_id"]))].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"])
    return frame


def phase_estimates(
    metrics: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """Scene counts, detection counts and matching rates per year-group and phase."""
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_scenes = int(cfg["stats"].get("min_scenes_per_period", 3))
    event_year = int(cfg["analysis"]["event_year"])
    rows: list[dict[str, Any]] = []

    year_groups = (
        ("2026_event", [event_year]),
        ("2025_placebo", [2025]),
        ("2024_placebo", [2024]),
        ("reference_2024_2025", list(cfg["analysis"]["reference_years"])),
    )
    for group_name, years in year_groups:
        for phase in PHASES:
            mask = pd.Series(False, index=metrics.index)
            for year in years:
                try:
                    mask |= in_window(metrics["date"], resolve_phases(cfg, year)[phase])
                except KeyError:
                    continue
            subset = metrics[mask]
            if subset.empty:
                continue
            numerator = subset["n_matched"].to_numpy()
            denominator = (subset["n_matched"] + subset["n_unmatched"]).to_numpy()
            interval = bootstrap_pooled_rate(
                numerator, denominator, iterations=iterations, level=level, seed=seed,
                min_units_for_ci=min_scenes,
            )
            scene_rates = subset.loc[subset["n_total"] > 0, "matching_rate"].dropna()
            rows.append(
                {
                    "aoi_key": aoi_key,
                    "aoi_role": resolve_aoi(cfg, aoi_key)[1].get("role"),
                    "year_group": group_name,
                    "years": "+".join(str(year) for year in years),
                    "phase": phase,
                    "n_scenes": int(len(subset)),
                    "n_scenes_with_detections": int((subset["n_total"] > 0).sum()),
                    "n_acquisition_weeks": int(subset["week_key"].nunique()),
                    "n_detections": int(subset["n_total"].sum()),
                    "n_matched": int(subset["n_matched"].sum()),
                    "n_unmatched": int(subset["n_unmatched"].sum()),
                    "n_unknown": int(subset["n_unknown"].sum()),
                    "pooled_matching_rate": interval.estimate,
                    "pooled_ci_low": interval.low,
                    "pooled_ci_high": interval.high,
                    "ci_computed": interval.ci_computed,
                    "scene_mean_matching_rate": float(scene_rates.mean()) if len(scene_rates) else np.nan,
                    "scene_median_matching_rate": float(scene_rates.median()) if len(scene_rates) else np.nan,
                    "median_coverage_fraction": float(subset["coverage_fraction"].median()),
                    "median_sar_density": float(subset["sar_density"].median()),
                    "median_length_m": float(subset["median_length_m"].median()),
                    "bootstrap_unit": "scene",
                    "bootstrap_iterations": iterations,
                }
            )
    logger.info("phase estimates: %d rows for AOI %s", len(rows), aoi_key)
    return pd.DataFrame(rows)


def placebo_table(
    metrics: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """Formal pre/post calendar split for every year, with a bootstrap interval."""
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_scenes = int(cfg["stats"].get("min_scenes_per_period", 3))
    event_year = int(cfg["analysis"]["event_year"])
    rows: list[dict[str, Any]] = []
    for year in cfg["analysis"]["years"]:
        split = resolve_pre_post(cfg, year)
        year_scenes = metrics[metrics["date"].dt.year == year]
        pre = year_scenes[in_window(year_scenes["date"], split["pre"])]
        post = year_scenes[in_window(year_scenes["date"], split["post"])]
        interval = bootstrap_rate_difference(
            pre["n_matched"].to_numpy(),
            (pre["n_matched"] + pre["n_unmatched"]).to_numpy(),
            post["n_matched"].to_numpy(),
            (post["n_matched"] + post["n_unmatched"]).to_numpy(),
            iterations=iterations, level=level, seed=seed, min_units_for_ci=min_scenes,
        )
        pre_den = (pre["n_matched"] + pre["n_unmatched"]).sum()
        post_den = (post["n_matched"] + post["n_unmatched"]).sum()
        pre_scene_rates = pre.loc[pre["n_total"] > 0, "matching_rate"].dropna()
        post_scene_rates = post.loc[post["n_total"] > 0, "matching_rate"].dropna()
        rows.append(
            {
                "aoi_key": aoi_key,
                "aoi_role": resolve_aoi(cfg, aoi_key)[1].get("role"),
                "year": year,
                "role": "event_year" if year == event_year else "placebo_year",
                "split_pre": f"{split['pre'][0].date()}..{split['pre'][1].date()}",
                "split_post": f"{split['post'][0].date()}..{split['post'][1].date()}",
                "n_scenes_pre": int(len(pre)),
                "n_scenes_post": int(len(post)),
                "n_detections_pre": int(pre["n_total"].sum()),
                "n_detections_post": int(post["n_total"].sum()),
                "n_matched_pre": int(pre["n_matched"].sum()),
                "n_matched_post": int(post["n_matched"].sum()),
                "n_unmatched_pre": int(pre["n_unmatched"].sum()),
                "n_unmatched_post": int(post["n_unmatched"].sum()),
                "n_unknown_pre": int(pre["n_unknown"].sum()),
                "n_unknown_post": int(post["n_unknown"].sum()),
                "pooled_rate_pre": float(pre["n_matched"].sum() / pre_den) if pre_den else np.nan,
                "pooled_rate_post": float(post["n_matched"].sum() / post_den) if post_den else np.nan,
                "scene_mean_rate_pre": float(pre_scene_rates.mean()) if len(pre_scene_rates) else np.nan,
                "scene_mean_rate_post": float(post_scene_rates.mean()) if len(post_scene_rates) else np.nan,
                "delta_pooled_pp": 100.0 * interval.estimate,
                "delta_ci_low_pp": 100.0 * interval.low,
                "delta_ci_high_pp": 100.0 * interval.high,
                "delta_scene_mean_pp": (
                    100.0
                    * (post_scene_rates.mean() - pre_scene_rates.mean())
                    if len(pre_scene_rates) and len(post_scene_rates)
                    else np.nan
                ),
                "ci_computed": interval.ci_computed,
                "bootstrap_unit": "scene",
            }
        )
    logger.info("placebo/pre-post rows: %d for AOI %s", len(rows), aoi_key)
    return pd.DataFrame(rows)


def definition_crosstab(
    detections: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """matched_category x MMSI presence x matching_score distribution.

    The three groups of interest are reported explicitly:
    ``matched + MMSI present``, ``unmatched + MMSI present`` and
    ``unmatched + MMSI absent``. For the middle group the quantiles of
    ``matching_score`` are the evidence used to decide whether it looks like a set
    of low-confidence candidate identities. Nothing here identifies vessel
    behaviour.
    """
    frame = detections.copy()
    frame["mmsi_present"] = frame["mmsi"].notna()
    conditions = [
        frame["matched_flag"],
        (~frame["matched_flag"]) & frame["mmsi_present"],
        (~frame["matched_flag"]) & (~frame["mmsi_present"]),
    ]
    labels = ["matched_category_matched__mmsi_present",
              "matched_category_unmatched__mmsi_present",
              "matched_category_unmatched__mmsi_absent"]
    frame["definition_group"] = np.select(conditions, labels, default="other")
    frame["year"] = pd.to_datetime(frame["timestamp"]).dt.year

    phase_labels = pd.Series("outside_phases", index=frame.index, dtype=object)
    for phase in PHASES:
        for year in cfg["analysis"]["years"]:
            try:
                window = resolve_phases(cfg, year)[phase]
            except KeyError:
                continue
            mask = in_window(pd.to_datetime(frame["timestamp"]), window)
            phase_labels.loc[mask] = f"{year}_{phase}"
    frame["phase_label"] = phase_labels

    records: list[dict[str, Any]] = []
    groupings = (
        ("year", ["year"]),
        ("phase", ["phase_label"]),
    )
    for grouping_name, keys in groupings:
        for key, group in frame.groupby(keys[0] if len(keys) == 1 else keys, sort=True):
            label = key if isinstance(key, str) else str(key)
            if grouping_name == "phase" and label == "outside_phases":
                continue
            total = len(group)
            for definition_group, sub in group.groupby("definition_group"):
                scores = pd.to_numeric(sub["matching_score"], errors="coerce").dropna()
                records.append(
                    {
                        "aoi_key": aoi_key,
                        "grouping": grouping_name,
                        "group_key": label,
                        "definition_group": definition_group,
                        "n_detections": int(len(sub)),
                        "share_of_group": float(len(sub) / total) if total else np.nan,
                        "n_with_mmsi": int(sub["mmsi_present"].sum()),
                        "matching_score_p05": float(scores.quantile(0.05)) if len(scores) else np.nan,
                        "matching_score_p25": float(scores.quantile(0.25)) if len(scores) else np.nan,
                        "matching_score_median": float(scores.median()) if len(scores) else np.nan,
                        "matching_score_p75": float(scores.quantile(0.75)) if len(scores) else np.nan,
                        "matching_score_p95": float(scores.quantile(0.95)) if len(scores) else np.nan,
                        "share_matching_score_below_1": float((scores < 1.0).mean()) if len(scores) else np.nan,
                        "median_length_m": float(sub["length_m"].median()) if len(sub) else np.nan,
                    }
                )
    out = pd.DataFrame(records).sort_values(["grouping", "group_key", "definition_group"])
    logger.info("definition cross-tab: %d rows for AOI %s", len(out), aoi_key)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("04_matching_analysis", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 4 (Phase 3) - formal matching-rate analysis")

    aoi_keys = args.aois or default_aois(cfg)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    phase_frames: list[pd.DataFrame] = []
    placebo_frames: list[pd.DataFrame] = []
    crosstab_frames: list[pd.DataFrame] = []
    for aoi_key in aoi_keys:
        metrics = load_scene_metrics(cfg, aoi_key, logger)
        detections = load_detections(cfg, aoi_key, logger)
        phase_frames.append(phase_estimates(metrics, cfg, aoi_key, logger))
        placebo_frames.append(placebo_table(metrics, cfg, aoi_key, logger))
        if cfg.get("identification", {}).get("matching_definition_crosstab", True):
            crosstab_frames.append(definition_crosstab(detections, cfg, aoi_key, logger))

    phases = pd.concat(phase_frames, ignore_index=True)
    phases.to_csv(tables / "matching_phase_estimates.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "matching_phase_estimates.csv", len(phases))

    placebo = pd.concat(placebo_frames, ignore_index=True)
    placebo.to_csv(tables / "placebo_results.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "placebo_results.csv", len(placebo))

    if crosstab_frames:
        crosstab = pd.concat(crosstab_frames, ignore_index=True)
        crosstab.to_csv(tables / "matching_definition_crosstab.csv", index=False)
        logger.info("wrote %s (%d rows)", tables / "matching_definition_crosstab.csv", len(crosstab))

    # ---- console headline: event year vs placebo years -------------------
    headline = phases[
        (phases["phase"] == "persistent_disruption")
        & (phases["year_group"].isin(["2026_event", "2025_placebo", "2024_placebo"]))
    ][["aoi_key", "year_group", "phase", "n_scenes", "n_detections", "pooled_matching_rate",
       "scene_mean_matching_rate", "pooled_ci_low", "pooled_ci_high"]]
    logger.info("persistent-phase matching rates:\n%s", headline.round(3).to_string(index=False))
    logger.info("pre/post calibration splits:\n%s",
                placebo[["aoi_key", "year", "role", "pooled_rate_pre", "pooled_rate_post",
                         "delta_pooled_pp", "delta_ci_low_pp", "delta_ci_high_pp"]]
                .round(3).to_string(index=False))

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "matching_analysis_summary.json",
        {
            "terminology": {
                "primary_definition": "matched_category != 'unmatched'",
                "mmsi_present": "MMSI-present diagnostic (NOT a confirmed match)",
            },
            "aois": aoi_keys,
            "primary_aoi": primary_aoi_key(cfg),
            "bootstrap_unit": "scene",
            "bootstrap_iterations": int(cfg["stats"]["bootstrap_iterations"]),
            "persistent_phase_matching_rates": headline.to_dict(orient="records"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
