"""STEP 9 (Phase 3) - identification and robustness checks.

Two independent groups of checks, neither of which is a causal control:

A. Composition diagnostics (``outputs/tables/composition_shift.csv``)
   Pre/post distributions of ``length_m``, ``presence_score``, ``fishing_score``,
   spatial cell and ``matched_category`` for every AOI, plus a total-variation
   distance summarising how much the composition moved.

B. Pipeline-transition falsification (``outputs/tables/pipeline_falsification.csv``)
   6A. Global: the archive-wide monthly detection files, comparing February with
       March for 2026 (the pipev3 -> pipev4 transition) and for 2024 / 2025 as
       placebo seasonal transitions, globally and by fixed coarse region.
   6B. External technical reference corridors (Gibraltar, Singapore, Dover): the
       same February-to-March comparison inside pre-specified boxes run through
       the identical AOI pipeline.

   Purpose: did AIS-SAR matching undergo a broad discontinuity at 2026-03-01?
   These corridors are **not** causal controls for the Hormuz disruption and are
   not used to estimate an event effect.

Length-standardised changes use direct standardisation to the "before" period's
length distribution over bins that have enough observations in both periods.

Usage
-----
    python src/09_identification_checks.py
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
    aoi_geometry,
    aoi_scoped_path,
    banner,
    bbox_of,
    discover_sar_files,
    ensure_dirs,
    files_in_window,
    get_logger,
    list_aois,
    load_config,
    primary_aoi_key,
    read_csv_fast,
    resolve_aoi,
    resolve_path,
    resolve_phases,
    in_window,
    write_json,
)
from hormuz_stats import bootstrap_rate_difference  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 3 identification checks")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--skip-global", action="store_true",
                        help="skip the archive-wide diagnostic (slower)")
    return parser.parse_args(argv)


def cell_labels(lat: pandas.Series, lon: pandas.Series, degrees: float) -> pandas.Series:
    """Fixed geographic cell label from a pre-specified grid size."""
    lat_idx = np.floor(pd.to_numeric(lat, errors="coerce") / degrees).astype("Int64")
    lon_idx = np.floor(pd.to_numeric(lon, errors="coerce") / degrees).astype("Int64")
    return lat_idx.astype("string") + "_" + lon_idx.astype("string")


# ---------------------------------------------------------------------------
# A. composition diagnostics
# ---------------------------------------------------------------------------
def total_variation(counter_a: pd.Series, counter_b: pd.Series) -> float:
    """Total-variation distance between two categorical distributions."""
    a = counter_a / counter_a.sum() if counter_a.sum() else counter_a
    b = counter_b / counter_b.sum() if counter_b.sum() else counter_b
    index = a.index.union(b.index)
    return float(0.5 * np.abs(a.reindex(index, fill_value=0) - b.reindex(index, fill_value=0)).sum())


def composition_shift(
    detections: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """Pre/post distribution comparison for every composition variable."""
    cell_degrees = float(cfg["identification"]["spatial_cell_degrees"])
    bins = [float(value) for value in cfg["identification"]["length_bins_m"]]
    event_year = int(cfg["analysis"]["event_year"])
    phases = resolve_phases(cfg, event_year)
    event_window = (
        phases["acute_shock"][0],
        phases["persistent_disruption"][1],
    )
    frame = detections.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    # The pre/post contrast is made *within the event year* so that it isolates the
    # disruption rather than mixing in the reference years as a pseudo-baseline.
    frame = frame[
        frame["timestamp"].dt.year == event_year
    ].copy()
    frame["length_bin"] = pd.cut(frame["length_m"], bins=bins, right=False).astype(str)
    frame["cell"] = cell_labels(frame["lat"], frame["lon"], cell_degrees)
    pre_window = phases["baseline"]
    frame["is_pre"] = in_window(frame["timestamp"], pre_window).astype(int)
    frame["is_post"] = in_window(frame["timestamp"], event_window).astype(int)
    dropped_outside = int(((frame["is_pre"] == 0) & (frame["is_post"] == 0)).sum())
    frame = frame[(frame["is_pre"] == 1) | (frame["is_post"] == 1)].copy()
    frame["match_label"] = frame["match_label"].fillna("NA").astype(str)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variable in ("length_bin", "cell", "match_label"):
        for period, value in (("pre_2026_baseline", 0), ("post_2026_event", 1)):
            subset = frame[frame["is_post"] == value]
            counts = subset[variable].value_counts(dropna=False)
            for level, count in counts.items():
                rows.append(
                    {
                        "aoi_key": aoi_key,
                        "variable": variable,
                        "level": str(level),
                        "period": period,
                        "n_detections": int(count),
                        "share_within_period": float(count / counts.sum()) if counts.sum() else np.nan,
                    }
                )
        pre_counts = frame.loc[frame["is_post"] == 0, variable].value_counts(dropna=False)
        post_counts = frame.loc[frame["is_post"] == 1, variable].value_counts(dropna=False)
        summary.append(
            {
                "aoi_key": aoi_key,
                "variable": variable,
                "level": "__distribution_distance__",
                "period": "pre_vs_post",
                "n_detections": int(pre_counts.sum() + post_counts.sum()),
                "share_within_period": total_variation(pre_counts, post_counts),
            }
        )

    for variable, column in (
        ("length_m", "length_m"),
        ("presence_score", "presence_score"),
        ("fishing_score", "fishing_score"),
        ("matching_score_reference_only", "matching_score"),
    ):
        if column not in frame.columns:
            logger.info("composition variable %s absent; skipped", column)
            continue
        for period, value in (("pre_2026_baseline", 0), ("post_2026_event", 1)):
            series = pd.to_numeric(frame.loc[frame["is_post"] == value, column], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "aoi_key": aoi_key,
                    "variable": variable,
                    "level": "p25",
                    "period": period,
                    "n_detections": int(len(series)),
                    "share_within_period": np.nan,
                    "value": float(series.quantile(0.25)),
                }
            )
            for level, statistic in (("median", series.median()), ("p75", series.quantile(0.75)),
                                     ("mean", series.mean())):
                rows.append(
                    {
                        "aoi_key": aoi_key,
                        "variable": variable,
                        "level": level,
                        "period": period,
                        "n_detections": int(len(series)),
                        "share_within_period": np.nan,
                        "value": float(statistic),
                    }
                )
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(summary)], ignore_index=True)
    logger.info(
        "composition shift: %d rows for AOI %s (2026 only; %d detections dropped as "
        "outside the baseline/event windows)",
        len(out),
        aoi_key,
        dropped_outside,
    )
    return out


# ---------------------------------------------------------------------------
# B. pipeline-transition falsification
# ---------------------------------------------------------------------------
def standardised_rate_change(
    before: pd.DataFrame,
    after: pd.DataFrame,
    bins: Sequence[float],
    min_per_bin: int,
    logger,
) -> dict[str, Any]:
    """Length-standardised matching-rate change, standardised to the 'before' mix.

    Only bins with at least ``min_per_bin`` classified detections in **both**
    periods are used for the standardised figure; the rest are reported as
    excluded so a reader can see how much of the sample the standardisation
    actually covers.
    """
    used: list[str] = []
    excluded: list[str] = []
    bin_stats: dict[str, dict[str, float]] = {}
    for left, right in zip(bins[:-1], bins[1:]):
        label = f"[{int(left)},{int(right)})"
        pre_group = before[(before["length_m"] >= left) & (before["length_m"] < right)]
        post_group = after[(after["length_m"] >= left) & (after["length_m"] < right)]
        pre_den = int(pre_group["matched_flag"].notna().sum())
        post_den = int(post_group["matched_flag"].notna().sum())
        bin_stats[label] = {
            "rate_before": float(pre_group["matched_flag"].mean()) if pre_den else np.nan,
            "rate_after": float(post_group["matched_flag"].mean()) if post_den else np.nan,
            "n_before": pre_den,
            "n_after": post_den,
        }
        if pre_den >= min_per_bin and post_den >= min_per_bin:
            used.append(label)
        else:
            excluded.append(label)
    if used:
        weights = np.array([bin_stats[label]["n_before"] for label in used], dtype=float)
        weights = weights / weights.sum()
        rate_before_std = float(np.sum(weights * np.array([bin_stats[label]["rate_before"] for label in used])))
        rate_after_std = float(np.sum(weights * np.array([bin_stats[label]["rate_after"] for label in used])))
    else:
        rate_before_std = rate_after_std = np.nan
    result: dict[str, Any] = {
        "standardised_rate_before": rate_before_std,
        "standardised_rate_after": rate_after_std,
        "standardised_delta_pp": 100.0 * (rate_after_std - rate_before_std)
        if np.isfinite(rate_before_std) and np.isfinite(rate_after_std)
        else np.nan,
        "standardised_bins_used": ",".join(used),
        "standardised_bins_excluded": ",".join(excluded),
        "standardised_share_of_before_sample": float(
            sum(bin_stats[label]["n_before"] for label in used)
            / max(sum(stats["n_before"] for stats in bin_stats.values()), 1)
        ),
    }
    for label, stats in bin_stats.items():
        key = "rate_len_" + label.strip("[])").replace(",", "_")
        result[key + "_before"] = stats["rate_before"]
        result[key + "_after"] = stats["rate_after"]
        result[key + "_n_before"] = stats["n_before"]
        result[key + "_n_after"] = stats["n_after"]
    logger.info(
        "standardised change %.2f pp (bins used: %s; excluded: %s)",
        result["standardised_delta_pp"] if np.isfinite(result["standardised_delta_pp"]) else float("nan"),
        ",".join(used) or "none",
        ",".join(excluded) or "none",
    )
    return result


def monthly_falsification_row(
    label: str,
    unit_type: str,
    detections: pd.DataFrame,
    before_month: str,
    after_month: str,
    cfg: dict[str, Any],
    logger,
    n_scenes_lookup: dict[str, int] | None = None,
) -> dict[str, Any]:
    """One February-vs-March comparison for a single unit."""
    frame = detections.copy()
    frame["month_key"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m")
    before = frame[frame["month_key"] == before_month]
    after = frame[frame["month_key"] == after_month]
    classified_before = before[before["matched_flag"].notna()].copy()
    classified_after = after[after["matched_flag"].notna()].copy()
    # Scene-level clustering: aggregate to (scene_id) before resampling so the
    # interval reflects scene-level sampling, not detection-level noise.
    pre_scene = classified_before.groupby("scene_id")["matched_flag"].agg(["sum", "size"])
    post_scene = classified_after.groupby("scene_id")["matched_flag"].agg(["sum", "size"])
    interval = bootstrap_rate_difference(
        pre_scene["sum"].to_numpy(dtype=float),
        pre_scene["size"].to_numpy(dtype=float),
        post_scene["sum"].to_numpy(dtype=float),
        post_scene["size"].to_numpy(dtype=float),
        iterations=int(cfg["stats"]["bootstrap_iterations"]),
        level=float(cfg["stats"]["bootstrap_ci_level"]),
        seed=int(cfg["project"]["random_seed"]),
        min_units_for_ci=int(cfg["stats"].get("min_scenes_per_period", 3)),
    )
    bins = [float(value) for value in cfg["identification"]["length_bins_m"]]
    row: dict[str, Any] = {
        "unit": label,
        "unit_type": unit_type,
        "before_month": before_month,
        "after_month": after_month,
        "n_detections_before": int(len(before)),
        "n_detections_after": int(len(after)),
        "n_classified_before": int(len(classified_before)),
        "n_classified_after": int(len(classified_after)),
        "n_matched_before": int(before["matched_flag"].sum()),
        "n_matched_after": int(after["matched_flag"].sum()),
        "n_unmatched_before": int(before["unmatched_flag"].sum()),
        "n_unmatched_after": int(after["unmatched_flag"].sum()),
        "pooled_rate_before": float(classified_before["matched_flag"].mean())
        if len(classified_before) else np.nan,
        "pooled_rate_after": float(classified_after["matched_flag"].mean())
        if len(classified_after) else np.nan,
        "delta_pp": 100.0 * interval.estimate if np.isfinite(interval.estimate) else np.nan,
        "delta_ci_low_pp": 100.0 * interval.low,
        "delta_ci_high_pp": 100.0 * interval.high,
        "ci_computed": interval.ci_computed,
    }
    if n_scenes_lookup is not None:
        row["n_scenes_before"] = n_scenes_lookup.get(f"{label}|{before_month}", np.nan)
        row["n_scenes_after"] = n_scenes_lookup.get(f"{label}|{after_month}", np.nan)
    else:
        row["n_scenes_before"] = int(before["scene_id"].nunique())
        row["n_scenes_after"] = int(after["scene_id"].nunique())
    row["detections_per_scene_before"] = (
        row["n_detections_before"] / row["n_scenes_before"]
        if row["n_scenes_before"] else np.nan
    )
    row["detections_per_scene_after"] = (
        row["n_detections_after"] / row["n_scenes_after"]
        if row["n_scenes_after"] else np.nan
    )
    row.update(
        standardised_rate_change(
            classified_before.assign(matched_flag=classified_before["matched_flag"]),
            classified_after.assign(matched_flag=classified_after["matched_flag"]),
            bins,
            int(cfg["identification"]["small_sample_min_detections"]),
            logger,
        )
    )
    return row


def load_global_month(month_key: str, cfg: dict[str, Any], logger) -> pd.DataFrame:
    """Load one archive-wide monthly detection file with the needed columns."""
    selected, _ = discover_sar_files(cfg, logger=None, kind="detections")
    month = pd.Timestamp(f"{month_key}-01")
    candidates = selected[selected["yyyymm"] == month.strftime("%Y%m")]
    if candidates.empty:
        logger.warning("no global detection file for %s", month_key)
        return pd.DataFrame()
    path = Path(candidates.iloc[0]["path"])
    columns = ["scene_id", "timestamp", "lat", "lon", "length_m", "matched_category"]
    frame = read_csv_fast(path, columns)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"].astype(str).str.replace(" UTC", "", regex=False), errors="coerce"
    )
    frame["matched_flag"] = frame["matched_category"].astype("string").str.strip().str.lower().ne("unmatched")
    frame.loc[frame["matched_category"].isna(), "matched_flag"] = pd.NA
    frame["unmatched_flag"] = (
        frame["matched_category"].astype("string").str.strip().str.lower().eq("unmatched")
    )
    frame["layer"] = "global"
    logger.info(
        "loaded global %s: %d detections, %d scenes", month_key, len(frame), frame["scene_id"].nunique()
    )
    return frame


def assign_regions(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Assign detections to fixed coarse regions (and the corridor boxes)."""
    out = frame.copy()
    out["region"] = "other"
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for name, spec in cfg["identification"]["falsification"]["regions"].items():
        boxes[name] = (spec["lon_min"], spec["lat_min"], spec["lon_max"], spec["lat_max"])
    for aoi_key in list_aois(cfg):
        if resolve_aoi(cfg, aoi_key)[1].get("role") != "technical_reference_corridor":
            continue
        min_lon, min_lat, max_lon, max_lat = aoi_geometry(cfg, aoi_key).bounds
        boxes[f"corridor_box:{aoi_key}"] = (min_lon, min_lat, max_lon, max_lat)
    for name, (min_lon, min_lat, max_lon, max_lat) in boxes.items():
        mask = (
            (out["region"] == "other")
            & out["lon"].between(min_lon, max_lon)
            & out["lat"].between(min_lat, max_lat)
        )
        out.loc[mask, "region"] = name
    return out


def load_aoi_detections(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load AOI detections restricted to the primary analysis sample."""
    metrics_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet", aoi_key, cfg
    )
    metrics = pd.read_parquet(metrics_path)
    keep = set(metrics.loc[metrics["usable_for_primary_metrics"], "scene_id"])
    frame = pd.read_parquet(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet", aoi_key, cfg
        )
    )
    frame = frame[frame["scene_id"].isin(keep)].copy()
    frame["matched_flag"] = frame["matched_flag"].astype("boolean")
    frame["layer"] = aoi_key
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("09_identification_checks", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 9 (Phase 3) - identification and robustness checks")

    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    falsification_cfg = cfg["identification"]["falsification"]

    # ---- A. composition diagnostics --------------------------------------
    composition_frames: list[pd.DataFrame] = []
    for aoi_key in list_aois(cfg):
        try:
            detections = load_aoi_detections(cfg, aoi_key, logger)
        except FileNotFoundError as error:
            logger.warning("skipping composition diagnostics for %s: %s", aoi_key, error)
            continue
        composition_frames.append(composition_shift(detections, cfg, aoi_key, logger))
    composition = pd.concat(composition_frames, ignore_index=True)
    composition.to_csv(tables / "composition_shift.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "composition_shift.csv", len(composition))

    # ---- B. falsification -------------------------------------------------
    rows: list[dict[str, Any]] = []
    comparison_pairs = [
        (falsification_cfg["before_month"], falsification_cfg["after_month"], "primary_break"),
    ]
    for before, after in falsification_cfg.get("placebo_pairs", []):
        comparison_pairs.append((before, after, "placebo_transition"))

    # 6B. external corridors + the two Hormuz AOIs, via the identical AOI pipeline
    aoi_units = ["hormuz_core", "hormuz_narrow_strait"] + [
        key for key in list_aois(cfg)
        if resolve_aoi(cfg, key)[1].get("role") == "technical_reference_corridor"
    ]
    for aoi_key in aoi_units:
        try:
            detections = load_aoi_detections(cfg, aoi_key, logger)
        except FileNotFoundError as error:
            logger.warning("skipping falsification for %s: %s", aoi_key, error)
            continue
        unit_type = (
            "hormuz_aoi" if aoi_key.startswith("hormuz")
            else "technical_reference_corridor"
        )
        for before_month, after_month, comparison in comparison_pairs:
            row = monthly_falsification_row(
                aoi_key, unit_type, detections, before_month, after_month, cfg, logger
            )
            row["comparison"] = comparison
            row["layer"] = "aoi_pipeline"
            rows.append(row)

    # 6A. global and regional diagnostic from the archive-wide files
    if not args.skip_global:
        months = sorted({month for pair in comparison_pairs for month in pair[:2]})
        global_frames = [load_global_month(month, cfg, logger) for month in months]
        global_frames = [frame for frame in global_frames if not frame.empty]
        if global_frames:
            global_frame = assign_regions(pd.concat(global_frames, ignore_index=True), cfg)
            for before_month, after_month, comparison in comparison_pairs:
                scoped = global_frame[
                    global_frame["timestamp"].dt.strftime("%Y-%m").isin([before_month, after_month])
                ]
                for unit, unit_type in [("global_all_regions", "global_archive")]:
                    row = monthly_falsification_row(
                        unit, unit_type, scoped, before_month, after_month, cfg, logger
                    )
                    row["comparison"] = comparison
                    row["layer"] = "global_archive"
                    rows.append(row)
                for region in sorted(scoped["region"].unique()):
                    region_frame = scoped[scoped["region"] == region]
                    if len(region_frame) < 500:
                        logger.info("region %s skipped for %s (n=%d)", region, before_month, len(region_frame))
                        continue
                    row = monthly_falsification_row(
                        f"region:{region}", "global_region", region_frame,
                        before_month, after_month, cfg, logger,
                    )
                    row["comparison"] = comparison
                    row["layer"] = "global_archive"
                    rows.append(row)

    falsification = pd.DataFrame(rows)
    falsification.to_csv(tables / "pipeline_falsification.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "pipeline_falsification.csv", len(falsification))

    headline = falsification[
        (falsification["comparison"] == "primary_break")
        & (falsification["after_month"] == falsification_cfg["after_month"])
    ][["unit", "unit_type", "n_classified_before", "n_classified_after",
       "pooled_rate_before", "pooled_rate_after", "delta_pp", "delta_ci_low_pp",
       "delta_ci_high_pp", "standardised_delta_pp", "standardised_bins_used"]]
    logger.info("Feb -> Mar 2026 matching-rate change by unit:\n%s", headline.round(3).to_string(index=False))

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "identification_checks_summary.json",
        {
            "primary_break": {
                "break_date": falsification_cfg["break_date"],
                "before_month": falsification_cfg["before_month"],
                "after_month": falsification_cfg["after_month"],
            },
            "placebo_pairs": falsification_cfg.get("placebo_pairs", []),
            "units": falsification["unit"].unique().tolist(),
            "primary_break_deltas": headline.to_dict(orient="records"),
            "spatial_cell_degrees": float(cfg["identification"]["spatial_cell_degrees"]),
            "length_bins_m": cfg["identification"]["length_bins_m"],
            "small_sample_min_detections": int(cfg["identification"]["small_sample_min_detections"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
