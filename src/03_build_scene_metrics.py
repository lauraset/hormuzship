"""STEP 3 - scene-level SAR metrics for the Hormuz AOI.

Inputs (produced by step 01)
----------------------------
``data/interim/sar_scenes.parquet``
``data/interim/sar_detections_aoi.parquet``
``data/interim/sar_daily_aoi_coverage.parquet``

Outputs
-------
``data/processed/sar_scene_metrics.parquet``
    One row per SAR scene: coverage, counts, matching rates, coverage-normalised
    SAR density, platform, pipeline, ISO week, coverage-threshold flag.
``data/processed/sar_weekly.parquet``
    Weekly panel built from scene rows only: a week without any SAR acquisition
    is **absent** (NA), never zero. Includes the weekly median SAR density, the
    number of scenes, and the covered AOI area.
``outputs/tables/data_coverage.csv``
    Month x year coverage table (required deliverable).
``outputs/tables/scene_metrics_preview.csv``
    Small human-readable preview of the scene table.
``outputs/tables/detection_composition_by_month.csv``
    Detection composition by GFW match label (needed for technical-caveat checks).
``outputs/diagnostics/scene_count_anomalies.csv``
    Scenes where ``n_total != n_matched + n_unmatched`` (unknown labels present).
``outputs/diagnostics/coverage_threshold_sensitivity.csv``
    How the sample and the pooled matching rate change with the scene coverage
    threshold - the evidence used to accept/reject
    ``sar.min_scene_coverage_fraction``.
``outputs/diagnostics/scene_metrics_ledger.csv``
    Every filter applied here, with before/after/removed/reason.

Usage
-----
    python src/03_build_scene_metrics.py
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
    FilterLedger,
    add_calendar_fields,
    aoi_scoped_path,
    banner,
    discover_sar_files,
    ensure_dirs,
    files_in_window,
    get_logger,
    label_or_na,
    load_config,
    list_aois,
    parse_gfw_datetimes,
    primary_aoi_key,
    read_csv_fast,
    resolve_path,
    safe_divide,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build scene-level SAR metrics for the AOI")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aoi", default=None,
                        help="configured AOI key (default: the primary AOI)")
    parser.add_argument("--start", default=None, help="override analysis.load_start")
    parser.add_argument("--end", default=None, help="override analysis.load_end")
    parser.add_argument("--event-date", default=None,
                        help="event date used for the technical_period label "
                             "(default: analysis.event_date)")
    parser.add_argument("--window-tag", default=None,
                        help="read/write the window-tagged artifacts produced by "
                             "src/01_prepare_sar.py --window-tag")
    return parser.parse_args(argv)


def build_scene_metrics(
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    cfg: dict[str, Any],
    event_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join AOI detections onto scenes and derive rates and densities.

    Scenes with zero AOI detections are kept: a scene that imaged the AOI and
    found nothing is a genuine zero, which is fundamentally different from a
    date with no acquisition at all.
    """
    density_scale = float(cfg["sar"]["sar_density_scale_km2"])

    # Step 01 already adds calendar fields, but step 03 must not silently depend
    # on that (e.g. when it is called on a hand-built frame in a unit test).
    if "iso_week" not in scenes.columns or "week_start" not in scenes.columns:
        scenes = add_calendar_fields(scenes, "scene_date_parsed")

    grouped = detections.groupby("scene_id").agg(
        n_total=("scene_id", "size"),
        n_matched=("matched_flag", "sum"),
        n_unmatched=("unmatched_flag", "sum"),
        n_unknown=("unknown_flag", "sum"),
        n_matched_mmsi=("mmsi_matched_flag", "sum"),
        median_matching_score=("matching_score", "median"),
        median_length_m=("length_m", "median"),
        median_presence_score=("presence_score", "median"),
        first_detection=("timestamp", "min"),
        last_detection=("timestamp", "max"),
    )
    grouped["n_total"] = grouped["n_total"].astype("int64")
    for column in ("n_matched", "n_unmatched", "n_unknown", "n_matched_mmsi"):
        grouped[column] = grouped[column].astype("int64")

    metrics = scenes.merge(
        grouped,
        how="left",
        left_on="scene_id",
        right_index=True,
        validate="one_to_one",
    )
    count_columns = ["n_total", "n_matched", "n_unmatched", "n_unknown", "n_matched_mmsi"]
    metrics[count_columns] = metrics[count_columns].fillna(0).astype("int64")

    # Densities use the *covered* area of the AOI, not the AOI area: a scene that
    # only imaged 40% of the AOI can only be expected to see 40% of the vessels.
    metrics["sar_density"] = safe_divide(metrics["n_total"], metrics["coverage_km2"]) * density_scale
    metrics["sar_density_matched"] = (
        safe_divide(metrics["n_matched"], metrics["coverage_km2"]) * density_scale
    )
    metrics["sar_density_unmatched"] = (
        safe_divide(metrics["n_unmatched"], metrics["coverage_km2"]) * density_scale
    )
    metrics["matching_rate"] = safe_divide(metrics["n_matched"], metrics["n_total"])
    metrics["unmatched_rate"] = safe_divide(metrics["n_unmatched"], metrics["n_total"])
    metrics["unknown_rate"] = safe_divide(metrics["n_unknown"], metrics["n_total"])
    metrics["matching_rate_known"] = safe_divide(
        metrics["n_matched"], metrics["n_matched"] + metrics["n_unmatched"]
    )
    metrics["matching_rate_mmsi"] = safe_divide(metrics["n_matched_mmsi"], metrics["n_total"])

    # explicit integrity flag: unknown labels break the partition identity
    metrics["count_identity_ok"] = metrics["n_total"].eq(
        metrics["n_matched"] + metrics["n_unmatched"]
    )
    metrics["has_detections"] = metrics["n_total"] > 0
    metrics["passes_coverage_threshold"] = metrics["coverage_fraction"] >= float(
        cfg["sar"]["min_scene_coverage_fraction"]
    )
    metrics["coverage_km2_ge_min"] = metrics["coverage_km2"] >= float(
        cfg["sar"]["min_scene_coverage_km2"]
    )
    # ``technical_period`` is a labelling convenience only. It must follow the
    # event being studied, so an explicit date can be supplied for non-Hormuz runs
    # instead of silently labelling a 2021 or 2023 dataset with the 2026 date.
    reference_event_date = pd.Timestamp(event_date) if event_date is not None else pd.Timestamp(
        cfg["analysis"]["event_date"]
    )
    metrics["technical_period"] = np.select(
        [
            metrics["scene_date_parsed"] <= reference_event_date,
            metrics["scene_date_parsed"] < pd.Timestamp("2026-03-28"),
            metrics["scene_date_parsed"] >= pd.Timestamp("2026-03-28"),
        ],
        ["pre_event_or_control", "event_acute_mask_pre", "event_persistent_mask_post"],
        default="unknown",
    )
    return metrics


def build_weekly_panel(scene_metrics: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Aggregate scene metrics into an ISO-week SAR panel.

    Weeks without a single SAR acquisition are absent from the output; callers
    must reindex onto a full calendar if they need a complete time axis, and the
    missing entries stay NA (never 0).
    """
    usable = scene_metrics[scene_metrics["usable_for_primary_metrics"]].copy()
    if usable.empty:
        return pd.DataFrame()
    grouped = usable.groupby(["iso_year", "iso_week", "week_key", "week_start"], as_index=False).agg(
        n_scenes=("scene_id", "size"),
        n_scenes_with_detections=("has_detections", "sum"),
        sar_density_median=("sar_density", "median"),
        sar_density_mean=("sar_density", "mean"),
        sar_density_min=("sar_density", "min"),
        sar_density_max=("sar_density", "max"),
        sar_density_std=("sar_density", "std"),
        n_detections=("n_total", "sum"),
        n_matched=("n_matched", "sum"),
        n_unmatched=("n_unmatched", "sum"),
        coverage_km2_sum=("coverage_km2", "sum"),
        coverage_km2_median=("coverage_km2", "median"),
        matching_rate_scene_median=("matching_rate", "median"),
        matching_rate_scene_mean=("matching_rate", "mean"),
        platforms=("satellite_platform", lambda values: ",".join(sorted(set(values)))),
        pipelines=("pipeline", lambda values: ",".join(sorted(set(values)))),
    )
    grouped["matching_rate_pooled"] = grouped["n_matched"] / (
        grouped["n_matched"] + grouped["n_unmatched"]
    ).replace(0, np.nan)
    grouped["sar_density_coverage_weighted"] = grouped["n_detections"] / grouped["coverage_km2_sum"] * float(
        cfg["sar"]["sar_density_scale_km2"]
    )
    return grouped.sort_values("week_start").reset_index(drop=True)


def build_monthly_coverage(
    scene_metrics: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Month x year coverage table (the required ``data_coverage.csv``)."""
    usable = scene_metrics[scene_metrics["usable_for_primary_metrics"]]
    base = (
        scene_metrics.groupby(["year", "month"], as_index=False)
        .agg(
            sar_scenes_all=("scene_id", "size"),
            sar_scenes_unfiltered=("scene_id", "size"),
            footprint_area_km2=("footprint_area_km2", "sum"),
        )
    )
    usable_agg = usable.groupby(["year", "month"], as_index=False).agg(
        sar_scenes=("scene_id", "size"),
        sar_scenes_with_detections=("has_detections", "sum"),
        detections=("n_total", "sum"),
        matched=("n_matched", "sum"),
        unmatched=("n_unmatched", "sum"),
        unknown=("n_unknown", "sum"),
        observed_area_km2=("coverage_km2", "sum"),
        coverage_km2_median=("coverage_km2", "median"),
        sar_density_median=("sar_density", "median"),
        matching_rate_scene_median=("matching_rate", "median"),
    )
    table = base.merge(usable_agg, how="left", on=["year", "month"])
    table["matching_rate_pooled"] = table["matched"] / (
        table["matched"] + table["unmatched"]
    ).replace(0, np.nan)
    acquisition_days = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(
            sar_acquisition_days=("date", "size"),
            mean_daily_aoi_coverage_fraction=("aoi_covered_fraction", "mean"),
            max_daily_aoi_coverage_fraction=("aoi_covered_fraction", "max"),
        )
    )
    table = table.merge(acquisition_days, how="left", on=["year", "month"])
    days_in_month = pd.to_datetime(
        dict(year=table["year"], month=table["month"], day=1)
    ) + pd.offsets.MonthEnd(1)
    table["days_in_month"] = days_in_month.dt.day
    table["share_of_days_with_acquisition"] = table["sar_acquisition_days"] / table["days_in_month"]
    # Phase 1 has no AIS yet; the column exists so that step 05 can fill it in
    # without changing the table schema.
    table["ais_valid_days"] = np.nan
    table["ais_valid_days_note"] = "phase_1_not_available"
    columns = [
        "year",
        "month",
        "days_in_month",
        "sar_scenes_all",
        "sar_scenes",
        "sar_scenes_with_detections",
        "sar_acquisition_days",
        "share_of_days_with_acquisition",
        "detections",
        "matched",
        "unmatched",
        "unknown",
        "matching_rate_pooled",
        "matching_rate_scene_median",
        "observed_area_km2",
        "coverage_km2_median",
        "mean_daily_aoi_coverage_fraction",
        "max_daily_aoi_coverage_fraction",
        "sar_density_median",
        "ais_valid_days",
        "ais_valid_days_note",
    ]
    return table[columns].sort_values(["year", "month"]).reset_index(drop=True)


def build_length_stratified_matching(
    detections: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Matching rate by vessel-length bin and calendar month.

    This diagnostic separates two competing explanations for a falling matching
    rate:

    * *composition*: the vessel population changed (large AIS-carrying ships
      left, small non-AIS craft remained), which mechanically lowers the pooled
      matching rate;
    * *observability*: at a given vessel size, an AIS identity became less
      likely to be attached to a SAR detection.

    Only the second explanation is evidence of a change in AIS
    representativeness, so the matching rate must always be read within size
    bins as well as pooled.
    """
    bins = [float(value) for value in cfg["sar"]["matching_rate_length_bins_m"]]
    frame = detections[detections["scene_in_footprints"]].copy()
    frame["year"] = pd.to_datetime(frame["timestamp"]).dt.year
    frame["month"] = pd.to_datetime(frame["timestamp"]).dt.month
    frame["length_bin"] = pd.cut(frame["length_m"], bins=bins, right=True)
    grouped = (
        frame.groupby(["year", "month", "length_bin"], observed=True, as_index=False)
        .agg(
            n_detections=("scene_id", "size"),
            n_matched=("matched_flag", "sum"),
            n_unmatched=("unmatched_flag", "sum"),
            median_length_m=("length_m", "median"),
            median_matching_score=("matching_score", "median"),
        )
    )
    grouped["matching_rate"] = grouped["n_matched"] / (
        grouped["n_matched"] + grouped["n_unmatched"]
    ).replace(0, np.nan)
    grouped["length_bin_left_m"] = grouped["length_bin"].map(lambda interval: interval.left)
    grouped["length_bin_right_m"] = grouped["length_bin"].map(lambda interval: interval.right)
    grouped["length_bin"] = grouped["length_bin"].astype(str)
    return grouped.sort_values(["year", "month", "length_bin_left_m"]).reset_index(drop=True)


def build_threshold_sensitivity(scene_metrics: pd.DataFrame) -> pd.DataFrame:
    """Show how sample size and matching rate move with the coverage threshold."""
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    records = []
    for threshold in thresholds:
        subset = scene_metrics[scene_metrics["coverage_fraction"] >= threshold]
        with_detections = subset[subset["has_detections"]]
        records.append(
            {
                "coverage_threshold": threshold,
                "n_scenes": int(len(subset)),
                "n_scenes_with_detections": int(len(with_detections)),
                "n_detections": int(subset["n_total"].sum()),
                "n_matched": int(subset["n_matched"].sum()),
                "n_unmatched": int(subset["n_unmatched"].sum()),
                "matching_rate_pooled": (
                    float(subset["n_matched"].sum())
                    / float(subset["n_matched"].sum() + subset["n_unmatched"].sum())
                    if (subset["n_matched"].sum() + subset["n_unmatched"].sum()) > 0
                    else np.nan
                ),
                "matching_rate_scene_median": float(with_detections["matching_rate"].median())
                if len(with_detections)
                else np.nan,
                "median_sar_density": float(subset["sar_density"].median()),
            }
        )
    return pd.DataFrame(records)


def build_platform_diagnostic(
    scene_metrics: pd.DataFrame,
    cfg: dict[str, Any],
    logger,
) -> tuple[pd.DataFrame, list[str]]:
    """Detect platforms that appear in the footprint product but never in the detections.

    GFW ships its footprint product and its detection product separately. When a
    new satellite joins the constellation, footprints can appear before (or
    without) any matched detections. Such scenes show ``n_total == 0``, which is
    *not* an observation of "zero vessels" and must not enter density metrics.

    Returns
    -------
    (pandas.DataFrame, list[str])
        The per-platform diagnostic table and the list of platforms to exclude.
    """
    diagnostic = (
        scene_metrics.groupby("satellite_platform", as_index=False)
        .agg(
            n_scenes=("scene_id", "size"),
            n_scenes_above_coverage_threshold=("passes_coverage_threshold", "sum"),
            n_detections=("n_total", "sum"),
            median_coverage_km2=("coverage_km2", "median"),
            median_coverage_fraction=("coverage_fraction", "median"),
            first_scene_date=("scene_date_parsed", "min"),
            last_scene_date=("scene_date_parsed", "max"),
        )
        .sort_values("satellite_platform")
        .reset_index(drop=True)
    )
    diagnostic["detections_per_scene"] = diagnostic["n_detections"] / diagnostic["n_scenes"]
    diagnostic["interpretation"] = np.where(
        diagnostic["n_detections"] == 0,
        "footprint-only platform: no detections at all in the archive window",
        "has detections",
    )
    min_scenes = int(cfg["sar"].get("platform_diagnostic_min_scenes", 3))
    to_exclude: list[str] = []
    if cfg["sar"].get("exclude_platforms_without_detections", True):
        to_exclude = diagnostic.loc[
            (diagnostic["n_detections"] == 0) & (diagnostic["n_scenes"] >= min_scenes),
            "satellite_platform",
        ].tolist()
        for platform in to_exclude:
            row = diagnostic[diagnostic["satellite_platform"] == platform].iloc[0]
            logger.warning(
                "platform %s appears in %d footprint(s) between %s and %s but has 0 detections: "
                "excluded from primary metrics (footprint-only acquisition)",
                platform,
                int(row["n_scenes"]),
                str(pd.Timestamp(row["first_scene_date"]).date()),
                str(pd.Timestamp(row["last_scene_date"]).date()),
            )
    return diagnostic, to_exclude


def build_global_archive_label_check(
    cfg: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Compute the archive-wide GFW match-label share per month and pipeline.

    Rationale
    ---------
    The AOI matching rate can move for two very different reasons: (i) vessels
    really stop being observable in AIS, or (ii) GFW changed its matching
    pipeline. Reading the *global* monthly detection files separates those two
    hypotheses, because a pipeline change moves the global share in the same
    month, whereas a local disruption does not.

    This is a measurement-validity diagnostic. It is deliberately **not** used as
    a causal control region: the archive-wide share also responds to global
    coverage and composition changes.
    """
    load_start = pd.Timestamp(cfg["analysis"]["load_start"])
    load_end = pd.Timestamp(cfg["analysis"]["load_end"])
    selected, _ = discover_sar_files(cfg, logger=None, kind="detections")
    detection_files = files_in_window(selected, load_start, load_end, kind="detections")
    match_column = "matched_category"
    length_column = "length_m"
    large_vessel_threshold_m = 200.0

    records: list[dict[str, Any]] = []
    for row in detection_files.itertuples(index=False):
        path = Path(row.path)
        frame = read_csv_fast(path, ["scene_id", match_column, length_column])
        labels = frame[match_column].astype("string").str.strip().str.lower()
        total = len(frame)
        n_unmatched = int(labels.eq("unmatched").sum())
        n_unknown = int(labels.isna().sum() + labels.isin(["", "na", "nan", "none", "null"]).sum())
        platforms = frame["scene_id"].str.split("_").str[0].value_counts()
        # Same statistic restricted to large vessels: this is the archive-wide
        # counterpart of the AOI result that big ships stop being matched.
        lengths = pd.to_numeric(frame[length_column], errors="coerce")
        large = lengths >= large_vessel_threshold_m
        large_labels = labels[large]
        n_large = int(large.sum())
        n_large_unmatched = int(large_labels.eq("unmatched").sum())
        records.append(
            {
                "yyyymm": row.yyyymm,
                "pipeline": label_or_na(row.pipeline),
                "n_detections": total,
                "n_unmatched": n_unmatched,
                "n_unknown_label": n_unknown,
                "unmatched_share": n_unmatched / total if total else np.nan,
                "n_scenes": int(frame["scene_id"].nunique()),
                "n_s1a": int(platforms.get("S1A", 0)),
                "n_s1b": int(platforms.get("S1B", 0)),
                "n_s1c": int(platforms.get("S1C", 0)),
                "n_s1d": int(platforms.get("S1D", 0)),
                "n_detections_len_ge_200m": n_large,
                "unmatched_share_len_ge_200m": (
                    n_large_unmatched / n_large if n_large else np.nan
                ),
                "file_name": row.file_name,
            }
        )
        logger.info(
            "global label check %s (pipeline %s): unmatched share %.4f over %d detections",
            row.yyyymm,
            label_or_na(row.pipeline),
            records[-1]["unmatched_share"],
            total,
        )
    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.window_tag:
        # A single config key routes every aoi_scoped_path call in this run, so a
        # window-tagged input is read and the outputs stay tagged too.
        cfg["paths"]["window_tag"] = args.window_tag
    logger = get_logger("03_build_scene_metrics", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 3 - scene-level SAR metrics, matching rates and densities")

    interim = resolve_path(cfg["paths"]["interim_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    tables = resolve_path(cfg["paths"]["tables_dir"])
    diagnostics = resolve_path(cfg["paths"]["diagnostics_dir"])
    for directory in (processed, tables, diagnostics):
        directory.mkdir(parents=True, exist_ok=True)

    aoi_key = args.aoi or primary_aoi_key(cfg)
    if args.aoi is not None and args.aoi not in list_aois(cfg):
        logger.error("unknown AOI '%s'; configured AOIs: %s", args.aoi, list_aois(cfg))
        return 2
    banner(logger, f"STEP 3 - scene metrics for AOI '{aoi_key}'")

    scenes = pd.read_parquet(aoi_scoped_path(interim / "sar_scenes.parquet", aoi_key, cfg))
    detections = pd.read_parquet(
        aoi_scoped_path(interim / "sar_detections_aoi.parquet", aoi_key, cfg)
    )
    daily = pd.read_parquet(
        aoi_scoped_path(interim / "sar_daily_aoi_coverage.parquet", aoi_key, cfg)
    )
    logger.info("loaded %d scenes, %d AOI detections, %d acquisition dates", len(scenes), len(detections), len(daily))

    ledger = FilterLedger(name="build_scene_metrics")

    # Detections whose scene has no footprint row cannot enter scene metrics.
    before = len(detections)
    orphans = detections[~detections["scene_in_footprints"]]
    ledger.record(
        step="orphan_detections_excluded",
        before=before,
        after=before - len(orphans),
        reason="scene_id not present in the footprint table, so coverage/density is undefined",
        n_orphan_scenes=int(orphans["scene_id"].nunique()) if len(orphans) else 0,
    )
    model_detections = detections[detections["scene_in_footprints"]].copy()

    metrics = build_scene_metrics(scenes, model_detections, cfg, event_date=args.event_date)

    # ---- input-validity diagnostics --------------------------------------
    platform_diagnostic, unsupported_platforms = build_platform_diagnostic(metrics, cfg, logger)
    platform_diagnostic.to_csv(
        aoi_scoped_path(diagnostics / "platform_detection_coverage.csv", aoi_key, cfg), index=False
    )
    metrics["platform_has_detections"] = ~metrics["satellite_platform"].isin(unsupported_platforms)
    before_platform = len(metrics)
    ledger.record(
        step="platform_without_detections_excluded",
        before=before_platform,
        after=int(metrics["platform_has_detections"].sum()),
        reason="footprint-only platform with zero detections in the whole window",
        excluded_platforms=",".join(unsupported_platforms) if unsupported_platforms else "",
    )
    metrics["usable_for_primary_metrics"] = (
        metrics["passes_coverage_threshold"] & metrics["platform_has_detections"]
    )

    metrics["aoi_key"] = aoi_key
    metrics_path = aoi_scoped_path(processed / "sar_scene_metrics.parquet", aoi_key, cfg)
    metrics.to_parquet(metrics_path, index=False)
    logger.info("wrote %s (%d scenes)", metrics_path, len(metrics))

    # ---- integrity checks ------------------------------------------------
    anomalies = metrics[~metrics["count_identity_ok"]]
    anomalies[
        [
            "scene_id",
            "scene_date_parsed",
            "pipeline",
            "coverage_fraction",
            "n_total",
            "n_matched",
            "n_unmatched",
            "n_unknown",
            "unknown_rate",
        ]
    ].to_csv(
        aoi_scoped_path(diagnostics / "scene_count_anomalies.csv", aoi_key, cfg), index=False
    )
    if len(anomalies):
        logger.warning(
            "count identity n_total == n_matched + n_unmatched failed for %d scene(s) "
            "(%.2f%% of detections carry an unknown match label); see scene_count_anomalies.csv",
            len(anomalies),
            100.0 * anomalies["n_unknown"].sum() / max(metrics["n_total"].sum(), 1),
        )
    else:
        logger.info("count identity n_total == n_matched + n_unmatched holds for all scenes")

    # ---- threshold sensitivity + filters ---------------------------------
    sensitivity = build_threshold_sensitivity(metrics)
    sensitivity.to_csv(
        aoi_scoped_path(diagnostics / "coverage_threshold_sensitivity.csv", aoi_key, cfg),
        index=False,
    )
    threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    kept = metrics[metrics["usable_for_primary_metrics"]]
    ledger.record(
        step="primary_analysis_sample",
        before=len(metrics),
        after=len(kept),
        reason=(
            f"coverage_fraction < sar.min_scene_coverage_fraction ({threshold}) "
            "or platform has no detections in the archive"
        ),
    )
    logger.info(
        "coverage threshold %.2f keeps %d/%d scenes (%.1f%%) carrying %d detections",
        threshold,
        len(kept),
        len(metrics),
        100.0 * len(kept) / max(len(metrics), 1),
        int(kept["n_total"].sum()),
    )
    ledger.log_summary(logger)

    # ---- monthly coverage table ------------------------------------------
    coverage = build_monthly_coverage(metrics, daily, cfg)
    coverage_path = aoi_scoped_path(tables / "data_coverage.csv", aoi_key, cfg)
    coverage.to_csv(coverage_path, index=False)
    logger.info("wrote %s", coverage_path)

    # ---- weekly panel ----------------------------------------------------
    weekly = build_weekly_panel(metrics, cfg)
    weekly_path = aoi_scoped_path(processed / "sar_weekly.parquet", aoi_key, cfg)
    weekly.to_parquet(weekly_path, index=False)
    logger.info("wrote %s (%d weeks with at least one usable scene)", weekly_path, len(weekly))

    # ---- detection composition ------------------------------------------
    composition = (
        model_detections.assign(
            year=model_detections["timestamp"].dt.year,
            month=model_detections["timestamp"].dt.month,
        )
        .groupby(["year", "month", "match_label"], dropna=False, as_index=False)
        .agg(n_detections=("scene_id", "size"), median_length_m=("length_m", "median"))
    )
    totals = composition.groupby(["year", "month"])["n_detections"].transform("sum")
    composition["share_of_month"] = composition["n_detections"] / totals
    composition_path = aoi_scoped_path(
        tables / "detection_composition_by_month.csv", aoi_key, cfg
    )
    composition.to_csv(composition_path, index=False)
    logger.info("wrote %s", composition_path)

    # ---- matching rate by vessel size ------------------------------------
    length_stratified = build_length_stratified_matching(model_detections, cfg)
    length_stratified_path = aoi_scoped_path(
        tables / "matching_rate_by_length_bin.csv", aoi_key, cfg
    )
    length_stratified.to_csv(length_stratified_path, index=False)
    logger.info("wrote %s", length_stratified_path)

    # ---- preview ---------------------------------------------------------
    preview_columns = [
        "scene_id",
        "scene_date_parsed",
        "pipeline",
        "satellite_platform",
        "coverage_km2",
        "coverage_fraction",
        "n_total",
        "n_matched",
        "n_unmatched",
        "n_unknown",
        "matching_rate",
        "matching_rate_mmsi",
        "sar_density",
        "passes_coverage_threshold",
    ]
    metrics[preview_columns + ["aoi_key"]].to_csv(
        aoi_scoped_path(tables / "scene_metrics_preview.csv", aoi_key, cfg), index=False
    )

    ledger.to_frame().to_csv(
        aoi_scoped_path(diagnostics / "scene_metrics_ledger.csv", aoi_key, cfg), index=False
    )

    # ---- global pipeline-artifact check ----------------------------------
    global_check = pd.DataFrame()
    is_primary = aoi_key == primary_aoi_key(cfg)
    if cfg.get("diagnostics", {}).get("global_archive_label_check", False) and is_primary:
        logger.info(
            "running the archive-wide match-label check (pipeline-artifact test; "
            "AOI-independent, so it runs once per AOI-agnostic pipeline pass)"
        )
        global_check = build_global_archive_label_check(cfg, logger)
        global_check.to_csv(diagnostics / "global_archive_label_share.csv", index=False)
        logger.info("wrote %s", diagnostics / "global_archive_label_share.csv")

    summary = {
        "aoi_key": aoi_key,
        "n_scenes_total": int(len(metrics)),
        "n_scenes_used": int(len(kept)),
        "coverage_threshold": threshold,
        "n_detections_total": int(metrics["n_total"].sum()),
        "n_detections_used": int(kept["n_total"].sum()),
        "pooled_matching_rate_used": float(
            kept["n_matched"].sum() / max(kept["n_matched"].sum() + kept["n_unmatched"].sum(), 1)
        ),
        "n_scenes_with_unknown_labels": int((~metrics["count_identity_ok"]).sum()),
        "n_weeks_with_sar": int(len(weekly)),
        "weeks_min_date": str(weekly["week_start"].min().date()) if len(weekly) else None,
        "weeks_max_date": str(weekly["week_start"].max().date()) if len(weekly) else None,
        "unsupported_platforms_excluded": unsupported_platforms,
    }
    write_json(
        aoi_scoped_path(diagnostics / "scene_metrics_summary.json", aoi_key, cfg), summary
    )
    logger.info("STEP 3 summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
