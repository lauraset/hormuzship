"""STEP 5 (Phase 2) - AIS vs SAR activity anomalies and the observability gap.

Question
--------
Q2: does AIS-derived maritime activity show a *different* disruption magnitude
from independent SAR-derived activity?

Two AIS series are built for every AOI
-------------------------------------
``AIS_all_days``
    Weekly AIS presence using every day on which AIS data exist.
``AIS_SAR_days``
    Weekly AIS presence restricted to calendar days on which at least one usable
    SAR acquisition intersects the AOI (each year restricted to its own
    acquisition days). This is the stricter temporal-alignment test: continuous
    AIS observation and intermittent SAR sampling are made to see the same days.

Both are stored as *presence hours* (primary), plus mean hours per observed day
and the number of observed days, so a week with few observed days can never look
like a quiet week by accident. A week with fewer than
``ais.min_observed_days_per_week`` observed days stays NaN and is flagged.

Normalisation
-------------
For each ISO week::

    expected_X    = median(X_2024_same_week, X_2025_same_week)
    X_log_anomaly = log((X_2026_week + eps) / (expected_X + eps))
    observability_assessment_gap = AIS_log_anomaly - SAR_log_anomaly

The gap is an assessment divergence between two independent observation systems.
It is **not** the share of vessels that switched AIS off.

Outputs
-------
``data/processed/ais_weekly.parquet``
``data/processed/sar_weekly_variants.parquet``  (total + vessel-length subsets)
``outputs/tables/activity_anomaly_weekly.csv``
``outputs/tables/phase_summary.csv``
``outputs/tables/ais_vessel_population_sensitivity.csv``
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
    add_calendar_fields,
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    in_window,
    list_aois,
    load_config,
    primary_aoi_key,
    resolve_path,
    resolve_phases,
    safe_divide,
    write_json,
)
from hormuz_stats import bootstrap_log_ratio  # noqa: E402

SERIES_VARIANTS = ("AIS_all_days", "AIS_SAR_days")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AIS vs SAR activity anomalies (Phase 2)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None, help="AOI keys (default: all)")
    return parser.parse_args(argv)


def default_aois(cfg: dict[str, Any]) -> list[str]:
    """AOIs that carry an AIS series.

    The technical reference corridors are SAR-only fixtures for the pipeline
    break test, so they are excluded unless explicitly requested.
    """
    from hormuz_common import resolve_aoi

    keys = []
    for key in list_aois(cfg):
        if resolve_aoi(cfg, key)[1].get("role") == "technical_reference_corridor":
            continue
        keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_ais_daily(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load the per-AOI daily AIS table produced by step 02."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily.parquet", aoi_key, cfg
    )
    if not path.exists():
        logger.warning(
            "missing %s (run `python src/02_fetch_ais.py`); AIS series skipped for %s",
            path.name,
            aoi_key,
        )
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_ais_status(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load the per-day AIS observation status written by step 02."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily_status.parquet", aoi_key, cfg
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; re-run src/02_fetch_ais.py (it classifies every day as "
            "observed_positive / structural_zero / missing_unknown)"
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    counts = frame["ais_observation_status"].value_counts().to_dict()
    logger.info("AOI %s AIS day status: %s", aoi_key, counts)
    return frame


def load_sar_days(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load the per-AOI SAR acquisition days (calendar days with at least one scene)."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_daily_aoi_coverage.parquet", aoi_key, cfg
    )
    if not path.exists():
        logger.warning("missing %s; SAR-day alignment skipped for %s", path.name, aoi_key)
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame[["date", "n_scenes", "aoi_covered_km2", "aoi_covered_fraction"]]


def load_scene_metrics(cfg: dict[str, Any], aoi_key: str, logger) -> pd.DataFrame:
    """Load scene metrics for the AOI, keeping the primary analysis sample."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet", aoi_key, cfg
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run src/01_prepare_sar.py and src/03_build_scene_metrics.py "
            f"--aoi {aoi_key} first"
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    return frame[frame["usable_for_primary_metrics"]].copy()


# ---------------------------------------------------------------------------
# AIS weekly series
# ---------------------------------------------------------------------------
def build_ais_weekly(
    ais_daily: pd.DataFrame,
    sar_days: pd.DataFrame,
    status: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """Build both weekly AIS series for one AOI and every vessel-type selector.

    Returns a long frame with one row per (vessel_type, series, week) holding the
    presence hours, observed-day count and mean hours per observed day.

    Days classified ``structural_zero`` count as observed days with zero activity
    (the report completed and contained no entry for the AOI). Days classified
    ``missing_unknown`` are excluded from both the numerator and the denominator,
    so an unavailable report can never look like a quiet week.
    """
    if ais_daily.empty:
        return pd.DataFrame()
    if status is None or status.empty:
        raise ValueError(
            f"AOI {aoi_key}: the AIS observation-status table is required; "
            "re-run src/02_fetch_ais.py so that zero and missing days are separated"
        )
    frame = ais_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    usable_status = status[
        status["ais_observation_status"].isin(["observed_positive", "structural_zero"])
    ].copy()
    usable_status["date"] = pd.to_datetime(usable_status["date"]).dt.normalize()
    usable_days = set(usable_status["date"])
    structural_zero_days = set(
        usable_status.loc[
            usable_status["ais_observation_status"] == "structural_zero", "date"
        ]
    )
    excluded_days = set(pd.to_datetime(status["date"])) - usable_days
    if excluded_days:
        logger.warning(
            "AOI %s: %d day(s) excluded from the weekly AIS series as missing_unknown",
            aoi_key,
            len(excluded_days),
        )

    # Materialise the structural zeros as explicit zero-activity rows so they
    # count as observed days with zero hours rather than disappearing. A row is
    # created for every (classification, class_value) pair present in the AOI, so
    # a category's day count is not inflated by its own absence.
    category_pairs = frame[["classification", "class_value"]].drop_duplicates()
    zero_rows = pd.DataFrame(
        {
            "aoi_key": aoi_key,
            "date": sorted(structural_zero_days),
            "classification": "all",
            "class_value": "__ALL__",
            "presence_hours": 0.0,
            "n_records": 0,
            "n_vessels": 0,
        }
    )
    if len(structural_zero_days) and len(category_pairs):
        frames_for_zeros = []
        for date in sorted(structural_zero_days):
            block = category_pairs.copy()
            block["aoi_key"] = aoi_key
            block["date"] = date
            block["presence_hours"] = 0.0
            block["n_records"] = 0
            block["n_vessels"] = 0
            frames_for_zeros.append(block)
        zero_rows = pd.concat(frames_for_zeros, ignore_index=True)
    if len(zero_rows):
        frame = pd.concat([frame, zero_rows], ignore_index=True)
    frame = frame[frame["date"].isin(usable_days)]
    frame = add_calendar_fields(frame, "date")

    acquisition_days = set(pd.to_datetime(sar_days["date"])) if len(sar_days) else set()
    frame["is_sar_day"] = frame["date"].isin(acquisition_days)

    records: list[pd.DataFrame] = []
    for variant in SERIES_VARIANTS:
        subset = frame[frame["is_sar_day"]] if variant == "AIS_SAR_days" else frame
        if subset.empty:
            logger.warning("AOI %s: no AIS days for series %s", aoi_key, variant)
            continue
        grouped = (
            subset.groupby(
                [
                    "aoi_key",
                    "classification",
                    "class_value",
                    "week_key",
                    "iso_year",
                    "iso_week",
                    "week_start",
                ],
                as_index=False,
            ).agg(
                presence_hours=("presence_hours", "sum"),
                days_observed=("date", "nunique"),
                n_vessels=("n_vessels", "max"),
            )
        )
        grouped["series"] = variant
        records.append(grouped)
    if not records:
        return pd.DataFrame()

    weekly = pd.concat(records, ignore_index=True)
    weekly["mean_hours_per_observed_day"] = safe_divide(
        weekly["presence_hours"], weekly["days_observed"]
    )
    weekly["days_in_week_expected"] = 7
    weekly["days_excluded_missing_unknown"] = weekly.apply(
        lambda row: sum(
            1
            for date in pd.date_range(
                pd.Timestamp(row["week_start"]), periods=7, freq="D"
            )
            if date.normalize() in excluded_days
        ),
        axis=1,
    )

    # Each series has its own statistic and its own observed-day threshold:
    # a weekly sum needs almost the whole week, whereas a mean per SAR day only
    # needs the day(s) that were actually imaged.
    statistics = {
        "AIS_all_days": (
            str(cfg["ais"].get("weekly_statistic_all_days", "sum")),
            int(cfg["ais"].get("min_observed_days_per_week_all_days", 3)),
        ),
        "AIS_SAR_days": (
            str(cfg["ais"].get("weekly_statistic_sar_days", "mean_observed_day")),
            int(cfg["ais"].get("min_observed_days_per_week_sar_days", 1)),
        ),
    }
    weekly["value_column"] = weekly["series"].map(lambda name: statistics[name][0])
    weekly["min_days"] = weekly["series"].map(lambda name: statistics[name][1])
    weekly["sufficient_days"] = weekly["days_observed"] >= weekly["min_days"]
    weekly["analysis_value"] = np.where(
        weekly["value_column"] == "sum",
        weekly["presence_hours"],
        weekly["mean_hours_per_observed_day"],
    )
    weekly.loc[~weekly["sufficient_days"], "analysis_value"] = np.nan
    logger.info(
        "AOI %s: %d weekly AIS rows (%d below their observed-day threshold); "
        "statistics=%s",
        aoi_key,
        len(weekly),
        int((~weekly["sufficient_days"]).sum()),
        {name: "%s>=%d days" % values for name, values in statistics.items()},
    )
    return weekly.sort_values(
        ["classification", "class_value", "series", "week_start"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SAR weekly series (total + vessel-length subsets)
# ---------------------------------------------------------------------------
def build_sar_weekly_variants(
    scene_metrics: pd.DataFrame,
    detections: pd.DataFrame,
    cfg: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Weekly SAR density for the total fleet and for vessel-length subsets.

    Density is always normalised by the *covered* area of the AOI, so scenes that
    imaged only part of the AOI stay comparable.
    """
    scale = float(cfg["sar"]["sar_density_scale_km2"])
    usable_ids = set(scene_metrics["scene_id"])
    frame = detections[detections["scene_id"].isin(usable_ids)].copy()
    frame["length_m"] = pd.to_numeric(frame["length_m"], errors="coerce")

    subsets = {
        "ALL_detections": frame,
        "length_ge_50m": frame[frame["length_m"] >= 50.0],
        "length_ge_100m": frame[frame["length_m"] >= 100.0],
    }
    scene_columns = [
        "scene_id",
        "week_key",
        "iso_year",
        "iso_week",
        "week_start",
        "coverage_km2",
    ]
    records: list[pd.DataFrame] = []
    for name, subset in subsets.items():
        counts = subset.groupby("scene_id").size().rename("n_subset")
        merged = scene_metrics[scene_columns].merge(
            counts, how="left", left_on="scene_id", right_index=True
        )
        merged["n_subset"] = merged["n_subset"].fillna(0).astype(int)
        merged["density"] = safe_divide(merged["n_subset"], merged["coverage_km2"]) * scale
        weekly = (
            merged.groupby(
                ["week_key", "iso_year", "iso_week", "week_start"], as_index=False
            ).agg(
                n_scenes=("scene_id", "size"),
                n_detections=("n_subset", "sum"),
                density_median=("density", "median"),
                density_mean=("density", "mean"),
                coverage_km2_sum=("coverage_km2", "sum"),
            )
        )
        weekly["density_weighted"] = (
            safe_divide(weekly["n_detections"], weekly["coverage_km2_sum"]) * scale
        )
        weekly["sar_variant"] = name
        records.append(weekly)
    out = pd.concat(records, ignore_index=True)
    logger.info("SAR weekly variants: %d rows across %d variants", len(out), len(subsets))
    return out.sort_values(["sar_variant", "week_start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------
def expected_by_iso_week(
    frame: pd.DataFrame,
    value_column: str,
    reference_years: Sequence[int],
    logger,
) -> pd.DataFrame:
    """Historical expectation per ISO week: median over the reference years.

    The expectation is formed **across years** for the same ISO week
    (``median(2024_same_week, 2025_same_week)``), so the result is keyed by
    ``iso_week`` alone and can be joined onto the event year.
    """
    references = frame[frame["iso_year"].isin(list(reference_years))]
    if references.empty:
        logger.warning("no reference-year rows for column %s", value_column)
        return pd.DataFrame(columns=["iso_week", "expected", "n_reference_years"])
    return (
        references.groupby("iso_week", as_index=False)
        .agg(expected=(value_column, "median"), n_reference_years=(value_column, "count"))
    )


def add_anomaly_columns(
    frame: pd.DataFrame,
    value_column: str,
    expected_frame: pd.DataFrame,
    epsilon: float,
    event_year: int,
) -> pd.DataFrame:
    """Attach the historical expectation and the log anomaly to every week.

    The join is on ``iso_week`` only, because the expectation is the cross-year
    median for that calendar week.
    """
    out = frame.merge(expected_frame, how="left", on="iso_week")
    out = out.rename(columns={"expected": "expected_value"})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_anomaly"] = np.log(
            (out[value_column] + epsilon) / (out["expected_value"] + epsilon)
        )
    missing = out[value_column].isna() | out["expected_value"].isna()
    out.loc[missing, "log_anomaly"] = np.nan
    out["is_event_year"] = out["iso_year"] == event_year
    out["expected_missing"] = out["expected_value"].isna()
    return out


def build_weekly_anomaly_table(
    cfg: dict[str, Any],
    per_aoi_inputs: dict[str, dict[str, Any]],
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble the weekly anomaly table, the phase summary and the population table."""
    epsilon = float(cfg["aggregation"]["epsilon"])
    event_year = int(cfg["analysis"]["event_year"])
    reference_years = list(cfg["analysis"]["reference_years"])
    # The AIS weekly frame already carries ``analysis_value``, chosen per series
    # (sum for all-days, mean per observed SAR day) with the matching thresholds.
    ais_value_column = "analysis_value"
    output_columns = [
        "aoi_key",
        "series",
        "iso_year",
        "iso_week",
        "week_key",
        "week_start",
        "value",
        "expected",
        "log_anomaly",
        "n_scenes",
        "days_observed",
        "is_event_year",
    ]
    weekly_records: list[pd.DataFrame] = []
    population_records: list[dict[str, Any]] = []

    for aoi_key, inputs in per_aoi_inputs.items():
        # ---- SAR series: total fleet plus the vessel-length subsets
        for variant, group in inputs["sar_weekly"].groupby("sar_variant"):
            expected = expected_by_iso_week(group, "density_median", reference_years, logger)
            annotated = add_anomaly_columns(group, "density_median", expected, epsilon, event_year)
            annotated["aoi_key"] = aoi_key
            annotated["series"] = f"SAR_{variant}"
            annotated["value"] = annotated["density_median"]
            annotated["expected"] = annotated["expected_value"]
            annotated["days_observed"] = np.nan
            weekly_records.append(annotated[output_columns])

        if inputs["ais_weekly"].empty:
            continue
        ais_weekly = inputs["ais_weekly"]
        population_sets = cfg["ais"].get("vessel_population_sets") or [
            {"name": "ALL", "classification": "all", "values": ["__ALL__"]}
        ]
        for population in population_sets:
            selector = str(population["name"])
            classification = str(population.get("classification", "all"))
            values = [str(value) for value in population.get("values", [])]
            selected = ais_weekly[
                (ais_weekly["classification"] == classification)
                & (ais_weekly["class_value"].isin(values))
            ]
            if selected.empty:
                continue
            per_series = (
                selected.groupby(
                    ["series", "week_key", "iso_year", "iso_week", "week_start"], as_index=False
                ).agg(
                    value=(ais_value_column, "sum"),
                    presence_hours=("presence_hours", "sum"),
                    days_observed=("days_observed", "max"),
                    n_vessels=("n_vessels", "sum"),
                )
            )
            for series_name, group in per_series.groupby("series"):
                expected = expected_by_iso_week(group, "value", reference_years, logger)
                annotated = add_anomaly_columns(group, "value", expected, epsilon, event_year)
                annotated["aoi_key"] = aoi_key
                annotated["series"] = f"{series_name}|{selector}"
                annotated["expected"] = annotated["expected_value"]
                annotated["n_scenes"] = np.nan
                weekly_records.append(annotated[output_columns])
                population_records.append(
                    {
                        "aoi_key": aoi_key,
                        "series": series_name,
                        "vessel_population": selector,
                        "classification": classification,
                        "class_values": ",".join(values),
                        "n_weeks": int(len(annotated)),
                        "weeks_with_expectation": int((~annotated["expected_missing"]).sum()),
                        "value_total": float(group["value"].sum(skipna=True)),
                    }
                )

    weekly = pd.concat(weekly_records, ignore_index=True) if weekly_records else pd.DataFrame()
    if weekly.empty:
        return weekly, pd.DataFrame(), pd.DataFrame(population_records)

    # ---- observability gap on the AIS rows, using the matching SAR week.
    # The lookup must use exactly ONE SAR series (the all-detections total);
    # indexing on several variants at once would return a Series per lookup.
    sar_lookup = (
        weekly[weekly["series"] == "SAR_ALL_detections"]
        .set_index(["aoi_key", "week_key"])["log_anomaly"]
    )
    if sar_lookup.index.has_duplicates:
        raise ValueError("SAR anomaly lookup is not unique per (aoi_key, week_key)")
    weekly["sar_log_anomaly"] = [
        sar_lookup.get((aoi_key, week), np.nan)
        for aoi_key, week in zip(weekly["aoi_key"], weekly["week_key"])
    ]
    weekly["observability_assessment_gap"] = np.where(
        weekly["series"].str.startswith("AIS_"),
        weekly["log_anomaly"] - weekly["sar_log_anomaly"],
        np.nan,
    )

    # ---- phase summary for the event year (with a bootstrap interval)
    phase_rows: list[dict[str, Any]] = []
    for aoi_key in per_aoi_inputs:
        for phase, window in resolve_phases(cfg, event_year).items():
            subset = weekly[
                (weekly["aoi_key"] == aoi_key)
                & (weekly["iso_year"] == event_year)
                & in_window(pd.to_datetime(weekly["week_start"]), window)
            ]
            for series_name, group in subset.groupby("series"):
                if group.empty:
                    continue
                interval = bootstrap_log_ratio(
                    group["value"].to_numpy(dtype=float),
                    group["expected"].to_numpy(dtype=float),
                    epsilon=epsilon,
                    iterations=int(cfg["stats"]["bootstrap_iterations"]),
                    level=float(cfg["stats"]["bootstrap_ci_level"]),
                    seed=int(cfg["project"]["random_seed"]),
                    min_units_for_ci=int(cfg["stats"].get("min_scenes_per_period", 3)),
                )
                phase_rows.append(
                    {
                        "aoi_key": aoi_key,
                        "phase": phase,
                        "series": series_name,
                        "n_weeks": int(len(group)),
                        "mean_log_anomaly": float(group["log_anomaly"].mean(skipna=True)),
                        "median_log_anomaly": float(group["log_anomaly"].median(skipna=True)),
                        "anomaly_ci_low": interval.low,
                        "anomaly_ci_high": interval.high,
                        "ci_computed": interval.ci_computed,
                        "mean_observability_gap": float(
                            group["observability_assessment_gap"].mean(skipna=True)
                        ),
                        "mean_value": float(group["value"].mean(skipna=True)),
                        "mean_expected": float(group["expected"].mean(skipna=True)),
                    }
                )
    phases_frame = pd.DataFrame(phase_rows)
    population_frame = pd.DataFrame(population_records)
    logger.info(
        "weekly anomaly table: %d rows (%d series); phase summary: %d rows",
        len(weekly),
        weekly["series"].nunique(),
        len(phases_frame),
    )
    return weekly, phases_frame, population_frame


def build_same_day_table(
    cfg: dict[str, Any],
    per_aoi_inputs: dict[str, dict[str, Any]],
    logger,
) -> pd.DataFrame:
    """Same-day SAR vs AIS comparison on every valid SAR acquisition day.

    Descriptive evidence only. ``ais_observation_status`` is carried through so
    that a zero-AIS day can be read as a structural zero rather than as a data
    outage, and ``zero_ais_with_positive_sar`` flags the case observations where
    SAR detected vessels while AIS reported none.
    """
    rows: list[pd.DataFrame] = []
    for aoi_key, inputs in per_aoi_inputs.items():
        metrics = inputs["scene_metrics"].copy()
        metrics["date"] = pd.to_datetime(metrics["scene_date_parsed"]).dt.normalize()
        daily_sar = metrics.groupby("date", as_index=False).agg(
            sar_scenes=("scene_id", "size"),
            sar_detections=("n_total", "sum"),
            sar_matched=("n_matched", "sum"),
            sar_unmatched=("n_unmatched", "sum"),
            sar_coverage_km2=("coverage_km2", "sum"),
            sar_density_median=("sar_density", "median"),
            sar_matching_rate=("matching_rate", "median"),
        )
        status = inputs["ais_status"][
            ["date", "ais_observation_status"]
        ].copy()
        daily = daily_sar.merge(status, how="left", on="date")
        ais_daily = inputs["ais_daily"]
        ais_all = ais_daily[ais_daily["classification"] == "all"].copy()
        if len(ais_all):
            ais_all["date"] = pd.to_datetime(ais_all["date"]).dt.normalize()
            ais_all = ais_all[["date", "presence_hours", "n_vessels"]]
        else:
            ais_all = pd.DataFrame(columns=["date", "presence_hours", "n_vessels"])
        daily = daily.merge(ais_all, how="left", on="date")
        daily["presence_hours"] = daily["presence_hours"].fillna(0.0)
        daily["n_vessels"] = daily["n_vessels"].fillna(0.0)
        daily["aoi_key"] = aoi_key
        daily["zero_ais_with_positive_sar"] = (
            (daily["ais_observation_status"] == "structural_zero")
            & (daily["sar_detections"] > 0)
        )
        rows.append(daily)
    frame = pd.concat(rows, ignore_index=True)
    frame = frame.sort_values(["aoi_key", "date"]).reset_index(drop=True)
    logger.info(
        "same-day table: %d rows; %d zero-AIS days with positive SAR detections",
        len(frame),
        int(frame["zero_ais_with_positive_sar"].sum()),
    )
    return frame


def build_divergence_alignment(
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Onset alignment between the matching-rate decline and the negative gap.

    Onset rule (fixed before looking at the result): for each AOI, a signal has
    "turned" in the first event-year week whose value falls below the 5th
    percentile of that signal's pre-event 2026 weeks. Reporting the onset of the
    gap, of the AIS anomaly, of the SAR anomaly and of the matching rate in the
    same table is what makes the temporal alignment checkable rather than
    asserted.
    """
    event_year = int(cfg["analysis"]["event_year"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    records: list[dict[str, Any]] = []
    for aoi_key in weekly["aoi_key"].unique():
        subset = weekly[(weekly["aoi_key"] == aoi_key) & (weekly["iso_year"] == event_year)]
        if subset.empty:
            continue
        pre = subset[subset["series"] == "SAR_ALL_detections"]
        pre = pre[pd.to_datetime(pre["week_start"]) < pd.Timestamp(cfg["analysis"]["event_date"])]
        for series_name in (
            "AIS_all_days|ALL",
            "AIS_SAR_days|ALL",
            "SAR_ALL_detections",
        ):
            series = subset[subset["series"] == series_name].copy()
            if series.empty:
                continue
            series["week_start"] = pd.to_datetime(series["week_start"])
            baseline = series[series["week_start"] < pd.Timestamp(cfg["analysis"]["event_date"])]
            values = baseline["log_anomaly"].dropna()
            threshold = float(values.quantile(0.05)) if len(values) >= 3 else float("nan")
            post = series[
                (series["week_start"] >= pd.Timestamp(cfg["analysis"]["event_date"]))
                & series["log_anomaly"].notna()
            ]
            onset = pd.NaT
            if len(post) and np.isfinite(threshold):
                crossing = post[post["log_anomaly"] < threshold]
                if len(crossing):
                    onset = crossing["week_start"].min()
            records.append(
                {
                    "aoi_key": aoi_key,
                    "signal": series_name,
                    "baseline_week_5th_percentile": threshold,
                    "onset_week": onset,
                    "weeks_to_first_crossing": (
                        int((onset - pd.Timestamp(cfg["analysis"]["event_date"])).days // 7)
                        if pd.notna(onset) else np.nan
                    ),
                    "event_year_mean_anomaly": float(
                        post["log_anomaly"].mean(skipna=True)
                    ) if len(post) else np.nan,
                }
            )
        # matching rate, taken from the scene-level weekly panel
        weekly_path = aoi_scoped_path(processed / "sar_weekly.parquet", aoi_key, cfg)
        if weekly_path.exists():
            panel = pd.read_parquet(weekly_path)
            panel["week_start"] = pd.to_datetime(panel["week_start"])
            baseline_mask = panel["week_start"] < pd.Timestamp(cfg["analysis"]["event_date"])
            baseline_values = panel.loc[baseline_mask, "matching_rate_scene_median"].dropna()
            threshold = float(baseline_values.quantile(0.05)) if len(baseline_values) >= 3 else np.nan
            post = panel[
                (panel["week_start"] >= pd.Timestamp(cfg["analysis"]["event_date"]))
                & panel["matching_rate_scene_median"].notna()
            ]
            onset = pd.NaT
            if len(post) and np.isfinite(threshold):
                crossing = post[post["matching_rate_scene_median"] < threshold]
                if len(crossing):
                    onset = crossing["week_start"].min()
            records.append(
                {
                    "aoi_key": aoi_key,
                    "signal": "matching_rate_scene_median",
                    "baseline_week_5th_percentile": threshold,
                    "onset_week": onset,
                    "weeks_to_first_crossing": (
                        int((onset - pd.Timestamp(cfg["analysis"]["event_date"])).days // 7)
                        if pd.notna(onset) else np.nan
                    ),
                    "event_year_mean_anomaly": np.nan,
                }
            )
    out = pd.DataFrame(records).sort_values(["aoi_key", "signal"]).reset_index(drop=True)
    logger.info("divergence alignment:\n%s", out.to_string(index=False))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("05_activity_anomaly", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 5 (Phase 2) - AIS vs SAR activity anomalies")

    aoi_keys = args.aois or default_aois(cfg)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    per_aoi_inputs: dict[str, dict[str, Any]] = {}
    for aoi_key in aoi_keys:
        scene_metrics = load_scene_metrics(cfg, aoi_key, logger)
        detections = pd.read_parquet(
            aoi_scoped_path(
                resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet",
                aoi_key,
                cfg,
            )
        )
        sar_days = load_sar_days(cfg, aoi_key, logger)
        ais_daily = load_ais_daily(cfg, aoi_key, logger)
        ais_status = load_ais_status(cfg, aoi_key, logger)
        sar_weekly = build_sar_weekly_variants(scene_metrics, detections, cfg, logger)
        ais_weekly = build_ais_weekly(ais_daily, sar_days, ais_status, cfg, aoi_key, logger)
        per_aoi_inputs[aoi_key] = {
            "sar_weekly": sar_weekly,
            "ais_weekly": ais_weekly,
            "scene_metrics": scene_metrics,
            "ais_daily": ais_daily,
            "ais_status": ais_status,
        }
        if not sar_weekly.empty:
            sar_weekly.to_parquet(
                aoi_scoped_path(processed / "sar_weekly_variants.parquet", aoi_key, cfg),
                index=False,
            )
        if not ais_weekly.empty:
            ais_weekly.to_parquet(
                aoi_scoped_path(processed / "ais_weekly.parquet", aoi_key, cfg), index=False
            )

    weekly, phases, population = build_weekly_anomaly_table(cfg, per_aoi_inputs, logger)
    if weekly.empty:
        logger.error(
            "nothing to summarise: neither an AIS nor a SAR weekly series was available"
        )
        return 3

    weekly.to_csv(tables / "activity_anomaly_weekly.csv", index=False)
    logger.info("wrote %s", tables / "activity_anomaly_weekly.csv")
    if not phases.empty:
        phases.to_csv(tables / "phase_summary.csv", index=False)
        logger.info("wrote %s", tables / "phase_summary.csv")
    if not population.empty:
        population.to_csv(tables / "ais_vessel_population_sensitivity.csv", index=False)
        logger.info("wrote %s", tables / "ais_vessel_population_sensitivity.csv")

    same_day = build_same_day_table(cfg, per_aoi_inputs, logger)
    same_day.to_csv(tables / "same_day_sar_ais.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "same_day_sar_ais.csv", len(same_day))

    alignment = build_divergence_alignment(weekly, cfg, logger)
    alignment.to_csv(tables / "divergence_alignment.csv", index=False)
    logger.info("wrote %s", tables / "divergence_alignment.csv")

    event_year = int(cfg["analysis"]["event_year"])
    summary_rows = []
    for aoi_key in aoi_keys:
        for series_name in (
            "AIS_all_days|ALL",
            "AIS_SAR_days|ALL",
            "SAR_ALL_detections",
        ):
            subset = weekly[(weekly["aoi_key"] == aoi_key) & (weekly["series"] == series_name)]
            event = subset[subset["iso_year"] == event_year]
            summary_rows.append(
                {
                    "aoi": aoi_key,
                    "series": series_name,
                    "n_event_weeks": int(len(event)),
                    "mean_log_anomaly_event": float(event["log_anomaly"].mean(skipna=True))
                    if len(event)
                    else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)
    logger.info("event-year mean anomalies:\n%s", summary.round(3).to_string(index=False))
    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "activity_anomaly_summary.json",
        {
            "ais_value_column": str(cfg["ais"].get("weekly_statistic", "sum")),
            "epsilon": float(cfg["aggregation"]["epsilon"]),
            "event_year": event_year,
            "reference_years": list(cfg["analysis"]["reference_years"]),
            "event_year_mean_anomalies": summary.to_dict(orient="records"),
            "series_available": sorted(weekly["series"].unique().tolist()),
            "aois": aoi_keys,
            "primary_aoi": primary_aoi_key(cfg),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
