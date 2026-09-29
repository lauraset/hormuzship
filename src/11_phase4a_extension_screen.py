"""STEP 11 (Phase 4A-Extension) - feasibility screen for two further candidate events.

Scope: data feasibility and identification checks only. No composition-standardized
model, no final event inference and no pooled cross-event model is fitted here.
The frozen Hormuz results are not touched, and the already-screened Red Sea 2023 /
Suez 2021 cases are not modified.

The emerging hypothesis is *not* "all disruptions reduce AIS-SAR matching". The
screen exists to decide whether two new cases can support a test of whether
**physical maritime disruption** and **AIS observability disruption** are
empirically distinct dimensions:

    bab_mandeb_2026  concurrent spillover case under the same regional crisis and
                     the same GFW pipev3 -> pipev4 transition as Hormuz 2026.
                     Never an independent validation, never a control group.
    panama_2023      non-security climate/capacity disruption contrast. Never a
                     negative control.

A stable matching rate during a strong physical disruption is a valid and
informative outcome; nothing here is tuned toward a particular direction.

Outputs
-------
``outputs/tables/phase4a_extension_sar_feasibility.csv``
    One row per (event, year role, phase) with the required feasibility fields.
``outputs/tables/phase4a_extension_matching_metrics.csv``
    Screening-level matching metrics: phase rates with scene-bootstrap intervals,
    pre->event change, length-bin rates, placebo changes and (for Bab) the
    February->March 2026 month-pair change.
``outputs/tables/phase4a_extension_activity_diagnostics.csv``
    Weekly SAR / AIS anomalies and the preliminary observability gap, aligned to
    the frozen Hormuz ISO-week calendar.

Usage
-----
    python src/11_phase4a_extension_screen.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    add_calendar_fields,
    aoi_area_km2,
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
    safe_divide,
    write_json,
)
from hormuz_stats import bootstrap_pooled_rate, bootstrap_rate_difference  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4A-extension feasibility screen")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--events", nargs="*", default=None,
                        help="extension event keys to screen (default: all)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# phase windows
# ---------------------------------------------------------------------------
def phase_window(
    year: int,
    start_md: str,
    end_md: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Resolve a month-day phase to ``(start, end_inclusive, end_exclusive)``.

    A phase whose end month-day precedes its start month-day crosses the year
    boundary (the Panama restricted phase runs 11-03 -> 02-29). The end day is
    clamped to the last day of its month, so ``02-29`` becomes ``02-28`` in a
    non-leap placebo year instead of raising.
    """
    start = pd.Timestamp(year=int(year), month=int(start_md[:2]), day=int(start_md[3:]))
    end_year = int(year) + 1 if end_md < start_md else int(year)
    end_month, end_day = int(end_md[:2]), int(end_md[3:])
    last_day = pd.Period(f"{end_year}-{end_month:02d}").days_in_month
    end = pd.Timestamp(year=end_year, month=end_month, day=min(end_day, last_day))
    return start, end, end + pd.Timedelta(days=1)


def event_windows(
    event: dict[str, Any],
    logger,
) -> dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]]:
    """Phase windows for the event year and every placebo year."""
    onset_year = pd.Timestamp(event["event_onset"]).year
    year_roles: list[tuple[str, int]] = [("event_year", int(onset_year))]
    for year in event.get("placebo_years", []):
        year_roles.append((f"placebo_{int(year)}", int(year)))
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = {}
    for label, year in year_roles:
        offsets = event.get("phase_year_offsets", {}) or {}
        windows[label] = {
            phase: phase_window(
                int(year) + int(offsets.get(phase, 0)), bounds[0], bounds[1]
            )
            for phase, bounds in event["phases"].items()
        }
    logger.info(
        "event %s: onset %s | phases %s | placebo years %s",
        event.get("case_role", "?"),
        event["event_onset"],
        {
            phase: f"{window[0].date()}..{window[1].date()}"
            for phase, window in windows["event_year"].items()
        },
        [label for label, _ in year_roles if label != "event_year"],
    )
    return windows


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_scene_table(cfg: dict[str, Any], aoi_key: str, tag: str | None) -> pd.DataFrame:
    """Load a (possibly window-tagged) scene table from step 03."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run src/01_prepare_sar.py --aoi {aoi_key} "
            f"--start ... --end ... (and step 03) first"
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    return frame


def load_aoi_detections(cfg: dict[str, Any], aoi_key: str, tag: str | None) -> pd.DataFrame:
    """Load AOI detections for the same window as the scene table."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_ais(cfg: dict[str, Any], aoi_key: str, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the AIS daily table and the day-status table (either may be absent)."""
    processed = resolve_path(cfg["paths"]["processed_dir"])
    daily_path = aoi_scoped_path(processed / "ais_daily.parquet", aoi_key, cfg)
    status_path = aoi_scoped_path(processed / "ais_daily_status.parquet", aoi_key, cfg)
    daily = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
    status = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame()
    if len(daily):
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    if len(status):
        status["date"] = pd.to_datetime(status["date"]).dt.normalize()
        logger.info("AIS status for %s: %s", aoi_key, status["ais_observation_status"].value_counts().to_dict())
    else:
        logger.warning("no AIS status table for %s; AIS metrics will be reported as unavailable", aoi_key)
    return daily, status


# ---------------------------------------------------------------------------
# coverage: bounding box and footprint-derived water domain
# ---------------------------------------------------------------------------
def water_domain(
    scenes: pd.DataFrame,
    logger,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Attach water-normalized coverage and report how robust the domain is.

    ``coverage_fraction_bbox`` = scene ∩ AOI / bounding-box area (the Phase 1 metric).
    ``coverage_fraction_water`` = scene ∩ AOI / water-observable area, where the
    water-observable area is the **union of every scene∩AOI intersection** over
    the loaded window.

    The domain is created from GFW footprint geometry only. GFW footprints are
    water-clipped polygons, so this union is a purely geometric, reproducible
    proxy for the AOI's observable water; **no vessel detection is used to build
    the mask**. Robustness is reported by recomputing the union year by year: if
    the per-year unions agree closely, the denominator does not depend on the
    window choice.
    """
    frame = scenes.copy()
    frame["coverage_fraction_bbox"] = frame["coverage_fraction"]
    geometries = gpd.GeoSeries(
        shapely.from_wkt(frame["intersection_wkt"].fillna("").to_numpy(), on_invalid="ignore"),
        crs="EPSG:4326",
    )

    def union_km2(series: gpd.GeoSeries) -> float:
        usable = series[series.notna() & ~series.is_empty]
        if usable.empty:
            return 0.0
        return float(
            gpd.GeoSeries([usable.union_all()], crs="EPSG:4326")
            .to_crs("EPSG:6933")
            .area.iloc[0] / 1e6
        )

    proxy_km2 = union_km2(geometries)
    per_year = {}
    for year in sorted(pd.to_datetime(frame["date"]).dt.year.unique()):
        mask = pd.to_datetime(frame["date"]).dt.year == year
        per_year[int(year)] = union_km2(geometries[mask.to_numpy()])
    if proxy_km2 > 0:
        frame["coverage_fraction_water"] = frame["coverage_km2"] / proxy_km2
    else:
        frame["coverage_fraction_water"] = np.nan
    robustness = {
        "water_domain_km2": proxy_km2,
        "water_domain_by_year_km2": per_year,
        "max_abs_relative_difference_vs_full_window": (
            max(
                abs(value - proxy_km2) / proxy_km2
                for value in per_year.values()
                if proxy_km2
            )
            if proxy_km2 and per_year
            else np.nan
        ),
        "definition": (
            "union of all scene∩AOI intersections over the loaded window, built from "
            "land-masked GFW footprint geometry; no vessel detections are used"
        ),
    }
    logger.info(
        "water domain %.1f km2 (per-year: %s; max relative deviation %.1f%%)",
        proxy_km2,
        {year: round(value, 1) for year, value in per_year.items()},
        100.0 * robustness["max_abs_relative_difference_vs_full_window"]
        if np.isfinite(robustness["max_abs_relative_difference_vs_full_window"])
        else float("nan"),
    )
    return frame, proxy_km2, robustness


# ---------------------------------------------------------------------------
# per-phase metrics
# ---------------------------------------------------------------------------
def phase_metrics(
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    status: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
    coverage_threshold: float,
) -> dict[str, Any]:
    """Every phase-level number required by the extension feasibility table."""
    start, end_inclusive, end_exclusive = window
    period_scenes = scenes[(scenes["date"] >= start) & (scenes["date"] < end_exclusive)]
    period_dets = detections[
        (detections["timestamp"] >= start) & (detections["timestamp"] < end_exclusive)
    ]
    usable = period_scenes[period_scenes["coverage_fraction_water"] >= coverage_threshold]
    usable_dets = period_dets[period_dets["scene_id"].isin(set(usable["scene_id"]))]
    classified = usable_dets[
        usable_dets["matched_flag"].notna() | usable_dets["unmatched_flag"].notna()
    ]
    period_status = (
        status[(status["date"] >= start) & (status["date"] < end_exclusive)]
        if len(status) else pd.DataFrame(columns=["ais_observation_status"])
    )
    ais_valid = int(
        period_status["ais_observation_status"].isin(["observed_positive", "structural_zero"]).sum()
    ) if len(period_status) else 0
    ais_missing = int(
        (period_status["ais_observation_status"] == "missing_unknown").sum()
    ) if len(period_status) else int((end_exclusive - start).days)
    pipeline_values = sorted(
        {
            str(value).strip()
            for value in period_dets["pipeline"].dropna().unique()
            if str(value).strip() not in {"", "n/a", "none", "nan"}
        }
    )
    length = pd.to_numeric(usable_dets["length_m"], errors="coerce")
    return {
        "phase_start": start.date().isoformat(),
        "phase_end": end_inclusive.date().isoformat(),
        "phase_days": int((end_exclusive - start).days),
        "intersecting_scenes": int(len(period_scenes)),
        "usable_scenes_bbox": int(
            (period_scenes["coverage_fraction_bbox"] >= coverage_threshold).sum()
        ),
        "usable_scenes": int(len(usable)),
        "acquisition_days": int(usable["date"].nunique()),
        "acquisition_weeks": int(usable["week_key"].nunique()) if len(usable) else 0,
        "detections": int(len(usable_dets)),
        "detections_all_scenes": int(len(period_dets)),
        "matched": int(classified["matched_flag"].sum()) if len(classified) else 0,
        "unmatched": int(classified["unmatched_flag"].sum()) if len(classified) else 0,
        "n_classified": int(len(classified)),
        "preliminary_matching_rate": float(classified["matched_flag"].mean())
        if len(classified) else np.nan,
        "detections_per_scene": float(len(usable_dets) / len(usable)) if len(usable) else np.nan,
        "median_coverage_bbox": float(period_scenes["coverage_fraction_bbox"].median())
        if len(period_scenes) else np.nan,
        "median_coverage_water": float(period_scenes["coverage_fraction_water"].median())
        if len(period_scenes) else np.nan,
        "length_m_completeness": float(length.notna().mean()) if len(usable_dets) else np.nan,
        "matched_category_completeness": float(usable_dets["match_label"].notna().mean())
        if len(usable_dets) else np.nan,
        "platform": ",".join(sorted(period_scenes["satellite_platform"].dropna().unique()))
        if len(period_scenes) else "",
        "pipeline_version": ",".join(pipeline_values),
        "ais_valid_days": ais_valid,
        "ais_missing_unknown_days": ais_missing,
        "ais_valid_day_share": ais_valid / max((end_exclusive - start).days, 1),
    }


# ---------------------------------------------------------------------------
# screening-level matching metrics (no standardized model)
# ---------------------------------------------------------------------------
def matching_metrics(
    cfg: dict[str, Any],
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]],
    event: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Phase matching rates, pre->event change, length bins and placebo changes.

    Scene-level resampling is the primary bootstrap (3 000 draws, percentile),
    matching the frozen Hormuz convention. This is a screening diagnostic: no
    composition standardization and no cross-event pooling is performed.
    """
    iterations = int(cfg["stats"]["bootstrap_iterations"])
    level = float(cfg["stats"]["bootstrap_ci_level"])
    seed = int(cfg["project"]["random_seed"])
    min_scenes = int(cfg["stats"].get("min_scenes_per_period", 3))
    bins = [float(value) for value in cfg["identification"]["length_bins_m"]]
    coverage_threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    phase_roles = event["phase_roles"]

    rows: list[dict[str, Any]] = []

    def scene_slice(label: str, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        start, _end_inclusive, end_exclusive = windows[label][phase]
        period_scenes = scenes[(scenes["date"] >= start) & (scenes["date"] < end_exclusive)]
        usable = period_scenes[period_scenes["coverage_fraction_water"] >= coverage_threshold]
        period_dets = detections[
            (detections["timestamp"] >= start) & (detections["timestamp"] < end_exclusive)
        ]
        return usable, period_dets[period_dets["scene_id"].isin(set(usable["scene_id"]))]

    for label, per_phase in windows.items():
        for phase in per_phase:
            usable, dets = scene_slice(label, phase)
            classified = dets[dets["matched_flag"].notna() | dets["unmatched_flag"].notna()]
            interval = bootstrap_pooled_rate(
                usable["n_matched"].to_numpy(),
                (usable["n_matched"] + usable["n_unmatched"]).to_numpy(),
                iterations=iterations,
                level=level,
                seed=seed,
                min_units_for_ci=min_scenes,
            )
            row: dict[str, Any] = {
                "event": event["event_onset"],
                "case_role": event["case_role"],
                "year_role": label,
                "phase": phase,
                "phase_role": phase_roles.get(phase, "other"),
                "n_scenes": int(len(usable)),
                "n_detections": int(len(dets)),
                "n_matched": int(classified["matched_flag"].sum()) if len(classified) else 0,
                "n_unmatched": int(classified["unmatched_flag"].sum()) if len(classified) else 0,
                "pooled_matching_rate": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "ci_computed": interval.ci_computed,
                "scene_mean_matching_rate": float(
                    usable.loc[usable["n_total"] > 0, "matching_rate"].mean()
                ) if (usable["n_total"] > 0).any() else np.nan,
            }
            if phase_roles.get(phase) == "event":
                pre_phase = next(
                    (name for name, role in phase_roles.items() if role == "pre"), None
                )
                if pre_phase:
                    pre_scenes, _ = scene_slice(label, pre_phase)
                    delta = bootstrap_rate_difference(
                        pre_scenes["n_matched"].to_numpy(),
                        (pre_scenes["n_matched"] + pre_scenes["n_unmatched"]).to_numpy(),
                        usable["n_matched"].to_numpy(),
                        (usable["n_matched"] + usable["n_unmatched"]).to_numpy(),
                        iterations=iterations,
                        level=level,
                        seed=seed,
                        min_units_for_ci=min_scenes,
                    )
                    row["pre_phase"] = pre_phase
                    row["pre_matching_rate"] = float(
                        pre_scenes["n_matched"].sum()
                        / max(pre_scenes["n_matched"].sum() + pre_scenes["n_unmatched"].sum(), 1)
                    )
                    row["pre_to_event_delta_pp"] = 100.0 * delta.estimate
                    row["pre_to_event_delta_ci_low_pp"] = 100.0 * delta.low
                    row["pre_to_event_delta_ci_high_pp"] = 100.0 * delta.high
            rows.append(row)
            # length-bin screening rates for this phase
            if len(classified):
                binned = classified.copy()
                binned["length_bin"] = pd.cut(
                    pd.to_numeric(binned["length_m"], errors="coerce"), bins=bins, right=False
                ).astype(str)
                for length_bin, group in binned.groupby("length_bin"):
                    rows.append(
                        {
                            "event": event["event_onset"],
                            "case_role": event["case_role"],
                            "year_role": label,
                            "phase": phase,
                            "phase_role": "length_bin",
                            "n_detections": int(len(group)),
                            "n_matched": int(group["matched_flag"].sum()),
                            "n_unmatched": int(group["unmatched_flag"].sum()),
                            "pooled_matching_rate": float(group["matched_flag"].mean()),
                            "length_bin": length_bin,
                            "small_sample": bool(
                                len(group) < int(cfg["identification"]["small_sample_min_detections"])
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    logger.info("matching screening metrics: %d rows", len(frame))
    return frame


def month_pair_change(
    cfg: dict[str, Any],
    detections: pd.DataFrame,
    before_month: str,
    after_month: str,
    logger,
) -> dict[str, Any]:
    """February-vs-March style month-pair change, for descriptive comparison only.

    Deliberately identical in construction to the frozen ``pipeline_falsification``
    rows (scene-clustered bootstrap), so the numbers can be read side by side.
    This is not a difference-in-differences estimate.
    """
    frame = detections.copy()
    frame["month_key"] = frame["timestamp"].dt.strftime("%Y-%m")
    before = frame[(frame["month_key"] == before_month) & frame["matched_flag"].notna()]
    after = frame[(frame["month_key"] == after_month) & frame["matched_flag"].notna()]
    pre_scene = before.groupby("scene_id")["matched_flag"].agg(["sum", "size"])
    post_scene = after.groupby("scene_id")["matched_flag"].agg(["sum", "size"])
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
    row = {
        "comparison": f"{before_month}->{after_month}",
        "n_classified_before": int(len(before)),
        "n_classified_after": int(len(after)),
        "pooled_rate_before": float(before["matched_flag"].mean()) if len(before) else np.nan,
        "pooled_rate_after": float(after["matched_flag"].mean()) if len(after) else np.nan,
        "delta_pp": 100.0 * interval.estimate,
        "delta_ci_low_pp": 100.0 * interval.low,
        "delta_ci_high_pp": 100.0 * interval.high,
        "ci_computed": interval.ci_computed,
    }
    logger.info("month-pair change %s: %.2f pp [%.2f, %.2f]",
                row["comparison"], row["delta_pp"], row["delta_ci_low_pp"], row["delta_ci_high_pp"])
    return row


# ---------------------------------------------------------------------------
# preliminary activity diagnostics (SAR + AIS, same calendar weeks)
# ---------------------------------------------------------------------------
def activity_diagnostics(
    cfg: dict[str, Any],
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    sar_days: pd.DataFrame,
    ais_daily: pd.DataFrame,
    status: pd.DataFrame,
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]],
    event: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Weekly SAR density and AIS presence anomalies plus the preliminary gap.

    Expected values are the median of the same ISO weeks in the historical
    placebo years, with the anomaly ``log((current + eps) / (expected + eps))``.
    SAR is normalised by the covered area; AIS uses the weekly sum for all days
    and the mean per observed SAR acquisition day for the aligned series (the
    same convention as the frozen Hormuz analysis).
    """
    epsilon = float(cfg["aggregation"]["epsilon"])
    scale = float(cfg["sar"]["sar_density_scale_km2"])
    coverage_threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    reference_years = [int(year) for year in event.get("placebo_years", [])]
    coverage_threshold = coverage_threshold

    # Year-role membership is decided by the *windows*, not by the ISO year: an
    # event year can span two calendar years (Panama 2023-09 -> 2024-04), so
    # ``iso_year == onset_year`` would silently drop the post phase.
    role_windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for label, per_phase in windows.items():
        role_windows.append(
            (
                label,
                min(window[0] for window in per_phase.values()),
                max(window[1] for window in per_phase.values()),
            )
        )

    def assign_role(week_start: pd.Series) -> pd.Series:
        role = pd.Series("outside_windows", index=week_start.index, dtype=object)
        for label, start, end in role_windows:
            mask = (week_start >= start) & (week_start < end)
            role.loc[mask] = label
        return role

    usable = scenes[scenes["coverage_fraction_water"] >= coverage_threshold].copy()
    usable["density"] = safe_divide(usable["n_total"], usable["coverage_km2"]) * scale
    sar_weekly = (
        usable.groupby(["iso_year", "iso_week", "week_start"], as_index=False)
        .agg(sar_n_scenes=("scene_id", "size"), sar_density_median=("density", "median"))
    )
    acquisition_days = set(pd.to_datetime(sar_days["date"]).dt.normalize()) if len(sar_days) else set()

    records: list[dict[str, Any]] = []
    if len(ais_daily):
        ais = ais_daily[ais_daily["classification"] == "all"].copy()
        if len(status):
            usable_status = status[
                status["ais_observation_status"].isin(["observed_positive", "structural_zero"])
            ]
            structural_zero_days = set(
                status.loc[status["ais_observation_status"] == "structural_zero", "date"]
            )
            ais = ais[ais["date"].isin(set(usable_status["date"]))]
            zero_rows = pd.DataFrame(
                {"date": sorted(structural_zero_days - set(ais["date"]))}
            ).assign(presence_hours=0.0, n_vessels=0)
            if len(zero_rows):
                ais = pd.concat([ais, zero_rows], ignore_index=True)
        ais["is_sar_day"] = ais["date"].isin(acquisition_days)
        ais = add_calendar_fields(ais, "date")
        for variant, subset, statistic in (
            ("AIS_all_days", ais, "sum"),
            ("AIS_SAR_days", ais[ais["is_sar_day"]], "mean"),
        ):
            if subset.empty:
                continue
            grouped = (
                subset.groupby(["iso_year", "iso_week", "week_start"], as_index=False)
                .agg(ais_hours=("presence_hours", "sum"), days_observed=("date", "nunique"))
            )
            grouped["ais_value"] = np.where(
                statistic == "sum",
                grouped["ais_hours"],
                safe_divide(grouped["ais_hours"], grouped["days_observed"]),
            )
            grouped["ais_series"] = variant
            records.append(grouped)
    weekly_ais = (
        pd.concat(records, ignore_index=True) if records
        else pd.DataFrame(columns=["iso_year", "iso_week", "week_start", "ais_value", "ais_series"])
    )

    keys = ["iso_year", "iso_week"]
    sar_expected = (
        sar_weekly.assign(year_role=assign_role(sar_weekly["week_start"]))
        .query("year_role != 'event_year' and year_role != 'outside_windows'")
        .groupby(["iso_week"], as_index=False)["sar_density_median"].median()
        .rename(columns={"sar_density_median": "sar_expected"})
    )
    out = sar_weekly.merge(sar_expected, how="left", on="iso_week")
    out["sar_anomaly"] = np.log(
        (out["sar_density_median"] + epsilon) / (out["sar_expected"] + epsilon)
    )
    out["event_onset"] = event["event_onset"]
    out["case_role"] = event["case_role"]
    out["year_role"] = assign_role(out["week_start"])
    if len(weekly_ais):
        ais_expected = (
            weekly_ais.assign(year_role=assign_role(weekly_ais["week_start"]))
            .query("year_role != 'event_year' and year_role != 'outside_windows'")
            .groupby(["ais_series", "iso_week"], as_index=False)["ais_value"].median()
            .rename(columns={"ais_value": "ais_expected"})
        )
        weekly_ais = weekly_ais.merge(ais_expected, how="left", on=["ais_series", "iso_week"])
        weekly_ais["ais_anomaly"] = np.log(
            (weekly_ais["ais_value"] + epsilon) / (weekly_ais["ais_expected"] + epsilon)
        )
        merged = out.merge(
            weekly_ais[
                ["iso_year", "iso_week", "week_start", "ais_series", "ais_value",
                 "ais_expected", "ais_anomaly", "days_observed"]
            ],
            how="left",
            on=["iso_year", "iso_week", "week_start", "week_start"],
        )
        merged["observability_gap"] = merged["ais_anomaly"] - merged["sar_anomaly"]
        out = merged
    out["is_event_year"] = out["year_role"] == "event_year"
    logger.info(
        "activity diagnostics: %d weekly rows (%d event-year, AIS series: %s)",
        len(out),
        int(out["is_event_year"].sum()),
        sorted(out["ais_series"].dropna().unique()) if "ais_series" in out else [],
    )
    return out


# ---------------------------------------------------------------------------
# feasibility classification (observational support only, never effect size)
# ---------------------------------------------------------------------------
def classify_sar(
    event_phase: dict[str, Any],
    placebo_available: bool,
    screening: dict[str, Any],
) -> tuple[str, str]:
    """SAR-side verdict from acquisition support and technical comparability."""
    reasons: list[str] = []
    if event_phase["usable_scenes"] == 0:
        return "NO-GO", (
            "no usable SAR scene covers the event phase "
            f"(intersecting scenes: {event_phase['intersecting_scenes']})"
        )
    if event_phase["detections"] == 0:
        return "NO-GO", "no SAR detections inside the event phase"
    if (event_phase["median_coverage_water"] or 0) < float(
        screening["min_median_coverage_fraction"]
    ):
        reasons.append(
            "median water coverage below "
            f"{screening['min_median_coverage_fraction']}"
        )
    if event_phase["usable_scenes"] < int(screening["min_event_scenes"]):
        reasons.append(
            f"usable event scenes {event_phase['usable_scenes']} < "
            f"{screening['min_event_scenes']}"
        )
    if event_phase["acquisition_days"] < int(screening["min_event_acquisition_days"]):
        reasons.append(
            f"acquisition days {event_phase['acquisition_days']} < "
            f"{screening['min_event_acquisition_days']}"
        )
    if event_phase["detections"] < int(screening["min_event_detections"]):
        reasons.append(
            f"event detections {event_phase['detections']} < "
            f"{screening['min_event_detections']}"
        )
    prolonged = event_phase["phase_days"] >= int(screening["prolonged_phase_days"])
    if prolonged and event_phase["acquisition_weeks"] < int(
        screening["min_event_acquisition_weeks_prolonged"]
    ):
        reasons.append(
            f"prolonged phase has only {event_phase['acquisition_weeks']} acquisition week(s) "
            f"(< {screening['min_event_acquisition_weeks_prolonged']})"
        )
    if bool(screening.get("strong_requires_placebo", True)) and not placebo_available:
        reasons.append("no historical placebo year has usable scenes in the same calendar windows")
    if bool(screening.get("strong_requires_pipeline_interpretability", True)) and not event_phase[
        "pipeline_interpretable"
    ]:
        reasons.append(
            "pipeline structure across the event boundary is not interpretable "
            f"(pre: {event_phase['pipeline_pre'] or 'none'}; event: "
            f"{event_phase['pipeline_event'] or 'none'}; post: "
            f"{event_phase['pipeline_post'] or 'none'})"
        )
    if reasons:
        return "CONDITIONAL", "; ".join(reasons)
    return "STRONG", "sufficient SAR acquisitions and technically interpretable windows"


def classify_ais(
    event_phase: dict[str, Any],
    screening: dict[str, Any],
) -> tuple[str, str]:
    """AIS-side verdict from report coverage only (never from the values)."""
    if event_phase["ais_valid_days"] == 0:
        return "CONDITIONAL", (
            "AIS report coverage for the event phase is incomplete "
            f"({event_phase['ais_missing_unknown_days']} day(s) missing_unknown); "
            "no AIS-based claim is made"
        )
    share = event_phase["ais_valid_day_share"]
    if share < float(screening.get("min_ais_day_share", 0.9)):
        return "CONDITIONAL", (
            f"AIS coverage {share:.1%} of event-phase days "
            f"(< {float(screening.get('min_ais_day_share', 0.9)):.0%})"
        )
    return "STRONG", f"AIS report coverage complete for {event_phase['ais_valid_days']} day(s)"


def combine(sar: str, ais: str) -> str:
    """Combined verdict: the weaker of the two, never stronger."""
    order = {"NO-GO": 0, "CONDITIONAL": 1, "STRONG": 2}
    return min([sar, ais], key=lambda value: order[value])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def screen_event(
    cfg: dict[str, Any],
    event_key: str,
    event: dict[str, Any],
    logger,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Screen one extension event and return its table rows plus diagnostics."""
    aoi_key = str(event["aoi_key"])
    tag = event.get("sar_window_tag")
    screening = cfg["multievent"]["screening"]
    scenes = load_scene_table(cfg, aoi_key, tag)
    scenes, water_km2, robustness = water_domain(scenes, logger)
    detections = load_aoi_detections(cfg, aoi_key, tag)
    sar_days_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_daily_aoi_coverage.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    sar_days = pd.read_parquet(sar_days_path) if sar_days_path.exists() else pd.DataFrame()
    if len(sar_days):
        sar_days["date"] = pd.to_datetime(sar_days["date"]).dt.normalize()
    ais_daily, status = load_ais(cfg, aoi_key, logger)
    windows = event_windows(event, logger)

    metrics: dict[str, dict[str, dict[str, Any]]] = {
        label: {
            phase: phase_metrics(
                scenes,
                detections,
                status,
                window,
                float(cfg["sar"]["min_scene_coverage_fraction"]),
            )
            for phase, window in per_phase.items()
        }
        for label, per_phase in windows.items()
    }
    reference = metrics["event_year"]
    event_phase_name = str(event["event_phase"])
    event_phase = reference[event_phase_name]
    pre_name = next(name for name, role in event["phase_roles"].items() if role == "pre")
    post_name = next(name for name, role in event["phase_roles"].items() if role == "post")
    event_phase["pipeline_pre"] = reference[pre_name]["pipeline_version"]
    event_phase["pipeline_event"] = event_phase["pipeline_version"]
    event_phase["pipeline_post"] = reference[post_name]["pipeline_version"]
    # Interpretability: either a single pipeline generation, or a change that
    # coincides with the onset (as in Hormuz/Bab 2026) which the within-event
    # design can handle by comparing the change against the historical placebo.
    versions = {
        event_phase["pipeline_pre"], event_phase["pipeline_event"], event_phase["pipeline_post"]
    } - {""}
    same_pipeline_pre_event = bool(
        event_phase["pipeline_pre"] and event_phase["pipeline_pre"] == event_phase["pipeline_event"]
    )
    event_phase["same_pipeline_pre_event"] = same_pipeline_pre_event
    # A pipeline transition exactly at the event onset (Bab 2026 and Hormuz 2026
    # share the 2026-03-01 v3 -> v4 boundary) does not make the within-event
    # comparison uninterpretable: it makes the placebo-adjusted comparison
    # mandatory, exactly as in the frozen Hormuz design. That is different from a
    # transition that lands *inside* one of the phases, which would corrupt the
    # phase it splits.
    transition_at_onset = bool(event.get("pipeline_transition_at_onset", False))
    event_phase["pipeline_transition_at_onset"] = transition_at_onset
    event_phase["pipeline_interpretable"] = (
        len(versions) <= 1 or same_pipeline_pre_event or transition_at_onset
    )

    placebo_labels = [label for label in metrics if label != "event_year"]
    placebo_available = any(
        metrics[label][event_phase_name]["usable_scenes"] >= int(screening["min_placebo_scenes"])
        for label in placebo_labels
    ) if placebo_labels else False
    sar_status, sar_reason = classify_sar(event_phase, placebo_available, screening)
    ais_status, ais_reason = classify_ais(event_phase, screening)
    combined = combine(sar_status, ais_status)
    logger.info(
        "%s: SAR=%s AIS=%s combined=%s | %s",
        event_key,
        sar_status,
        ais_status,
        combined,
        sar_reason,
    )

    rows: list[dict[str, Any]] = []
    for label, per_phase in metrics.items():
        for phase, values in per_phase.items():
            rows.append(
                {
                    "event": event_key,
                    "case_role": event["case_role"],
                    "disruption_type": event["disruption_type"],
                    "event_onset": event["event_onset"],
                    "phase": phase,
                    "phase_role": event["phase_roles"].get(phase, "other"),
                    "aoi": aoi_key,
                    "aoi_area_km2": round(aoi_area_km2(cfg, aoi_key=aoi_key), 3),
                    "water_observable_area_km2": round(water_km2, 3),
                    "year_role": label,
                    "calendar_year": int(values["phase_start"][:4]),
                    "sar_window_tag": tag,
                    **values,
                    "pipeline_pre": event_phase["pipeline_pre"],
                    "pipeline_event": event_phase["pipeline_event"],
                    "pipeline_post": event_phase["pipeline_post"],
                    "same_pipeline_pre_event": same_pipeline_pre_event,
                    "onset_coincides_with_pipeline_transition": (
                        bool(event["event_onset"] == "2026-03-01")
                    ),
                    "historical_placebo_available": placebo_available,
                    "feasibility_status_sar": sar_status,
                    "feasibility_status_ais": ais_status,
                    "feasibility_status_combined": combined,
                    "failure_reason": "; ".join(
                        reason for reason in (sar_reason if sar_status != "STRONG" else "",
                                              ais_reason if ais_status != "STRONG" else "")
                        if reason
                    ) or "none",
                }
            )
    matching = matching_metrics(cfg, scenes, detections, windows, event, logger)
    matching["event"] = event_key
    activity = activity_diagnostics(
        cfg, scenes, detections, sar_days, ais_daily, status, windows, event, logger
    )
    activity["event"] = event_key
    activity.attrs["water_domain_robustness"] = robustness
    return rows, matching, activity


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("11_phase4a_extension_screen", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 11 (Phase 4A-Extension) - feasibility screen for two new events")

    events: dict[str, Any] = cfg["multievent"]["extension_events"]
    keys = args.events or list(events)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    row_frames: list[pd.DataFrame] = []
    matching_frames: list[pd.DataFrame] = []
    activity_frames: list[pd.DataFrame] = []
    month_pairs: list[dict[str, Any]] = []
    robustness: dict[str, Any] = {}
    for key in keys:
        if key not in events:
            logger.warning("unknown extension event %s; configured: %s", key, list(events))
            continue
        rows, matching, activity = screen_event(cfg, key, events[key], logger)
        row_frames.append(pd.DataFrame(rows))
        matching_frames.append(matching)
        activity_frames.append(activity)
        robustness[key] = activity.attrs.get("water_domain_robustness")
        if key == "bab_mandeb_2026":
            tag = events[key].get("sar_window_tag")
            detections = load_aoi_detections(cfg, str(events[key]["aoi_key"]), tag)
            pair = month_pair_change(cfg, detections, "2026-02", "2026-03", logger)
            pair["event"] = key
            month_pairs.append(pair)

    feasibility = pd.concat(row_frames, ignore_index=True)
    feasibility.to_csv(tables / "phase4a_extension_sar_feasibility.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "phase4a_extension_sar_feasibility.csv", len(feasibility))

    matching_table = pd.concat(matching_frames, ignore_index=True)
    matching_table.to_csv(tables / "phase4a_extension_matching_metrics.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "phase4a_extension_matching_metrics.csv", len(matching_table))

    activity_table = pd.concat(activity_frames, ignore_index=True)
    activity_table.to_csv(tables / "phase4a_extension_activity_diagnostics.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "phase4a_extension_activity_diagnostics.csv", len(activity_table))

    month_pair_table = pd.DataFrame(month_pairs)
    if len(month_pair_table):
        month_pair_table.to_csv(tables / "phase4a_extension_month_pair.csv", index=False)
        logger.info("wrote %s", tables / "phase4a_extension_month_pair.csv")

    headline = feasibility[feasibility["phase_role"] == "event"][
        ["event", "phase", "usable_scenes", "acquisition_days", "acquisition_weeks",
         "detections", "preliminary_matching_rate", "median_coverage_water",
         "ais_valid_days", "ais_missing_unknown_days", "pipeline_version",
         "same_pipeline_pre_event", "feasibility_status_sar", "feasibility_status_ais",
         "feasibility_status_combined"]
    ]
    logger.info("event-phase headline:\n%s", headline.round(3).to_string(index=False))
    for key in feasibility["event"].unique():
        subset = feasibility[feasibility["event"] == key]
        logger.info(
            "%-18s SAR=%-11s AIS=%-11s combined=%-11s | %s",
            key,
            subset["feasibility_status_sar"].iloc[0],
            subset["feasibility_status_ais"].iloc[0],
            subset["feasibility_status_combined"].iloc[0],
            subset["failure_reason"].iloc[0],
        )

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "phase4a_extension_summary.json",
        {
            "events": keys,
            "case_roles": {key: events[key]["case_role"] for key in keys if key in events},
            "screening_thresholds": cfg["multievent"]["screening"],
            "water_domain_robustness": robustness,
            "month_pair_row": month_pairs,
            "status_by_event": feasibility.groupby("event")[
                ["feasibility_status_sar", "feasibility_status_ais", "feasibility_status_combined"]
            ].first().to_dict(orient="index"),
            "monthly_screening_rates": feasibility.groupby("event")[
                "preliminary_matching_rate"
            ].describe().to_dict(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
