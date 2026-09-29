"""STEP 10 (Phase 4A) - multi-event feasibility screening.

Scope: DATA FEASIBILITY and IDENTIFICATION checks only. No event effect is
estimated here, and the frozen Hormuz specification is not touched (primary AOI
``hormuz_core``, narrow AOI as a pre-specified sensitivity, 2026-02-28 event
date, ``matched_category != unmatched`` matching definition, the Phase 3
standardization model).

What it does, per candidate event
--------------------------------
1. resolves the pre / event / post windows from the documented event chronology
   plus the configured 8-week pre and post margins, and the same calendar windows
   in one or two historical placebo years;
2. counts intersecting and usable SAR scenes, acquisition days and weeks,
   detections and coverage for every period;
3. records the GFW pipeline version(s) and satellite platform(s), and whether the
   pre and post periods sit on the same pipeline (they must, or the comparison is
   not identified);
4. records whether ``length_m`` and ``matched_category`` are populated, since the
   later event comparison needs both;
5. records AIS presence availability (observed days vs missing_unknown days) from
   the step-02 status table, never treating a missing report as zero activity;
6. classifies the event STRONG / CONDITIONAL / NO-GO against pre-specified
   thresholds and writes the reason.

Coverage denominators
---------------------
GFW footprints are land-masked water polygons. In a canal or chokepoint AOI whose
bounding box is mostly land (Suez is ~94 % desert), the box-relative coverage
fraction collapses to a few per cent even though the AOI is imaged completely.
The screen therefore reports **both** denominators:

``coverage_fraction``        intersection / AOI box area (the Phase 1 metric)
``coverage_fraction_water``  intersection / imageable-water proxy, where the
                             proxy is the union of every scene intersection with
                             the AOI over the whole archive window

The water proxy is a purely geometric, result-independent quantity and is applied
identically to every event.

Usage
-----
    python src/10_multievent_screening.py
    python src/10_multievent_screening.py --events suez_2021
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
    aoi_area_km2,
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
    write_json,
)
from hormuz_common import shapely as _shapely  # noqa: E402  (re-exported dependency check)
import geopandas as gpd  # noqa: E402

PERIODS = ("pre_event", "event", "post_event")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4A multi-event feasibility screen")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--events", nargs="*", default=None,
                        help="event keys to screen (default: all configured events)")
    return parser.parse_args(argv)


def event_windows(
    cfg: dict[str, Any],
    event: dict[str, Any],
    logger,
) -> dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    """Pre / event / post windows for the event year and every placebo year.

    Windows are ``(start_inclusive, end_exclusive)`` on whole days, so a window
    labelled ``2021-03-23 .. 2021-03-29`` includes everything up to
    ``2021-03-30T00:00`` and the periods tile without gaps. Using
    ``pd.Timestamp(end_date)`` directly would silently drop the final day's
    daytime observations - the same class of error as the GFW report API's
    exclusive ``date-range`` end.
    """
    event_start, event_end = (pd.Timestamp(value) for value in event["event_window"])
    event_end_exclusive = event_end + pd.Timedelta(days=1)
    pre_weeks = int(event.get("pre_weeks", 8))
    post_weeks = int(event.get("post_weeks", 8))
    years: list[tuple[str, int, int]] = [
        ("event_year", int(event_start.year), 0)
    ]
    for year in event.get("placebo_years", []):
        years.append((f"placebo_{int(year)}", int(year), int(event_start.year) - int(year)))

    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for label, year, shift in years:
        start = event_start - pd.DateOffset(years=shift)
        end_exclusive = event_end_exclusive - pd.DateOffset(years=shift)
        windows[label] = {
            "pre_event": (start - pd.Timedelta(weeks=pre_weeks), start),
            "event": (start, end_exclusive),
            "post_event": (
                end_exclusive,
                end_exclusive + pd.Timedelta(weeks=post_weeks),
            ),
        }
    logger.info(
        "windows for event_year %s: pre %s..%s | event %s..%s | post %s..%s "
        "(%d placebo year(s): %s)",
        windows["event_year"]["event"][0].date(),
        windows["event_year"]["pre_event"][0].date(),
        (windows["event_year"]["pre_event"][1] - pd.Timedelta(days=1)).date(),
        windows["event_year"]["event"][0].date(),
        (windows["event_year"]["event"][1] - pd.Timedelta(days=1)).date(),
        windows["event_year"]["post_event"][0].date(),
        (windows["event_year"]["post_event"][1] - pd.Timedelta(days=1)).date(),
        len(windows) - 1,
        ", ".join(label for label in windows if label != "event_year"),
    )
    return windows


def load_scene_table(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Load the AOI scene table written by steps 01/03."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet", aoi_key, cfg
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run src/01_prepare_sar.py --aoi {aoi_key} --start ... --end ... "
            f"and src/03_build_scene_metrics.py --aoi {aoi_key} first"
        )
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    return frame


def add_water_coverage(scenes: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add the imageable-water coverage fraction to a scene table.

    The proxy is the union of every scene intersection with the AOI across the
    whole loaded window: geometrically, "the part of this AOI that GFW ever
    images". It uses no matching-rate information.
    """
    frame = scenes.copy()
    geometries = gpd.GeoSeries(
        _shapely.from_wkt(frame["intersection_wkt"].fillna("").to_numpy(), on_invalid="ignore"),
        crs="EPSG:4326",
    )
    union = geometries.union_all()
    proxy_km2 = float(
        gpd.GeoSeries([union], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6
    )
    if proxy_km2 <= 0:
        frame["coverage_fraction_water"] = np.nan
        return frame, proxy_km2
    frame["coverage_fraction_water"] = frame["coverage_km2"] / proxy_km2
    return frame, proxy_km2


def load_aoi_detections(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Load AOI detections (all intersecting scenes, before the coverage filter)."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet", aoi_key, cfg
    )
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run src/01_prepare_sar.py --aoi {aoi_key} first")
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_ais_status(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Load the AIS day-status table when it exists (may not, if the fetch failed)."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily_status.parquet", aoi_key, cfg
    )
    if not path.exists():
        return pd.DataFrame(columns=["date", "ais_observation_status"])
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def period_metrics(
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    status: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    coverage_threshold: float,
) -> dict[str, Any]:
    """Every period-level number the feasibility table reports."""
    start, end = window
    # ``end`` is exclusive by construction (see event_windows).
    period_scenes = scenes[(scenes["date"] >= start) & (scenes["date"] < end)]
    period_dets = detections[
        (detections["timestamp"] >= start) & (detections["timestamp"] < end)
    ]
    usable_box = period_scenes[
        period_scenes["coverage_fraction"] >= coverage_threshold
    ]
    usable_water = period_scenes[
        period_scenes["coverage_fraction_water"] >= coverage_threshold
    ]
    usable_ids = set(usable_water["scene_id"])
    usable_dets = period_dets[period_dets["scene_id"].isin(usable_ids)]
    period_status = (
        status[(status["date"] >= start) & (status["date"] < end)]
        if len(status) else pd.DataFrame(columns=["ais_observation_status"])
    )
    ais_observed = (
        int(period_status["ais_observation_status"].isin(
            ["observed_positive", "structural_zero"]
        ).sum())
        if len(period_status) else 0
    )
    ais_missing = (
        int((period_status["ais_observation_status"] == "missing_unknown").sum())
        if len(period_status) else int((end - start).days)
    )
    classified = usable_dets[usable_dets["matched_flag"].notna() | usable_dets["unmatched_flag"].notna()]
    # Period-level fields are all prefixed with ``period_`` so that they can never
    # collide with the required event-level columns (``detections``,
    # ``median_coverage``, ``ais_valid_days`` ...) in the output table.
    return {
        "period_window_start": start.date().isoformat(),
        "period_window_end": (end - pd.Timedelta(days=1)).date().isoformat(),
        "period_days": int((end - start).days),
        "period_scenes_intersecting": int(len(period_scenes)),
        "period_scenes_usable_box": int(len(usable_box)),
        "period_scenes_usable_water": int(len(usable_water)),
        "period_acquisition_days": int(usable_water["date"].nunique()),
        "period_acquisition_weeks": int(usable_water["week_key"].nunique())
        if len(usable_water) else 0,
        "period_detections": int(usable_dets["scene_id"].size),
        "period_detections_all_scenes": int(len(period_dets)),
        "period_detections_classified": int(len(classified)),
        "period_matched": int(classified["matched_flag"].sum()) if len(classified) else 0,
        "period_unmatched": int(classified["unmatched_flag"].sum()) if len(classified) else 0,
        "period_median_coverage_fraction_box": float(
            period_scenes["coverage_fraction"].median()
        ) if len(period_scenes) else np.nan,
        "period_median_coverage_fraction_water": float(
            period_scenes["coverage_fraction_water"].median()
        ) if len(period_scenes) else np.nan,
        # The GFW matching-pipeline token lives in the detection file names, not
        # in the footprint file names (footprint rows carry no pipeline token).
        # It must therefore be read from the detections: this is the pipeline that
        # actually produced the matched/unmatched labels.
        "period_pipeline_versions": ",".join(
            sorted(
                str(value)
                for value in period_dets["pipeline"].dropna().unique()
                if str(value).strip() not in {"", "n/a", "none", "nan"}
            )
        ) if len(period_dets) else "",
        "period_platforms": ",".join(
            sorted(period_scenes["satellite_platform"].dropna().unique())
        ) if len(period_scenes) else "",
        "period_length_m_available_share": float(
            pd.to_numeric(usable_dets["length_m"], errors="coerce").notna().mean()
        ) if len(usable_dets) else np.nan,
        "period_matched_category_available_share": float(
            usable_dets["match_label"].notna().mean()
        ) if len(usable_dets) else np.nan,
        "period_ais_valid_days": ais_observed,
        "period_ais_missing_unknown_days": ais_missing,
        "period_ais_valid_day_share": (ais_observed / ((end - start).days)),
    }


def classify(
    event_summary: dict[str, Any],
    screening: dict[str, Any],
    include_ais: bool = True,
) -> tuple[str, str]:
    """STRONG / CONDITIONAL / NO-GO plus the reason, from pre-specified thresholds.

    ``include_ais=False`` evaluates the SAR-and-pipeline criteria alone. Both
    verdicts are written, so an AIS fetch that is blocked by the API report quota
    (an operational constraint, not a property of the event) cannot disguise a
    SAR-side STRONG case as a data-quality failure.
    """
    reasons: list[str] = []
    if event_summary["period_scenes_usable_water"] == 0:
        return "NO-GO", (
            "no usable SAR scene intersects the AOI inside the event window "
            f"(intersecting scenes: {event_summary['period_scenes_intersecting']})"
        )
    if event_summary["period_detections"] == 0:
        return "NO-GO", "no SAR detections inside the event window"
    if not event_summary["same_pipeline_pre_post"]:
        return "NO-GO", (
            "the GFW matching pipeline differs across the event boundary "
            f"(pre: {event_summary['pipeline_versions_pre'] or 'none'}; "
            f"post: {event_summary['pipeline_versions_post'] or 'none'})"
        )
    if (event_summary["period_median_coverage_fraction_water"] or 0) < float(
        screening["min_median_coverage_fraction"]
    ):
        reasons.append(
            "median water coverage fraction below "
            f"{screening['min_median_coverage_fraction']}"
        )
    if event_summary["period_scenes_usable_water"] < int(screening["min_event_scenes"]):
        reasons.append(
            f"usable event scenes {event_summary['period_scenes_usable_water']} < "
            f"{screening['min_event_scenes']}"
        )
    # The gate is on acquisition DAYS, not weeks: a 7-day blockage (Suez 2021)
    # can never contain three ISO weeks, so a weeks-based gate would reject every
    # short event by construction. Weeks are still reported.
    if event_summary["period_acquisition_days"] < int(
        screening["min_event_acquisition_days"]
    ):
        reasons.append(
            f"event acquisition days {event_summary['period_acquisition_days']} < "
            f"{screening['min_event_acquisition_days']}"
        )
    if event_summary["period_detections"] < int(screening["min_event_detections"]):
        reasons.append(
            f"event detections {event_summary['period_detections']} < "
            f"{screening['min_event_detections']}"
        )
    if bool(screening.get("strong_requires_placebo", True)) and not event_summary[
        "historical_placebo_available"
    ]:
        reasons.append("no historical placebo year has usable scenes in the same windows")
    if include_ais and event_summary["period_ais_valid_days"] < int(
        screening.get("strong_requires_ais_days", 1)
    ):
        reasons.append(
            "AIS presence unavailable for the event window (report quota / fetch incomplete)"
        )
    if reasons:
        return "CONDITIONAL", "; ".join(reasons)
    return "STRONG", "sufficient SAR acquisitions and technically comparable pre/post periods"


def screen_event(
    cfg: dict[str, Any],
    event_key: str,
    event: dict[str, Any],
    logger,
) -> list[dict[str, Any]]:
    """Build one row per (event, year role, period) for a single event."""
    aoi_key = str(event["aoi_key"])
    screening = cfg["multievent"]["screening"]
    coverage_threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    scenes = load_scene_table(cfg, aoi_key)
    scenes, water_proxy_km2 = add_water_coverage(scenes)
    detections = load_aoi_detections(cfg, aoi_key)
    status = load_ais_status(cfg, aoi_key)
    box_area = aoi_area_km2(cfg, aoi_key=aoi_key)
    logger.info(
        "event %s / AOI %s: box %.1f km2, imageable-water proxy %.1f km2, %d scenes",
        event_key,
        aoi_key,
        box_area,
        water_proxy_km2,
        len(scenes),
    )
    windows = event_windows(cfg, event, logger)

    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for label, per_period in windows.items():
        metrics[label] = {
            period: period_metrics(
                scenes, detections, status, window, coverage_threshold
            )
            for period, window in per_period.items()
        }

    reference = metrics["event_year"]
    placebo_labels = [label for label in metrics if label != "event_year"]
    placebo_available = any(
        metrics[label][period]["period_scenes_usable_water"]
        >= int(screening["min_placebo_scenes"])
        for label in placebo_labels
        for period in PERIODS
    ) if placebo_labels else False
    pipelines_pre = {
        value for value in reference["pre_event"]["period_pipeline_versions"].split(",") if value
    }
    pipelines_post = {
        value for value in reference["post_event"]["period_pipeline_versions"].split(",") if value
    }
    same_pipeline = bool(pipelines_pre) and pipelines_pre == pipelines_post
    event_summary = {
        **reference["event"],
        "historical_placebo_available": placebo_available,
        "same_pipeline_pre_post": same_pipeline,
        "pipeline_versions_pre": ",".join(sorted(pipelines_pre)),
        "pipeline_versions_post": ",".join(sorted(pipelines_post)),
    }
    status_label, reason = classify(event_summary, screening, include_ais=True)
    sar_only_label, sar_only_reason = classify(event_summary, screening, include_ais=False)
    logger.info(
        "event %s -> %s (SAR-only %s) | %s",
        event_key,
        status_label,
        sar_only_label,
        reason,
    )

    rows: list[dict[str, Any]] = []
    for label, per_period in metrics.items():
        for period, values in per_period.items():
            row: dict[str, Any] = {
                "event": event_key,
                "event_type": event["event_type"],
                "event_date": event["event_date"],
                "event_window": f"{event['event_window'][0]}..{event['event_window'][1]}",
                "aoi_key": aoi_key,
                "aoi_role": "event_aoi_multievent",
                "ano_area_note": "aoi_area is the bounding-box area; imageable_water_area_km2 is the geometric proxy",
                "aoi_area": round(float(box_area), 3),
                "imageable_water_area_km2": round(float(water_proxy_km2), 3),
                "year_role": label,
                "calendar_year": int(pd.Timestamp(values["period_window_start"]).year),
                "period": period,
                # period-level numbers
                **values,
                # event-level numbers, repeated so every required field is present
                "pipeline_version": reference["event"]["period_pipeline_versions"],
                "platform": reference["event"]["period_platforms"],
                "pre_scenes": reference["pre_event"]["period_scenes_usable_water"],
                "event_scenes": reference["event"]["period_scenes_usable_water"],
                "post_scenes": reference["post_event"]["period_scenes_usable_water"],
                "event_acquisition_days": reference["event"]["period_acquisition_days"],
                "event_acquisition_weeks": reference["event"]["period_acquisition_weeks"],
                "detections": reference["event"]["period_detections"],
                "median_coverage": reference["event"]["period_median_coverage_fraction_water"],
                "ais_valid_days": reference["event"]["period_ais_valid_days"],
                "historical_placebo_available": placebo_available,
                "same_pipeline_pre_post": same_pipeline,
                "documented_chronology": event.get("documented_chronology", "").strip(),
                "feasibility_status": status_label,
                "feasibility_status_sar_only": sar_only_label,
                "failure_reason_sar_only": sar_only_reason,
                "failure_reason": reason,
            }
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("10_multievent_screening", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 10 (Phase 4A) - multi-event feasibility screening")

    events: dict[str, Any] = cfg["multievent"]["events"]
    keys = args.events or list(events)
    rows: list[dict[str, Any]] = []
    for key in keys:
        if key not in events:
            logger.warning("unknown event %s; configured: %s", key, list(events))
            continue
        rows.extend(screen_event(cfg, key, events[key], logger))

    table = pd.DataFrame(rows)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    out_path = tables / "multievent_feasibility.csv"
    table.to_csv(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(table))

    headline = table[table["period"] == "event"][
        ["event", "year_role", "period_scenes_usable_water", "period_acquisition_days",
         "period_acquisition_weeks", "period_detections",
         "period_median_coverage_fraction_water", "period_ais_valid_days",
         "same_pipeline_pre_post", "feasibility_status"]
    ]
    logger.info("event-period headline:\n%s", headline.to_string(index=False))
    for key in table["event"].unique():
        subset = table[table["event"] == key]
        logger.info(
            "%-14s status=%s reason=%s",
            key,
            subset["feasibility_status"].iloc[0],
            subset["failure_reason"].iloc[0],
        )

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "multievent_screening_summary.json",
        {
            "events": keys,
            "screening_thresholds": cfg["multievent"]["screening"],
            "coverage_denominators": [
                "coverage_fraction = intersection / AOI bounding-box area",
                "coverage_fraction_water = intersection / union of all scene intersections",
            ],
            "status_by_event": table.groupby("event")["feasibility_status"].first().to_dict(),
            "reason_by_event": table.groupby("event")["failure_reason"].first().to_dict(),
            "note": (
                "Feasibility screening only. No event effect is estimated here, and absolute "
                "matching-rate levels must not be compared across events whose GFW pipelines differ."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
