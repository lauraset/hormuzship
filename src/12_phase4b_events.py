"""STEP 12 (Phase 4B) - per-event publication-scale analysis.

For every frozen case in ``phase4b.events`` this step produces

* SAR activity: scene-level density, weekly density and the historical-normalized
  anomaly against the same ISO weeks of the placebo years;
* the AIS metric family, kept separate: presence hours, distinct vessels, both
  restricted to SAR acquisition days, and the presence-hours-per-vessel dwell
  proxy;
* the AIS-observability model: detection-level logistic regression of the
  confirmed match (``matched_category != "unmatched"``) on period, vessel length
  and spatial cell, with scene-clustered standard errors, standardized to the
  event-year pre phase, plus a placebo-only seasonal benchmark;
* the raw pre->event change with a scene-bootstrap interval.

Two dimensions are never merged: Dimension A is the SAR activity anomaly,
Dimension B is the composition-standardized matching change. No composite index is
built and no machine learning is used.

Outputs
-------
``outputs/tables/<event>_standardized_effects.csv``      (per case)
``outputs/tables/phase4b_sar_activity.csv``
``outputs/tables/phase4b_ais_metrics.csv``
``outputs/tables/phase4b_ais_phase_summary.csv``
``outputs/tables/phase4b_matching_coefficients.csv``
``outputs/diagnostics/phase4b_model_diagnostics.json``

Usage
-----
    python src/12_phase4b_events.py
    python src/12_phase4b_events.py --events redsea_2023
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
import hormuz_event_model as model  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4B per-event analysis")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--events", nargs="*", default=None,
                        help="event keys to analyse (default: all frozen cases)")
    return parser.parse_args(argv)


def analyse_event(
    cfg: dict[str, Any],
    event_key: str,
    event: dict[str, Any],
    logger,
) -> dict[str, Any]:
    """Run the full Phase 4B analysis for one case and return its tables."""
    analysis = cfg["phase4b"]["analysis"]
    aoi_key = str(event["aoi_key"])
    tag = event.get("sar_window_tag")
    coverage_threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    density_scale = float(cfg["sar"]["sar_density_scale_km2"])
    epsilon = float(cfg["aggregation"]["epsilon"])

    scenes = model.load_scene_table(cfg, aoi_key, tag)
    scenes, water_km2 = model.ensure_water_coverage(scenes)
    water_geom, _ = model.water_domain_geometry(scenes)
    detections = model.load_detections(cfg, aoi_key, tag)
    # AIS metrics are computed inside the SAR-observable water domain so that the
    # two dimensions describe the same water (decisive for Panama, where the
    # Gatun Lake section carries AIS hours that Sentinel-1 never images).
    ais_daily_full = model.load_ais_daily(cfg, aoi_key)
    ais_daily = model.load_ais_daily_in_domain(cfg, aoi_key, water_geom, logger)
    status = model.load_ais_status(cfg, aoi_key)
    vessel_days = model.load_vessel_days(
        cfg, aoi_key, logger, inside_geom=water_geom, cache_suffix="_waterdomain"
    )
    # compare like with like: the step-02 table also carries vessel_type and
    # gear_type breakdowns, so only the all-types classification is comparable
    if len(ais_daily_full) and "classification" in ais_daily_full:
        ais_daily_full = ais_daily_full[ais_daily_full["classification"] == "all"]
    domain_hours_full = float(ais_daily_full["presence_hours"].sum()) if len(ais_daily_full) else 0.0
    domain_hours_scoped = float(ais_daily["presence_hours"].sum()) if len(ais_daily) else 0.0
    logger.info(
        "AIS water-domain scoping for %s: %.0f of %.0f presence hours kept (%.1f%%)",
        aoi_key,
        domain_hours_scoped,
        domain_hours_full,
        100.0 * domain_hours_scoped / domain_hours_full if domain_hours_full else float("nan"),
    )
    sar_days_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_daily_aoi_coverage.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    sar_days = pd.read_parquet(sar_days_path) if sar_days_path.exists() else pd.DataFrame()
    if len(sar_days):
        sar_days["date"] = pd.to_datetime(sar_days["date"]).dt.normalize()

    windows = model.phase_windows(event, logger)
    event_role = model.event_year_label(event)
    # Phase names differ per event (pre/event/post, baseline/acute/persistent,
    # pre/restricted/adaptation, pre/blockage/backlog); the roles are declared in
    # the config so no name is ever hardcoded here.
    roles = event.get("phase_roles", {})
    pre_phase = next(name for name, role in roles.items() if role == "pre")
    event_phase = str(event["event_phase"])

    # ---- activity --------------------------------------------------------
    scene_activity, weekly_sar = model.sar_activity(
        scenes, event, windows, coverage_threshold, density_scale, epsilon, logger
    )
    weekly_ais, ais_phase = model.ais_metrics(
        ais_daily, status, vessel_days, sar_days, event, windows, epsilon, logger
    )

    sar_phase = model.phase_summary(
        weekly_sar, event, ["sar_density_median", "sar_activity_anomaly", "n_scenes"]
    )
    sar_phase["event"] = event_key
    if len(ais_phase):
        ais_phase["event"] = event_key

    # ---- matching model --------------------------------------------------
    frame, diagnostics = model.build_model_frame(scenes, detections, event, windows, cfg, logger)
    standardized = pd.DataFrame()
    coefficients = pd.DataFrame()
    raw = {}
    placebo = {}
    # support check: a standardized model on a handful of detections is noise
    period_counts = frame["period"].value_counts().to_dict() if not frame.empty else {}
    deficit = {
        label: int(count)
        for label, count in period_counts.items()
        if count < int(analysis["min_model_detections_per_period"])
    }
    model_supported = (
        not frame.empty
        and diagnostics.get("n_cells_kept", 0) >= int(analysis["min_model_cells"])
        and not deficit
    )
    if not model_supported and not frame.empty:
        diagnostics["model_skipped"] = (
            "insufficient sample support for the standardized model: "
            f"periods below {analysis['min_model_detections_per_period']} detections = {deficit}; "
            f"cells kept = {diagnostics.get('n_cells_kept')} "
            f"(minimum {analysis['min_model_cells']})"
        )
        logger.warning("%s: %s", event_key, diagnostics["model_skipped"])
    if event.get("standardize_to") and model_supported:
        target = str(event["standardize_to"])
        try:
            fitted, design_info = model.fit_detection_logit(
                frame,
                str(analysis["formula"]),
                str(analysis["cluster_variable"]),
                int(analysis["max_iterations"]),
                logger,
            )
            labels = [label for label in event["phases"] if label in set(frame["period"])]
            standardized = model.standardized_probabilities(
                fitted,
                design_info,
                frame,
                labels=labels,
                target_label=target,
                draws=int(analysis["simulation_draws"]),
                seed=int(cfg["project"]["random_seed"]),
            )
            standardized["event"] = event_key
            standardized["aoi_key"] = aoi_key
            standardized["case_role"] = event["case_role"]
            coefficients = model.coefficients_to_frame(
                fitted,
                aoi_key,
                event_key,
                "detection_logit_cluster_scene",
                str(analysis["formula"]),
                note=(
                    "period = event-year phases plus one pooled placebo_reference level; "
                    "matching_score and MMSI presence excluded as endogenous"
                ),
            )
            raw = model.raw_phase_change(
                frame,
                target,
                event_phase,
                str(analysis["cluster_variable"]),
                int(cfg["stats"]["bootstrap_iterations"]),
                float(cfg["stats"]["bootstrap_ci_level"]),
                int(cfg["project"]["random_seed"]),
                int(cfg["stats"].get("min_scenes_per_period", 3)),
            )
            placebo = model.placebo_benchmark(
                frame, cfg, event, target, event_phase, logger
            )
        except Exception as error:  # noqa: BLE001 - recorded, never silently skipped
            diagnostics["model_error"] = f"{type(error).__name__}: {error}"
            logger.warning("matching model failed for %s: %s", event_key, error)
    else:
        if event.get("standardize_to") is None:
            diagnostics["model_skipped"] = "standardize_to is null (descriptive case)"
        if event.get("frozen_standardized_pp") is not None:
            # The discovery case is frozen: its Dimension B estimate comes from the
            # Phase 3 tables and must not be recomputed here (a recomputation on a
            # pooled multi-year frame would be a different, non-comparable number).
            diagnostics["model_skipped"] = (
                "frozen case: standardized and raw matching changes are carried "
                f"through from {event.get('frozen_estimates_source')}"
            )
            logger.info("%s: %s", event_key, diagnostics["model_skipped"])
        elif not frame.empty:
            raw = model.raw_phase_change(
                frame.assign(period=frame["phase"]) if not frame.empty else frame,
                pre_phase,
                event_phase,
                str(analysis["cluster_variable"]),
                int(cfg["stats"]["bootstrap_iterations"]),
                float(cfg["stats"]["bootstrap_ci_level"]),
                int(cfg["project"]["random_seed"]),
                int(cfg["stats"].get("min_scenes_per_period", 3)),
            )

    event_row = (
        standardized[standardized["period"] == event_phase]
        if "period" in standardized.columns else standardized
    )
    pre_row = (
        standardized[standardized["period"] == str(event.get("standardize_to"))]
        if "period" in standardized.columns else standardized
    )
    summary = {
        "event": event_key,
        "case_role": event["case_role"],
        "disruption_type": event["disruption_type"],
        "aoi_key": aoi_key,
        "aoi_area_km2": round(aoi_area_km2(cfg, aoi_key=aoi_key), 3),
        "water_observable_area_km2": round(water_km2, 3),
        "coverage_metric": event["coverage_metric"],
        "event_onset": event["event_onset"],
        "event_phase": event_phase,
        "pre_phase": str(event.get("standardize_to")),
        "phase_days": int((windows[event_role][event_phase][2] - windows[event_role][event_phase][0]).days),
        "sar_window": "..".join(event["sar_window"]),
        "sar_window_tag": tag,
        "raw_pre_rate": raw.get("raw_rate_pre"),
        "raw_event_rate": raw.get("raw_rate_event"),
        "raw_change_pp": raw.get("raw_change_pp"),
        "raw_change_ci_low_pp": raw.get("raw_change_ci_low_pp"),
        "raw_change_ci_high_pp": raw.get("raw_change_ci_high_pp"),
        "standardized_pre_probability": float(pre_row["standardized_matching_probability"].iloc[0])
        if len(pre_row) else np.nan,
        "standardized_event_probability": float(
            event_row["standardized_matching_probability"].iloc[0]
        ) if len(event_row) else np.nan,
        "standardized_change_pp": float(event_row["standardized_change_vs_target_pp"].iloc[0])
        if len(event_row) else np.nan,
        "standardized_change_ci_low_pp": float(
            event_row["standardized_change_ci_low_pp"].iloc[0]
        ) if len(event_row) else np.nan,
        "standardized_change_ci_high_pp": float(
            event_row["standardized_change_ci_high_pp"].iloc[0]
        ) if len(event_row) else np.nan,
        "observed_pre_rate": float(pre_row["observed_matching_rate"].iloc[0])
        if len(pre_row) else np.nan,
        "observed_event_rate": float(event_row["observed_matching_rate"].iloc[0])
        if len(event_row) else np.nan,
        "n_pre_detections": raw.get("n_pre_detections"),
        "n_event_detections": raw.get("n_event_detections"),
        "n_pre_scenes": raw.get("n_pre_scenes"),
        "n_event_scenes": raw.get("n_event_scenes"),
        **{key: value for key, value in placebo.items()},
    }
    return {
        "event": event_key,
        "event_cfg": event,
        "summary": summary,
        "standardized": standardized,
        "coefficients": coefficients,
        "sar_phase": sar_phase,
        "sar_weekly": weekly_sar,
        "scene_activity": scene_activity,
        "ais_weekly": weekly_ais,
        "ais_phase": ais_phase,
        "diagnostics": diagnostics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("12_phase4b_events", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 12 (Phase 4B) - per-event analysis")

    events: dict[str, Any] = cfg["phase4b"]["events"]
    keys = args.events or list(events)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    standardized_all: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    sar_phase_frames: list[pd.DataFrame] = []
    sar_weekly_frames: list[pd.DataFrame] = []
    ais_phase_frames: list[pd.DataFrame] = []
    ais_weekly_frames: list[pd.DataFrame] = []
    scene_frames: list[pd.DataFrame] = []

    for key in keys:
        if key not in events:
            logger.warning("unknown event %s; configured: %s", key, list(events))
            continue
        result = analyse_event(cfg, key, events[key], logger)
        summaries.append(result["summary"])
        diagnostics[key] = result["diagnostics"]
        if not result["standardized"].empty:
            standardized_all.append(result["standardized"])
            result["standardized"].to_csv(
                tables / f"{key}_standardized_effects.csv", index=False
            )
            logger.info("wrote %s", tables / f"{key}_standardized_effects.csv")
        if not result["coefficients"].empty:
            coefficient_frames.append(result["coefficients"])
        sar_phase_frames.append(result["sar_phase"])
        sar_weekly_frames.append(result["sar_weekly"].assign(event=key))
        if not result["ais_phase"].empty:
            ais_phase_frames.append(result["ais_phase"])
        if not result["ais_weekly"].empty:
            ais_weekly_frames.append(result["ais_weekly"].assign(event=key))
        scene_frames.append(
            result["scene_activity"][
                [
                    "scene_id",
                    "date",
                    "phase",
                    "year_role",
                    "coverage_km2",
                    "coverage_fraction",
                    "coverage_fraction_water",
                    "n_total",
                    "n_matched",
                    "n_unmatched",
                    "sar_density",
                ]
            ].assign(event=key, aoi_key=result["summary"]["aoi_key"])
        )

    # per-event standardized tables are already written; keep the combined views
    if standardized_all:
        pd.concat(standardized_all, ignore_index=True).to_csv(
            tables / "phase4b_standardized_effects_all.csv", index=False
        )
    if coefficient_frames:
        pd.concat(coefficient_frames, ignore_index=True).to_csv(
            tables / "phase4b_matching_coefficients.csv", index=False
        )
    pd.concat(sar_phase_frames, ignore_index=True).to_csv(
        tables / "phase4b_sar_activity.csv", index=False
    )
    pd.concat(sar_weekly_frames, ignore_index=True).to_csv(
        tables / "phase4b_sar_weekly.csv", index=False
    )
    if ais_phase_frames:
        pd.concat(ais_phase_frames, ignore_index=True).to_csv(
            tables / "phase4b_ais_phase_summary.csv", index=False
        )
    if ais_weekly_frames:
        pd.concat(ais_weekly_frames, ignore_index=True).to_csv(
            tables / "phase4b_ais_metrics.csv", index=False
        )
    pd.concat(scene_frames, ignore_index=True).to_csv(
        tables / "phase4b_scene_activity.csv", index=False
    )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(tables / "phase4b_event_summaries.csv", index=False)

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "phase4b_model_diagnostics.json",
        {
            "events": keys,
            "model_settings": cfg["phase4b"]["analysis"],
            "diagnostics": diagnostics,
            "note": (
                "Dimension A = SAR activity anomaly; Dimension B = composition-standardized "
                "matching change. The two are never combined into a composite index."
            ),
        },
    )
    headline = summary_frame[
        ["event", "case_role", "n_pre_detections", "n_event_detections",
         "raw_change_pp", "standardized_change_pp", "standardized_change_ci_low_pp",
         "standardized_change_ci_high_pp", "placebo_standardized_change_pp"]
    ]
    logger.info("Phase 4B per-event headline:\n%s", headline.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
