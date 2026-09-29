"""STEP 13 (Phase 4B) - cross-event synthesis and the Panama metric decomposition.

Two dimensions are reported side by side and never merged:

* Dimension A - physical activity disruption: the historical-normalized SAR
  activity anomaly of the event phase;
* Dimension B - AIS observability disruption: the composition-standardized change
  in the probability that a SAR detection carries a confirmed AIS match.

The AIS activity metrics (presence hours, distinct vessels, both restricted to SAR
acquisition days, and the dwell proxy) are a third, supporting family. No composite
index is built, events are not ranked, and absolute matching-rate levels are never
compared across GFW pipeline generations - only within-event changes are.

Outputs
-------
``outputs/tables/cross_event_final.csv``
``outputs/tables/event_role_summary.csv``
``outputs/tables/panama_ais_metric_decomposition.csv``
``outputs/tables/panama_transit_core_sensitivity.csv``

Usage
-----
    python src/13_phase4b_cross_event.py
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
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
    write_json,
)

# Case-level interpretation and inference strength. These are curated statements
# written from the numbers in the tables, never computed from effect size.
CASE_NARRATIVE: dict[str, dict[str, str]] = {
    "hormuz_2026": {
        "case_role": "discovery_case",
        "disruption_type": "geopolitical_security",
        "inference_strength": "high (frozen Phase 3 estimate, cluster-robust, placebo-checked)",
        "main_interpretation": (
            "Both dimensions moved: SAR-observed corridor traffic collapsed and the "
            "confirmed AIS-SAR matching probability fell after composition "
            "standardization. The AIS-based activity measure overstated the decline "
            "relative to independent SAR."
        ),
    },
    "redsea_2023": {
        "case_role": "independent_security_validation",
        "disruption_type": "geopolitical_security",
        "inference_strength": "moderate (868 detections, 108 scenes, 2 placebo years)",
        "main_interpretation": (
            "Independent security disruption reproduces the Hormuz direction: a "
            "composition-standardized matching decline while the same-calendar "
            "placebo years show a small increase. Distinct vessels fell far more "
            "than presence hours, so part of the AIS activity signal is dwell-driven."
        ),
    },
    "panama_2023": {
        "case_role": "non_security_contrast",
        "disruption_type": "climate_capacity_disruption",
        "inference_strength": "moderate (consistent across two placebo years and a nested AOI)",
        "main_interpretation": (
            "Real physical disruption with no observability deterioration: matching "
            "probability was stable to slightly higher, and once AIS and SAR are "
            "measured on the same water the vessel-count metric agrees with SAR "
            "while presence hours understate the decline."
        ),
    },
    "bab_mandeb_2026": {
        "case_role": "concurrent_regional_falsification",
        "disruption_type": "geopolitical_security",
        "inference_strength": "descriptive (50 event detections; noisy 2025 placebo)",
        "main_interpretation": (
            "No Hormuz-like observability collapse under the same regional shock and "
            "the same pipeline transition: the standardized change is small and its "
            "interval spans zero, and SAR activity is if anything above its placebo "
            "weeks."
        ),
    },
    "suez_2021": {
        "case_role": "supplementary_physical_blockage",
        "disruption_type": "physical_blockage",
        "inference_strength": "descriptive (7-day event, one acquisition week, 57 detections)",
        "main_interpretation": (
            "Severe short physical blockage without an observability decline: SAR "
            "density inside the AOI rose sharply (congestion) and matching was flat."
        ),
    },
    "panama_transit_core_2023": {
        "case_role": "non_security_contrast_sensitivity_aoi",
        "disruption_type": "climate_capacity_disruption",
        "inference_strength": (
            "descriptive only (126 model detections, 7/13/12 per phase, 2 spatial cells; "
            "the standardized model is not supported)"
        ),
        "main_interpretation": (
            "Nested sensitivity AOI for Panama. Its SAR-observable water is the Pacific "
            "entrance (18.8 km2) because GFW footprints do not cover the canal's "
            "freshwater lake section, so it cannot isolate the lock-and-lake transit "
            "path. The physical decline is larger here than in the primary AOI and the "
            "AIS metrics move the other way, but the sample cannot carry an inference."
        ),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4B cross-event synthesis")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def load_tables(cfg: dict[str, Any], logger) -> dict[str, pd.DataFrame]:
    """Load the per-event artifacts produced by step 12."""
    tables = resolve_path(cfg["paths"]["tables_dir"])
    names = {
        "summaries": "phase4b_event_summaries.csv",
        "sar": "phase4b_sar_activity.csv",
        "ais": "phase4b_ais_phase_summary.csv",
        "standardized": "phase4b_standardized_effects_all.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, name in names.items():
        path = tables / name
        frames[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if frames[key].empty:
            logger.warning("missing or empty: %s", path)
    return frames


def event_phase_row(
    frame: pd.DataFrame,
    event: str,
    phase: str,
    year_role_contains: str = "event_year",
) -> pd.Series | None:
    """Fetch one (event, event-year, phase) row, tolerating the label schemes."""
    if frame.empty:
        return None
    subset = frame[frame["event"] == event]
    if "year_role" in subset.columns:
        subset = subset[subset["year_role"].astype(str).str.contains(year_role_contains)]
    subset = subset[subset["phase"] == phase]
    return subset.iloc[0] if len(subset) else None


def build_cross_event(
    cfg: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    logger,
) -> pd.DataFrame:
    """Assemble the cross-event table (one row per case, within-event changes only)."""
    events_cfg = cfg["phase4b"]["events"]
    summaries = tables["summaries"].set_index("event") if not tables["summaries"].empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, event in events_cfg.items():
        event_phase = str(event["event_phase"])
        pre_phase = next(name for name, role in event["phase_roles"].items() if role == "pre")
        sar_row = event_phase_row(tables["sar"], key, event_phase)
        ais_row = event_phase_row(tables["ais"], key, event_phase)
        ais_pre = event_phase_row(tables["ais"], key, pre_phase)
        summary = summaries.loc[key].to_dict() if (not summaries.empty and key in summaries.index) else {}
        hours_change = np.nan
        vessels_change = np.nan
        dwell_change = np.nan
        if ais_row is not None and ais_pre is not None:
            hours_change = float(
                np.log(
                    (ais_row["ais_presence_hours_daily_mean"] + 1.0)
                    / (ais_pre["ais_presence_hours_daily_mean"] + 1.0)
                )
            )
            vessels_change = float(
                np.log(
                    (ais_row["ais_distinct_vessels_daily_mean"] + 1.0)
                    / (ais_pre["ais_distinct_vessels_daily_mean"] + 1.0)
                )
            )
            pre_dwell = ais_pre["ais_presence_hours_per_vessel"]
            event_dwell = ais_row["ais_presence_hours_per_vessel"]
            if np.isfinite(pre_dwell) and np.isfinite(event_dwell) and pre_dwell:
                dwell_change = float(event_dwell / pre_dwell - 1.0)
        narrative = CASE_NARRATIVE.get(key, {})
        placebo_range = summary.get("placebo_raw_range_pp")
        # frozen cases carry their Phase 3 estimates instead of recomputed ones
        frozen = event.get("frozen_standardized_pp") is not None
        standardized_change = (
            float(event["frozen_standardized_pp"]) if frozen
            else summary.get("standardized_change_pp")
        )
        standardized_low = (
            float(event["frozen_standardized_ci_low_pp"]) if frozen
            else summary.get("standardized_change_ci_low_pp")
        )
        standardized_high = (
            float(event["frozen_standardized_ci_high_pp"]) if frozen
            else summary.get("standardized_change_ci_high_pp")
        )
        raw_change = (
            float(event["frozen_raw_pp"]) if frozen else summary.get("raw_change_pp")
        )
        rows.append(
            {
                "event": key,
                "case_role": narrative.get("case_role", event["case_role"]),
                "disruption_type": narrative.get("disruption_type", event["disruption_type"]),
                "event_duration": f"{event['event_onset']} + {int(summary.get('phase_days') or 0)} days",
                "pipeline_version": "",
                "event_sar_scenes": summary.get("n_event_scenes"),
                "event_detections": summary.get("n_event_detections"),
                # Dimension A - physical activity disruption
                "SAR_activity_anomaly": float(sar_row["sar_activity_anomaly"])
                if sar_row is not None else np.nan,
                "SAR_density_median": float(sar_row["sar_density_median"])
                if sar_row is not None else np.nan,
                # supporting AIS activity family
                "AIS_presence_hours_anomaly": float(ais_row["ais_presence_hours_anomaly"])
                if ais_row is not None else np.nan,
                "AIS_distinct_vessels_anomaly": float(ais_row["ais_distinct_vessels_anomaly"])
                if ais_row is not None else np.nan,
                "AIS_distinct_vessels_SAR_days_anomaly": float(
                    ais_row["ais_distinct_vessels_sar_days_anomaly"]
                ) if ais_row is not None else np.nan,
                "presence_hours_per_vessel_change": dwell_change,
                "AIS_hours_change_log": hours_change,
                "AIS_vessels_change_log": vessels_change,
                # Dimension B - AIS observability disruption
                "raw_matching_change_pp": raw_change,
                "standardized_matching_change_pp": standardized_change,
                "standardized_matching_CI_low": standardized_low,
                "standardized_matching_CI_high": standardized_high,
                "estimates_source": (
                    event.get("frozen_estimates_source", "") if frozen
                    else "outputs/tables/phase4b_event_summaries.csv (this run)"
                ),
                "historical_placebo_range": str(placebo_range) if placebo_range else "n/a",
                "placebo_standardized_change_pp": summary.get("placebo_standardized_change_pp"),
                "inference_strength": narrative.get("inference_strength", "not assessed"),
                "main_interpretation": narrative.get("main_interpretation", ""),
            }
        )
    out = pd.DataFrame(rows)
    # pipeline version per case, read from the detections (never assumed)
    versions: list[str] = []
    for key, event in events_cfg.items():
        detections_path = None
        interim = resolve_path(cfg["paths"]["interim_dir"])
        tag = event.get("sar_window_tag")
        suffix = f"__{event['aoi_key']}" if event["aoi_key"] != "hormuz_core" else ""
        if tag:
            suffix = f"{suffix}__{tag}"
        detections_path = interim / f"sar_detections_aoi{suffix}.parquet"
        if detections_path.exists():
            frame = pd.read_parquet(detections_path, columns=["pipeline", "timestamp"])
            window = pd.to_datetime(frame["timestamp"]).between(
                pd.Timestamp(event["event_onset"]),
                pd.Timestamp(event["event_onset"]) + pd.DateOffset(months=6),
            )
            values = sorted(
                str(value) for value in frame.loc[window, "pipeline"].dropna().unique()
                if str(value).strip() not in {"", "n/a", "none"}
            )
            versions.append("+".join(values))
        else:
            versions.append("n/a")
    out["pipeline_version"] = versions
    logger.info("cross-event table built: %d cases", len(out))
    return out


def build_event_roles(
    cfg: dict[str, Any],
    cross: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """Conceptual role table: the cases are not statistically equivalent replicas."""
    roles: list[dict[str, Any]] = []
    for key, event in cfg["phase4b"]["events"].items():
        narrative = CASE_NARRATIVE.get(key, {})
        row = cross[cross["event"] == key]
        roles.append(
            {
                "event": key,
                "role": narrative.get("case_role", event["case_role"]),
                "disruption_type": narrative.get("disruption_type", event["disruption_type"]),
                "why_included": {
                    "redsea_2023": "independent security-driven validation of the Hormuz finding",
                    "panama_2023": "non-security climate/capacity contrast",
                    "bab_mandeb_2026": "concurrent regional falsification under the same shock and pipeline transition",
                    "suez_2021": "supplementary short-duration physical blockage",
                    "hormuz_2026": "discovery case (frozen)",
                }.get(key, ""),
                "statistical_status": {
                    "redsea_2023": "independent but same pipeline generation as Suez/Panama; not Hormuz",
                    "panama_2023": "independent; different pipeline generation from Hormuz",
                    "bab_mandeb_2026": "NOT independent of Hormuz: same crisis, same pipeline boundary",
                    "suez_2021": "descriptive only",
                    "hormuz_2026": "primary, frozen",
                }.get(key, ""),
                "inference_strength": narrative.get("inference_strength", "not assessed"),
                "event_sar_scenes": row["event_sar_scenes"].iloc[0] if len(row) else np.nan,
                "event_detections": row["event_detections"].iloc[0] if len(row) else np.nan,
            }
        )
    frame = pd.DataFrame(roles)
    logger.info("event role table: %d rows", len(frame))
    return frame


def build_panama_decomposition(
    cfg: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    logger,
) -> pd.DataFrame:
    """Panama metric decomposition: is the flat presence-hours signal a dwell artefact?

    Reports, per phase and year role, both AIS metric families on the same water
    domain as SAR: presence hours (a duration measure), distinct vessels (a count
    measure), the SAR-day variants, and presence hours per distinct vessel.
    """
    ais = tables["ais"]
    rows: list[dict[str, Any]] = []
    for event_key in ("panama_2023", "panama_transit_core_2023"):
        subset = ais[ais["event"] == event_key]
        if subset.empty:
            continue
        for phase, group in subset.groupby("phase"):
            row: dict[str, Any] = {
                "event": event_key,
                "aoi": "panama_canal_core" if event_key == "panama_2023" else "panama_transit_core",
                "phase": phase,
                "year_roles": ";".join(sorted(group["year_role"])),
                "n_weeks": int(group["n_weeks"].sum()),
            }
            for metric, column in (
                ("presence_hours_daily_mean", "ais_presence_hours_daily_mean"),
                ("distinct_vessels_daily_mean", "ais_distinct_vessels_daily_mean"),
                ("distinct_vessels_weekly_mean", "ais_distinct_vessels_weekly_mean"),
                ("distinct_vessels_on_sar_days", "ais_distinct_vessels_on_sar_days"),
                ("presence_hours_on_sar_days", "ais_presence_hours_on_sar_days"),
                ("presence_hours_per_vessel", "ais_presence_hours_per_vessel"),
                ("distinct_vessels_anomaly", "ais_distinct_vessels_anomaly"),
                ("presence_hours_anomaly", "ais_presence_hours_anomaly"),
            ):
                event_year = group[group["year_role"].astype(str).str.contains("event_year")]
                value = (
                    float(event_year[column].mean())
                    if len(event_year) and column in event_year
                    else np.nan
                )
                row[metric] = value
            rows.append(row)
    frame = pd.DataFrame(rows)
    logger.info("Panama decomposition: %d rows", len(frame))
    return frame


def build_transit_core_sensitivity(
    cfg: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    logger,
) -> pd.DataFrame:
    """Primary vs transit-core Panama AOI: SAR density, AIS metrics, matching rate."""
    sar = tables["sar"]
    ais = tables["ais"]
    rows: list[dict[str, Any]] = []
    for event_key, aoi in (
        ("panama_2023", "panama_canal_core"),
        ("panama_transit_core_2023", "panama_transit_core"),
    ):
        sar_subset = sar[(sar["event"] == event_key) & (sar["year_role"].astype(str).str.contains("event_year"))]
        ais_subset = ais[(ais["event"] == event_key) & (ais["year_role"].astype(str).str.contains("event_year"))]
        for phase in ("pre", "restricted", "adaptation"):
            sar_row = sar_subset[sar_subset["phase"] == phase]
            ais_row = ais_subset[ais_subset["phase"] == phase]
            rows.append(
                {
                    "aoi": aoi,
                    "phase": phase,
                    "sar_scenes": int(sar_row["n_scenes"].iloc[0]) if len(sar_row) else 0,
                    "sar_density_median": float(sar_row["sar_density_median"].iloc[0])
                    if len(sar_row) else np.nan,
                    "sar_activity_anomaly": float(sar_row["sar_activity_anomaly"].iloc[0])
                    if len(sar_row) else np.nan,
                    "ais_presence_hours_daily_mean": float(
                        ais_row["ais_presence_hours_daily_mean"].iloc[0]
                    ) if len(ais_row) else np.nan,
                    "ais_distinct_vessels_daily_mean": float(
                        ais_row["ais_distinct_vessels_daily_mean"].iloc[0]
                    ) if len(ais_row) else np.nan,
                    "ais_presence_hours_per_vessel": float(
                        ais_row["ais_presence_hours_per_vessel"].iloc[0]
                    ) if len(ais_row) else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    logger.info("transit-core sensitivity: %d rows", len(frame))
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("13_phase4b_cross_event", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 13 (Phase 4B) - cross-event synthesis")

    tables = load_tables(cfg, logger)
    tables_out = resolve_path(cfg["paths"]["tables_dir"])
    cross = build_cross_event(cfg, tables, logger)
    cross.to_csv(tables_out / "cross_event_final.csv", index=False)
    logger.info("wrote %s", tables_out / "cross_event_final.csv")

    roles = build_event_roles(cfg, cross, logger)
    roles.to_csv(tables_out / "event_role_summary.csv", index=False)
    logger.info("wrote %s", tables_out / "event_role_summary.csv")

    decomposition = build_panama_decomposition(cfg, tables, logger)
    decomposition.to_csv(tables_out / "panama_ais_metric_decomposition.csv", index=False)
    logger.info("wrote %s", tables_out / "panama_ais_metric_decomposition.csv")

    sensitivity = build_transit_core_sensitivity(cfg, tables, logger)
    sensitivity.to_csv(tables_out / "panama_transit_core_sensitivity.csv", index=False)
    logger.info("wrote %s", tables_out / "panama_transit_core_sensitivity.csv")

    headline = cross[
        ["event", "case_role", "SAR_activity_anomaly", "AIS_distinct_vessels_anomaly",
         "raw_matching_change_pp", "standardized_matching_change_pp",
         "standardized_matching_CI_low", "standardized_matching_CI_high"]
    ]
    logger.info("cross-event headline:\n%s", headline.round(3).to_string(index=False))
    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "phase4b_cross_event_summary.json",
        {
            "cases": list(cfg["phase4b"]["events"]),
            "dimension_a": "SAR_activity_anomaly (historical-normalized)",
            "dimension_b": "standardized_matching_change_pp (composition-standardized)",
            "supporting_ais_family": [
                "AIS_presence_hours_anomaly",
                "AIS_distinct_vessels_anomaly",
                "AIS_distinct_vessels_SAR_days_anomaly",
                "presence_hours_per_vessel_change",
            ],
            "no_composite_index": True,
            "cross_event_headline": headline.to_dict(orient="records"),
            "interpretations": {key: value["main_interpretation"] for key, value in CASE_NARRATIVE.items()},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
