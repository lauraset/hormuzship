"""Unit tests for the calculations the feasibility verdict depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

import hormuz_common as hc
from importlib import import_module

scene_metrics = import_module("03_build_scene_metrics")


# ---------------------------------------------------------------------------
# coverage / area
# ---------------------------------------------------------------------------
def test_measure_intersections_half_covered_square() -> None:
    """A footprint covering half of the AOI has coverage_fraction ~0.5."""
    aoi = box(0.0, 0.0, 1.0, 1.0)
    footprint = pd.Series([box(0.5, 0.0, 1.5, 1.0)])
    result = hc.measure_intersections(footprint, aoi, equal_area_crs="EPSG:6933")
    assert result["geometry_valid"].iloc[0]
    assert result["coverage_fraction"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert result["intersection_km2"].iloc[0] > 0


def test_measure_intersections_disjoint_returns_zero() -> None:
    """A footprint that only overlaps the bbox but not the polygon yields 0 km2."""
    aoi = box(0.0, 0.0, 1.0, 1.0)
    footprint = pd.Series([box(2.0, 2.0, 3.0, 3.0)])
    result = hc.measure_intersections(footprint, aoi, equal_area_crs="EPSG:6933")
    assert result["intersection_km2"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert result["coverage_fraction"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_measure_intersections_flags_invalid_geometry() -> None:
    """A degenerate polygon is flagged instead of being silently used."""
    aoi = box(0.0, 0.0, 1.0, 1.0)
    degenerate = Polygon([(0, 0), (1, 1), (2, 2)])
    result = hc.measure_intersections(pd.Series([degenerate]), aoi, equal_area_crs="EPSG:6933")
    assert not result["geometry_valid"].iloc[0]
    assert np.isnan(result["intersection_km2"].iloc[0])


def test_points_within_handles_missing_coordinates() -> None:
    """NaN coordinates must be excluded, not treated as inside."""
    aoi = box(0.0, 0.0, 1.0, 1.0)
    inside = hc.points_within([0.5, np.nan, 5.0], [0.5, 0.5, 0.5], aoi)
    assert inside.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# scene metrics
# ---------------------------------------------------------------------------
def _scenes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_id": ["a", "b", "c", "d"],
            "scene_date_parsed": pd.to_datetime(
                ["2026-02-01", "2026-02-03", "2026-03-02", "2026-04-01"]
            ),
            "coverage_km2": [1000.0, 500.0, 1000.0, 1000.0],
            "coverage_fraction": [0.5, 0.1, 0.5, 0.5],
            "footprint_area_km2": [2000.0, 2000.0, 2000.0, 2000.0],
            "pipeline": ["v3", "v3", "v4", "v4"],
            "satellite_platform": ["S1A"] * 4,
            "geometry_valid": [True] * 4,
        }
    )


def _detections(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("matched_flag", "unmatched_flag", "unknown_flag", "mmsi_matched_flag"):
        frame[column] = frame[column].astype(bool) if column in frame.columns else pd.Series(dtype=bool)
    for column in ("scene_id", "timestamp", "matching_score", "length_m", "presence_score"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object" if column in {"scene_id", "timestamp"} else "float64")
    return frame


def test_scene_metrics_counts_and_rates() -> None:
    """Counts, matching rates and densities follow the documented definitions."""
    cfg = {"sar": {"sar_density_scale_km2": 1000.0, "min_scene_coverage_fraction": 0.3,
                   "min_scene_coverage_km2": 0.0},
           "analysis": {"event_date": "2026-02-28"}}
    detections = _detections(
        [
            dict(scene_id="a", matched_flag=True, unmatched_flag=False, unknown_flag=False, mmsi_matched_flag=True, matching_score=1.0, length_m=1.0, presence_score=1.0, timestamp=pd.Timestamp("2026-02-01")),
            dict(scene_id="a", matched_flag=False, unmatched_flag=True, unknown_flag=False, mmsi_matched_flag=False, matching_score=0.0, length_m=2.0, presence_score=1.0, timestamp=pd.Timestamp("2026-02-01")),
            dict(scene_id="a", matched_flag=False, unmatched_flag=True, unknown_flag=False, mmsi_matched_flag=False, matching_score=0.0, length_m=3.0, presence_score=1.0, timestamp=pd.Timestamp("2026-02-01")),
            dict(scene_id="c", matched_flag=False, unmatched_flag=False, unknown_flag=True, mmsi_matched_flag=False, matching_score=0.0, length_m=4.0, presence_score=1.0, timestamp=pd.Timestamp("2026-03-02")),
        ]
    )
    metrics = scene_metrics.build_scene_metrics(_scenes(), detections, cfg).set_index("scene_id")

    assert metrics.loc["a", "n_total"] == 3
    assert metrics.loc["a", "n_matched"] == 1
    assert metrics.loc["a", "n_unmatched"] == 2
    assert metrics.loc["a", "matching_rate"] == pytest.approx(1 / 3)
    assert metrics.loc["a", "sar_density"] == pytest.approx(3.0)  # 3 / 1000 km2 * 1000

    # scenes without detections stay in the table as genuine zeros
    assert metrics.loc["b", "n_total"] == 0
    assert np.isnan(metrics.loc["b", "matching_rate"])
    assert metrics.loc["b", "sar_density"] == pytest.approx(0.0)

    # the coverage threshold only flags scenes, it never removes them here
    assert not bool(metrics.loc["b", "passes_coverage_threshold"])
    assert bool(metrics.loc["c", "passes_coverage_threshold"])

    # a scene with an unknown match label breaks the count identity
    assert not bool(metrics.loc["c", "count_identity_ok"])
    assert bool(metrics.loc["a", "count_identity_ok"])


def test_weekly_panel_excludes_weeks_without_acquisition() -> None:
    """Weeks with no SAR acquisition must be absent (NA), never zero."""
    cfg = {"sar": {"sar_density_scale_km2": 1000.0, "min_scene_coverage_fraction": 0.3,
                   "min_scene_coverage_km2": 0.0},
           "analysis": {"event_date": "2026-02-28"}}
    metrics = scene_metrics.build_scene_metrics(_scenes(), _detections([]), cfg)
    metrics["usable_for_primary_metrics"] = metrics["passes_coverage_threshold"]
    weekly = scene_metrics.build_weekly_panel(metrics, cfg)
    assert len(weekly) == 3  # scenes a, c, d; b is below the coverage threshold
    assert (weekly["n_scenes"] == 1).all()


def test_weekly_panel_uses_scene_median_not_pooled_ratio() -> None:
    """The weekly SAR statistic is a median over scenes, per config."""
    cfg = {"sar": {"sar_density_scale_km2": 1000.0, "min_scene_coverage_fraction": 0.3,
                   "min_scene_coverage_km2": 0.0},
           "analysis": {"event_date": "2026-02-28"}}
    scenes = pd.concat([_scenes(), _scenes().assign(scene_id=["e", "f", "g", "h"],
                                                    scene_date_parsed=pd.to_datetime(
                                                        ["2026-02-01"] * 4))], ignore_index=True)
    metrics = scene_metrics.build_scene_metrics(scenes, _detections([]), cfg)
    metrics["usable_for_primary_metrics"] = metrics["passes_coverage_threshold"]
    weekly = scene_metrics.build_weekly_panel(metrics, cfg)
    first_week = weekly[weekly["week_key"] == "2026-W05"]
    assert len(first_week) == 1
    assert first_week["sar_density_median"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# windows, weeks, anomaly helpers
# ---------------------------------------------------------------------------
def test_resolve_phases_applies_same_calendar_window_to_every_year() -> None:
    """Phase windows are calendar-defined so placebo years line up exactly."""
    cfg = {
        "analysis": {
            "phases": {
                "baseline": {"start_md": "01-01", "end_md": "02-27"},
                "acute_shock": {"start_md": "03-01", "end_md": "03-27"},
            }
        }
    }
    for year in (2024, 2025, 2026):
        phases = hc.resolve_phases(cfg, year)
        assert phases["baseline"][0] == pd.Timestamp(f"{year}-01-01")
        assert phases["baseline"][1] == pd.Timestamp(f"{year}-02-27")
        assert phases["acute_shock"][0] == pd.Timestamp(f"{year}-03-01")


def test_add_calendar_fields_iso_week() -> None:
    """ISO week and week_start are derived consistently."""
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-03-01"])})
    out = hc.add_calendar_fields(frame, "date")
    assert out["week_key"].tolist() == ["2026-W01", "2026-W09"]
    assert out["week_start"].iloc[0] == pd.Timestamp("2025-12-29")


def test_safe_divide_handles_zero_denominator() -> None:
    """Division by zero yields NaN instead of inf."""
    out = hc.safe_divide(pd.Series([1.0, 1.0]), pd.Series([0.0, 2.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.5)


def test_parse_scene_platform_and_time() -> None:
    """Platform and acquisition time are read from the GFW scene id."""
    scene_id = "S1A_IW_GRDH_1SDV_20250101T045915_20250101T045940_057246_070AB3_0B4A"
    assert hc.parse_scene_platform(scene_id) == "S1A"
    assert hc.scene_acquisition_time(scene_id) == pd.Timestamp("2025-01-01 04:59:15")


def test_parse_gfw_datetimes_keeps_missing_as_nat() -> None:
    """Missing timestamps stay NaT and are never replaced."""
    values = pd.Series(["2025-01-01 04:59:28 UTC", None, "", "2025-02-01 00:00:00 UTC"])
    parsed = hc.parse_gfw_datetimes(values)
    assert parsed.iloc[0] == pd.Timestamp("2025-01-01 04:59:28")
    assert parsed.iloc[1:3].isna().all()
    assert parsed.iloc[3] == pd.Timestamp("2025-02-01")


def test_missing_tokens_mask_covers_gfw_null_vocabulary() -> None:
    """Both the empty string and the literal 'NA' are recognised as missing."""
    labels = pd.Series(["unmatched", "NA", "", "cargo", None], dtype="string")
    mask = hc.missing_tokens_mask(labels, ["", "NA", "nan", "none"])
    assert mask.tolist() == [False, True, True, False, True]


def test_filter_ledger_records_before_after_removed() -> None:
    """Every filter step reports before / after / removed / reason."""
    ledger = hc.FilterLedger(name="unit")
    ledger.record("bbox", before=100, after=40, reason="outside bbox")
    frame = ledger.to_frame()
    assert frame.loc[0, "before_rows"] == 100
    assert frame.loc[0, "after_rows"] == 40
    assert frame.loc[0, "removed_rows"] == 60
    assert frame.loc[0, "reason"] == "outside bbox"
