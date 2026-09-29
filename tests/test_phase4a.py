"""Tests for the Phase 4A multi-event feasibility screening."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import hormuz_common as hc

screening = import_module("10_multievent_screening")


class _NullLogger:
    """Logger stub."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# the frozen Hormuz specification and the new event AOIs
# ---------------------------------------------------------------------------
def test_hormuz_specification_is_unchanged(config: dict) -> None:
    """Phase 4A must not move the primary AOI, event date or matching definition."""
    assert hc.primary_aoi_key(config) == "hormuz_core"
    assert hc.resolve_aoi(config, "hormuz_core")[1]["role"] == "primary"
    assert hc.resolve_aoi(config, "hormuz_narrow_strait")[1]["role"] == "sensitivity_nested_aoi"
    assert config["analysis"]["event_date"] == "2026-02-28"
    assert config["sar"]["unmatched_label"] == "unmatched"
    assert config["aoi"]["definitions"]["hormuz_narrow_strait"]["tuned_on_results"] is False


def test_event_aois_are_pre_specified_and_separate(config: dict) -> None:
    """Event AOIs are geography-defined, pre-specified and away from Hormuz."""
    for key in ("suez_canal_core", "bab_el_mandeb_core"):
        definition = hc.resolve_aoi(config, key)[1]
        assert definition["role"] == "event_aoi_multievent"
        assert definition["pre_specified"] is True
        assert definition["tuned_on_results"] is False
        bounds = hc.aoi_geometry(config, key).bounds
        assert bounds[2] < 45 or bounds[0] > 60  # not inside 54-58 E / Hormuz


def test_event_dates_come_from_documented_chronology(config: dict) -> None:
    """Event dates and windows are declared explicitly and are historical."""
    events = config["multievent"]["events"]
    assert events["suez_2021"]["event_date"] == "2021-03-23"
    assert events["suez_2021"]["event_window"] == ["2021-03-23", "2021-03-29"]
    assert events["red_sea_2023"]["event_date"] == "2023-12-15"
    for key, event in events.items():
        assert event["documented_chronology"].strip(), f"{key} lacks a documented chronology"
        assert event["placebo_years"], f"{key} lacks placebo years"


# ---------------------------------------------------------------------------
# window construction
# ---------------------------------------------------------------------------
def test_windows_tile_without_gaps_and_include_the_final_day(config: dict) -> None:
    """Pre/event/post must tile, and the last day of a window must be inside it."""
    event = config["multievent"]["events"]["suez_2021"]
    windows = screening.event_windows(config, event, _NullLogger())["event_year"]
    pre_start, pre_end = windows["pre_event"]
    event_start, event_end = windows["event"]
    post_start, post_end = windows["post_event"]
    assert pre_end == event_start
    assert event_end == post_start
    # the declared event window is 2021-03-23 .. 2021-03-29 inclusive
    assert event_start == pd.Timestamp("2021-03-23")
    assert event_end == pd.Timestamp("2021-03-30")  # exclusive
    assert post_end - post_start == pd.Timedelta(weeks=8)
    # a mid-afternoon detection on the final event day must fall inside the window
    late = pd.Timestamp("2021-03-29 15:55:00")
    assert event_start <= late < event_end


def test_placebo_windows_reuse_the_same_calendar_days(config: dict) -> None:
    """Placebo years must use the identical calendar window, shifted by years."""
    event = config["multievent"]["events"]["suez_2021"]
    windows = screening.event_windows(config, event, _NullLogger())
    placebo = windows["placebo_2019"]["event"]
    assert placebo[0] == pd.Timestamp("2019-03-23")
    assert placebo[1] == pd.Timestamp("2019-03-30")
    assert windows["placebo_2020"]["event"][0] == pd.Timestamp("2020-03-23")


# ---------------------------------------------------------------------------
# coverage denominators
# ---------------------------------------------------------------------------
def _scene_table(areas: list[float]) -> pd.DataFrame:
    """Minimal scene table with square intersections of known area."""
    rows = []
    boxes = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.0, 0.2)]
    for index, area in enumerate(areas):
        lon, lat = boxes[index % len(boxes)]
        wkt = (
            f"POLYGON (({lon} {lat}, {lon + 0.1} {lat}, {lon + 0.1} {lat + 0.1}, "
            f"{lon} {lat + 0.1}, {lon} {lat}))"
        )
        rows.append(
            {"scene_id": f"s{index}", "intersection_wkt": wkt, "coverage_km2": float(area)}
        )
    return pd.DataFrame(rows)


def test_water_coverage_uses_the_union_of_intersections() -> None:
    """The proxy is the union: identical scenes give one area, not two."""
    import geopandas as gpd
    import shapely

    wkt = (
        "POLYGON ((32.30 29.90, 32.50 29.90, 32.50 30.10, 32.30 30.10, 32.30 29.90))"
    )
    true_km2 = float(
        gpd.GeoSeries([shapely.from_wkt(wkt)], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0]
        / 1e6
    )
    single = pd.DataFrame(
        {"scene_id": ["a"], "intersection_wkt": [wkt], "coverage_km2": [true_km2]}
    )
    enriched, proxy_km2 = screening.add_water_coverage(single)
    assert proxy_km2 == pytest.approx(true_km2, rel=1e-6)
    assert enriched["coverage_fraction_water"].iloc[0] == pytest.approx(1.0, rel=1e-6)

    # Two identical footprints must not double the imageable area.
    duplicated = pd.concat([single, single.assign(scene_id="b")], ignore_index=True)
    enriched_dup, proxy_dup = screening.add_water_coverage(duplicated)
    assert proxy_dup == pytest.approx(true_km2, rel=1e-6)
    assert enriched_dup["coverage_fraction_water"].max() == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# period metrics and pipeline handling
# ---------------------------------------------------------------------------
def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scenes, detections and AIS status for one synthetic period."""
    wkt = "POLYGON ((32.3 29.9, 32.5 29.9, 32.5 30.1, 32.3 30.1, 32.3 29.9))"
    scenes = pd.DataFrame(
        {
            "scene_id": ["a", "b"],
            "date": pd.to_datetime(["2021-03-24", "2021-03-26"]),
            "week_key": ["2021-W12", "2021-W12"],
            "coverage_km2": [50.0, 50.0],
            "coverage_fraction": [0.05, 0.05],
            "coverage_fraction_water": [0.90, 0.90],
            "pipeline": ["n/a", "n/a"],
            "satellite_platform": ["S1A", "S1B"],
        }
    )
    detections = pd.DataFrame(
        {
            "scene_id": ["a", "a", "b"],
            "timestamp": pd.to_datetime(["2021-03-24 15:56", "2021-03-24 15:57", "2021-03-26 03:44"]),
            "pipeline": ["20231026"] * 3,
            "length_m": [100.0, 120.0, 80.0],
            "match_label": ["cargo", "unmatched", "unmatched"],
            "matched_flag": [True, False, False],
            "unmatched_flag": [False, True, True],
        }
    )
    status = pd.DataFrame(
        {
            "date": pd.date_range("2021-03-23", periods=7, freq="D"),
            "ais_observation_status": ["observed_positive"] * 7,
        }
    )
    return scenes, detections, status


def test_period_metrics_reads_the_pipeline_from_detections() -> None:
    """The pipeline token must come from the detection files, never the footprints."""
    scenes, detections, status = _frames()
    metrics = screening.period_metrics(
        scenes, detections, status, (pd.Timestamp("2021-03-23"), pd.Timestamp("2021-03-30")), 0.30
    )
    assert metrics["period_pipeline_versions"] == "20231026"
    assert metrics["period_scenes_usable_water"] == 2
    assert metrics["period_scenes_usable_box"] == 0
    assert metrics["period_detections"] == 3
    assert metrics["period_detections_classified"] == 3
    assert metrics["period_matched"] == 1 and metrics["period_unmatched"] == 2
    assert metrics["period_length_m_available_share"] == pytest.approx(1.0)
    assert metrics["period_matched_category_available_share"] == pytest.approx(1.0)
    assert metrics["period_ais_valid_days"] == 7


def test_period_metrics_excludes_detections_after_the_window_end() -> None:
    """A detection on the day after the window must not be counted."""
    scenes, detections, status = _frames()
    metrics = screening.period_metrics(
        scenes, detections, status, (pd.Timestamp("2021-03-23"), pd.Timestamp("2021-03-25")), 0.30
    )
    assert metrics["period_detections"] == 2  # the 2021-03-26 detection is excluded


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _summary(**overrides) -> dict:
    base = {
        "period_scenes_usable_water": 14,
        "period_scenes_intersecting": 18,
        "period_detections": 90,
        "period_acquisition_days": 10,
        "period_median_coverage_fraction_water": 0.90,
        "period_ais_valid_days": 63,
        "historical_placebo_available": True,
        "same_pipeline_pre_post": True,
        "pipeline_versions_pre": "20231026",
        "pipeline_versions_post": "20231026",
    }
    base.update(overrides)
    return base


def test_classification_strong_when_all_criteria_met(config: dict) -> None:
    """All thresholds met -> STRONG."""
    label, reason = screening.classify(_summary(), config["multievent"]["screening"])
    assert label == "STRONG"
    assert "sufficient" in reason


def test_classification_no_go_when_no_scene_covers_the_aoi(config: dict) -> None:
    """No usable scene inside the event window -> NO-GO."""
    label, _reason = screening.classify(
        _summary(period_scenes_usable_water=0), config["multievent"]["screening"]
    )
    assert label == "NO-GO"


def test_classification_no_go_on_pipeline_change_across_the_event(config: dict) -> None:
    """A pipeline change across the boundary makes the comparison unidentified."""
    label, reason = screening.classify(
        _summary(same_pipeline_pre_post=False, pipeline_versions_pre="20231026",
                 pipeline_versions_post="3"),
        config["multievent"]["screening"],
    )
    assert label == "NO-GO"
    assert "pipeline" in reason


def test_classification_conditional_on_thin_event_samples(config: dict) -> None:
    """Too few event detections -> CONDITIONAL, and the reason names the count."""
    label, reason = screening.classify(
        _summary(period_detections=57), config["multievent"]["screening"]
    )
    assert label == "CONDITIONAL"
    assert "57" in reason


def test_sar_only_classification_ignores_the_ais_gate(config: dict) -> None:
    """A blocked AIS fetch must not turn a SAR-STRONG event into a SAR failure."""
    summary = _summary(period_ais_valid_days=0)
    combined, combined_reason = screening.classify(summary, config["multievent"]["screening"])
    sar_only, _ = screening.classify(
        summary, config["multievent"]["screening"], include_ais=False
    )
    assert combined == "CONDITIONAL"
    assert "AIS" in combined_reason
    assert sar_only == "STRONG"


def test_screening_thresholds_are_declared_before_use(config: dict) -> None:
    """Every screening threshold must be config-driven, not hardcoded."""
    thresholds = config["multievent"]["screening"]
    for key in (
        "min_event_scenes",
        "min_event_detections",
        "min_event_acquisition_days",
        "min_median_coverage_fraction",
        "min_placebo_scenes",
    ):
        assert key in thresholds
