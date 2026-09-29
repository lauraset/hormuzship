"""Tests for the Phase 2 activity-anomaly logic (synthetic fixtures, no network)."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

activity = import_module("05_activity_anomaly")
figures = import_module("07_make_figures")


class _NullLogger:
    """Logger stub for table builders."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def error(self, *args, **kwargs) -> None:  # noqa: D102
        pass


def _ais_daily() -> pd.DataFrame:
    """Seven consecutive January days in 2025 with 10 h/day of presence."""
    dates = pd.date_range("2025-01-06", periods=7, freq="D")  # ISO week 2025-W02
    return pd.DataFrame(
        {
            "aoi_key": "hormuz_core",
            "date": dates,
            "classification": "all",
            "class_value": "__ALL__",
            "presence_hours": [10.0] * 7,
            "n_records": [5] * 7,
            "n_vessels": [4] * 7,
        }
    )


def _sar_days(dates: list[str]) -> pd.DataFrame:
    """SAR acquisition day table."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "n_scenes": 1,
            "aoi_covered_km2": 1000.0,
            "aoi_covered_fraction": 0.4,
        }
    )


def _status_frame(dates: list[str], start: str = "2025-01-06", end: str = "2025-01-12") -> pd.DataFrame:
    """Observation status with every calendar day covered and no structural zero."""
    calendar = pd.date_range(start, end, freq="D")
    observed = set(pd.to_datetime(dates))
    frame = pd.DataFrame({"aoi_key": "hormuz_core", "date": calendar})
    frame["is_covered_by_completed_report"] = True
    frame["has_activity_entry"] = frame["date"].isin(observed)
    frame["ais_observation_status"] = np.where(
        frame["has_activity_entry"], "observed_positive", "structural_zero"
    )
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    return frame


def _scene_metrics() -> pd.DataFrame:
    """Two usable scenes in the same ISO week with known coverage."""
    return pd.DataFrame(
        {
            "scene_id": ["s1", "s2"],
            "scene_date_parsed": pd.to_datetime(["2026-03-03", "2026-03-05"]),
            "week_key": ["2026-W10", "2026-W10"],
            "iso_year": [2026, 2026],
            "iso_week": [10, 10],
            "week_start": pd.to_datetime(["2026-03-02", "2026-03-02"]),
            "coverage_km2": [1000.0, 1000.0],
            "usable_for_primary_metrics": [True, True],
        }
    )


def _detections() -> pd.DataFrame:
    """Detections with a known length mix: 2 large on s1, 1 small on s2."""
    return pd.DataFrame(
        {
            "scene_id": ["s1", "s1", "s2"],
            "length_m": [120.0, 60.0, 30.0],
        }
    )


# ---------------------------------------------------------------------------
# AIS weekly series
# ---------------------------------------------------------------------------
def test_ais_weekly_all_days_sums_every_observed_day(config: dict) -> None:
    """AIS_all_days keeps all seven days and sums the hours."""
    weekly = activity.build_ais_weekly(
        _ais_daily(),
        _sar_days(["2025-01-06", "2025-01-08"]),
        _status_frame(["2025-01-06", "2025-01-08"]),
        config,
        "hormuz_core",
        _NullLogger(),
    )
    all_days = weekly[weekly["series"] == "AIS_all_days"].iloc[0]
    assert all_days["days_observed"] == 7
    assert all_days["presence_hours"] == pytest.approx(70.0)
    assert all_days["sufficient_days"]


def test_ais_weekly_sar_days_restricts_to_acquisition_days(config: dict) -> None:
    """AIS_SAR_days uses only days that have a usable SAR acquisition."""
    cfg = dict(config)
    cfg["ais"] = dict(config["ais"], min_observed_days_per_week=1)
    weekly = activity.build_ais_weekly(
        _ais_daily(),
        _sar_days(["2025-01-06", "2025-01-08"]),
        _status_frame(["2025-01-06", "2025-01-08"]),
        cfg,
        "hormuz_core",
        _NullLogger(),
    )
    sar_days = weekly[weekly["series"] == "AIS_SAR_days"].iloc[0]
    assert sar_days["days_observed"] == 2
    assert sar_days["presence_hours"] == pytest.approx(20.0)
    assert sar_days["mean_hours_per_observed_day"] == pytest.approx(10.0)


def test_ais_weekly_flags_thin_weeks_instead_of_zero_filling(config: dict) -> None:
    """A week with too few observed days becomes NaN, never a quiet zero.

    Only ``analysis_value`` (the figure used by the anomaly step) is masked; the
    raw ``presence_hours`` stay visible so the shortfall can be inspected.
    """
    cfg = dict(config)
    cfg["ais"] = dict(config["ais"], min_observed_days_per_week_sar_days=5)
    weekly = activity.build_ais_weekly(
        _ais_daily(),
        _sar_days(["2025-01-06", "2025-01-08"]),
        _status_frame(["2025-01-06", "2025-01-08"]),
        cfg,
        "hormuz_core",
        _NullLogger(),
    )
    sar_days = weekly[weekly["series"] == "AIS_SAR_days"].iloc[0]
    assert not sar_days["sufficient_days"]
    assert np.isnan(sar_days["analysis_value"])
    assert sar_days["presence_hours"] == pytest.approx(20.0)  # still visible


def test_ais_weekly_without_data_returns_empty(config: dict) -> None:
    """No AIS rows must yield an empty frame rather than a zero series."""
    weekly = activity.build_ais_weekly(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), config, "hormuz_core", _NullLogger()
    )
    assert weekly.empty


# ---------------------------------------------------------------------------
# SAR weekly variants
# ---------------------------------------------------------------------------
def test_sar_weekly_variants_normalise_by_covered_area(config: dict) -> None:
    """Density is detections per covered km2, per length subset."""
    frame = activity.build_sar_weekly_variants(
        _scene_metrics(), _detections(), config, _NullLogger()
    )
    total = frame[frame["sar_variant"] == "ALL_detections"].iloc[0]
    # 3 detections over 2000 covered km2 -> 1.5 per 1000 km2
    assert total["n_detections"] == 3
    assert total["density_weighted"] == pytest.approx(1.5)
    large50 = frame[frame["sar_variant"] == "length_ge_50m"].iloc[0]
    assert large50["n_detections"] == 2
    assert large50["density_weighted"] == pytest.approx(1.0)
    large100 = frame[frame["sar_variant"] == "length_ge_100m"].iloc[0]
    assert large100["n_detections"] == 1


def test_sar_weekly_variants_exclude_unusable_scenes(config: dict) -> None:
    """A scene flagged unusable must not contribute detections."""
    metrics = _scene_metrics().copy()
    scenes = metrics[metrics["usable_for_primary_metrics"]]
    frame = activity.build_sar_weekly_variants(scenes, _detections(), config, _NullLogger())
    total = frame[frame["sar_variant"] == "ALL_detections"].iloc[0]
    assert total["n_scenes"] == 2


# ---------------------------------------------------------------------------
# expectation and anomaly
# ---------------------------------------------------------------------------
def test_expected_by_iso_week_is_the_median_of_reference_years(config: dict) -> None:
    """Expected activity is the median across the reference years."""
    frame = pd.DataFrame(
        {
            "iso_year": [2024, 2024, 2025, 2025],
            "iso_week": [10, 10, 10, 10],
            "value": [4.0, 6.0, 20.0, 30.0],
        }
    )
    expected = activity.expected_by_iso_week(frame, "value", [2024, 2025], _NullLogger())
    assert expected["expected"].iloc[0] == pytest.approx(13.0)  # median(4, 6, 20, 30)
    assert expected["n_reference_years"].iloc[0] == 4


def test_add_anomaly_columns_matches_the_documented_formula() -> None:
    """log_anomaly = log((value + eps) / (expected + eps))."""
    frame = pd.DataFrame({"iso_year": [2026], "iso_week": [10], "value": [5.0]})
    expected = pd.DataFrame({"iso_week": [10], "expected": [10.0]})
    out = activity.add_anomaly_columns(frame, "value", expected, epsilon=0.0, event_year=2026)
    assert out["log_anomaly"].iloc[0] == pytest.approx(np.log(0.5))
    # epsilon keeps a zero-value week finite
    frame_zero = pd.DataFrame({"iso_year": [2026], "iso_week": [10], "value": [0.0]})
    out_zero = activity.add_anomaly_columns(frame_zero, "value", expected, epsilon=1.0, event_year=2026)
    assert np.isfinite(out_zero["log_anomaly"].iloc[0])


def test_add_anomaly_columns_keeps_missing_expectation_as_nan() -> None:
    """A week with no reference-year counterpart stays NaN, not zero."""
    frame = pd.DataFrame({"iso_year": [2026], "iso_week": [53], "value": [5.0]})
    expected = pd.DataFrame({"iso_week": [10], "expected": [10.0]})
    out = activity.add_anomaly_columns(frame, "value", expected, epsilon=1.0, event_year=2026)
    assert np.isnan(out["log_anomaly"].iloc[0])
    assert bool(out["expected_missing"].iloc[0])


# ---------------------------------------------------------------------------
# figures: break markers come from the config
# ---------------------------------------------------------------------------
def test_break_markers_include_event_and_technical_breaks(config: dict) -> None:
    """Every in-window break plus the event date must be annotated, in order."""
    markers = figures.break_markers(config)
    dates = [date for date, _ in markers]
    assert dates == sorted(dates)
    joined = " ".join(label for _, label in markers)
    assert "event onset" in joined
    assert "pipeline v3" in joined
    assert "mask" in joined
    assert pd.Timestamp("2026-02-28") in dates
    assert pd.Timestamp("2026-03-01") in dates
    assert pd.Timestamp("2026-03-28") in dates
