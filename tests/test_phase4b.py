"""Tests for the Phase 4B multi-event machinery."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import hormuz_common as hc

model = import_module("hormuz_event_model")
cross = import_module("13_phase4b_cross_event")


class _NullLogger:
    """Logger stub."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# frozen specification and the case matrix
# ---------------------------------------------------------------------------
def test_hormuz_specification_frozen(config: dict) -> None:
    """Phase 4B must not move any frozen Hormuz parameter."""
    assert hc.primary_aoi_key(config) == "hormuz_core"
    assert config["analysis"]["event_date"] == "2026-02-28"
    assert config["sar"]["unmatched_label"] == "unmatched"
    assert config["model"]["primary_formula"] if "model" in config else True
    assert config["models"]["primary_formula"].startswith("matched_flag ~ C(period)")
    assert config["aoi"]["definitions"]["hormuz_narrow_strait"]["role"] == "sensitivity_nested_aoi"


def test_case_matrix_is_frozen_and_labelled(config: dict) -> None:
    """Every case carries a role; Bab is not a control and Panama not a negative control."""
    events = config["phase4b"]["events"]
    assert set(events) == {
        "hormuz_2026", "redsea_2023", "panama_2023",
        "panama_transit_core_2023", "bab_mandeb_2026", "suez_2021",
    }
    assert events["hormuz_2026"]["case_role"] == "discovery_case"
    assert events["redsea_2023"]["case_role"] == "independent_security_validation"
    assert events["panama_2023"]["case_role"] == "non_security_contrast"
    assert events["bab_mandeb_2026"]["case_role"] == "concurrent_regional_falsification"
    assert events["suez_2021"]["case_role"] == "supplementary_physical_blockage"
    assert events["hormuz_2026"]["frozen_standardized_pp"] == pytest.approx(-11.538)
    assert events["hormuz_2026"]["standardize_to"] is None


def test_panama_transit_core_is_pre_specified(config: dict) -> None:
    """The nested Panama sensitivity AOI is fixed before results and un-tuned."""
    definition = hc.resolve_aoi(config, "panama_transit_core")[1]
    assert definition["pre_specified"] is True
    assert definition["tuned_on_results"] is False
    bounds = hc.aoi_geometry(config, "panama_transit_core").bounds
    assert (round(bounds[0], 2), round(bounds[1], 2)) == (-79.72, 8.88)
    assert (round(bounds[2], 2), round(bounds[3], 2)) == (-79.52, 9.10)
    primary = hc.aoi_geometry(config, "panama_canal_core")
    assert primary.contains(hc.aoi_geometry(config, "panama_transit_core"))


def test_coverage_metric_respects_geography(config: dict) -> None:
    """Open-water chokepoints keep the box metric; land-heavy corridors use water."""
    events = config["phase4b"]["events"]
    assert events["hormuz_2026"]["coverage_metric"] == "coverage_fraction"
    assert events["redsea_2023"]["coverage_metric"] == "coverage_fraction"
    assert events["panama_2023"]["coverage_metric"] == "coverage_fraction_water"
    assert events["suez_2021"]["coverage_metric"] == "coverage_fraction_water"


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def test_redsea_event_phase_crosses_the_year_boundary(config: dict) -> None:
    """12-15 -> 02-15 must span the new year, and the post phase must follow."""
    event = config["phase4b"]["events"]["redsea_2023"]
    windows = model.phase_windows(event, _NullLogger())
    event_role = model.event_year_label(event)
    pre, main, post = windows[event_role]["pre"], windows[event_role]["event"], windows[event_role]["post"]
    assert pre[0] == pd.Timestamp("2023-10-20") and pre[1] == pd.Timestamp("2023-12-14")
    assert main[0] == pd.Timestamp("2023-12-15") and main[1] == pd.Timestamp("2024-02-15")
    assert post[0] == pd.Timestamp("2024-02-16") and post[1] == pd.Timestamp("2024-04-15")
    assert pre[2] == main[0] and main[2] == post[0]  # no gaps


def test_placebo_windows_reuse_the_same_calendar_days(config: dict) -> None:
    """Placebo years must mirror the event-year calendar, shifted by years."""
    event = config["phase4b"]["events"]["redsea_2023"]
    windows = model.phase_windows(event, _NullLogger())
    assert windows["2021_placebo"]["event"][0] == pd.Timestamp("2021-12-15")
    assert windows["2022_placebo"]["event"][0] == pd.Timestamp("2022-12-15")
    assert windows["2021_placebo"]["post"][0] == pd.Timestamp("2022-02-16")


def test_panama_adaptation_is_in_the_following_year(config: dict) -> None:
    """The adaptation phase belongs to the year after the restriction onset."""
    event = config["phase4b"]["events"]["panama_2023"]
    windows = model.phase_windows(event, _NullLogger())
    event_role = model.event_year_label(event)
    assert windows[event_role]["restricted"][0] == pd.Timestamp("2023-11-03")
    assert windows[event_role]["restricted"][1] == pd.Timestamp("2024-02-29")
    assert windows[event_role]["adaptation"][0] == pd.Timestamp("2024-03-01")


def test_phase_window_clamps_leap_day() -> None:
    """02-29 becomes 02-28 in a non-leap placebo year."""
    start, end_inclusive, _ = model.phase_window(2021, "11-03", "02-29")
    assert end_inclusive == pd.Timestamp("2022-02-28")


# ---------------------------------------------------------------------------
# water domain and AIS scoping
# ---------------------------------------------------------------------------
def test_water_domain_is_geometric_and_result_independent() -> None:
    """The domain is the union of footprint intersections; no detections involved."""
    import geopandas as gpd
    import shapely

    wkt = "POLYGON ((43.20 12.40, 43.40 12.40, 43.40 12.60, 43.20 12.60, 43.20 12.40))"
    true_km2 = float(
        gpd.GeoSeries([shapely.from_wkt(wkt)], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0]
        / 1e6
    )
    scenes = pd.DataFrame(
        {
            "scene_id": ["a", "b"],
            "intersection_wkt": [wkt, wkt],
            "coverage_km2": [true_km2, true_km2 / 2],
            "coverage_fraction": [0.2, 0.2],
        }
    )
    enriched, area = model.ensure_water_coverage(scenes)
    assert area == pytest.approx(true_km2, rel=1e-6)
    # identical footprints do not double the domain; a half-area scene scores 0.5
    assert enriched["coverage_fraction_water"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert enriched["coverage_fraction_water"].iloc[1] == pytest.approx(0.5, rel=1e-6)


def test_ais_metric_family_keeps_metrics_separate(config: dict) -> None:
    """Presence hours and distinct vessels are separate series, never merged."""
    dates = pd.date_range("2023-10-20", periods=10, freq="D")
    ais_daily = pd.DataFrame(
        {
            "date": dates,
            "classification": "all",
            "presence_hours": 100.0,
            "n_vessels": 10,
        }
    )
    status = pd.DataFrame(
        {"date": dates, "ais_observation_status": "observed_positive"}
    )
    vessel_days = pd.DataFrame(
        {
            "date": np.repeat(dates[:5], 2),
            "mmsi": [f"v{i}" for i in range(10)],
        }
    )
    sar_days = pd.DataFrame({"date": dates[::3]})
    event = config["phase4b"]["events"]["redsea_2023"]
    windows = model.phase_windows(event, _NullLogger())
    weekly, phases = model.ais_metrics(
        ais_daily, status, vessel_days, sar_days, event, windows, 1.0, _NullLogger()
    )
    assert set(weekly["metric"]) >= {"presence_hours", "distinct_vessels"}
    hours = weekly[weekly["metric"] == "presence_hours"]
    vessels = weekly[weekly["metric"] == "distinct_vessels"]
    assert hours["value"].iloc[0] != vessels["value"].iloc[0]
    assert "ais_distinct_vessels_anomaly" in phases.columns


def test_structural_zero_days_count_as_observed(config: dict) -> None:
    """A covered day with no activity is a zero, not a missing day."""
    # start on an ISO week boundary inside the Red Sea pre phase (2023-10-20..12-14)
    dates = pd.date_range("2023-10-23", periods=5, freq="D")
    ais_daily = pd.DataFrame(
        {
            "date": dates[:3],
            "classification": "all",
            "presence_hours": 50.0,
            "n_vessels": 5,
        }
    )
    status = pd.DataFrame(
        {
            "date": dates,
            "ais_observation_status": ["observed_positive"] * 3 + ["structural_zero"] * 2,
        }
    )
    event = config["phase4b"]["events"]["redsea_2023"]
    windows = model.phase_windows(event, _NullLogger())
    weekly, phases = model.ais_metrics(
        ais_daily, status, pd.DataFrame(columns=["date", "mmsi"]),
        pd.DataFrame({"date": dates[:1]}), event, windows, 1.0, _NullLogger()
    )
    hours_rows = weekly[weekly["metric"] == "presence_hours"]
    assert hours_rows["days_observed"].sum() == 5  # the two zero days are counted


# ---------------------------------------------------------------------------
# synthesis invariants
# ---------------------------------------------------------------------------
def test_no_composite_index_is_built() -> None:
    """The synthesis must expose the two dimensions without combining them."""
    source = (hc.PROJECT_ROOT / "src" / "13_phase4b_cross_event.py").read_text(encoding="utf-8")
    assert "no_composite_index" in source
    for banned in ("resilience_index", "composite_score", "combined_index"):
        assert banned not in source


def test_cross_event_requires_within_event_changes_only(config: dict) -> None:
    """The synthesis reads per-event changes, never pooled raw matching levels."""
    source = (hc.PROJECT_ROOT / "src" / "13_phase4b_cross_event.py").read_text(encoding="utf-8")
    assert "standardized_matching_change_pp" in source
    assert "raw_matching_change_pp" in source
    # no code path reads a raw absolute matching rate across events
    assert "pooled_matching_rate" not in source


def test_case_narratives_cover_every_event(config: dict) -> None:
    """Every case in the matrix has a curated interpretation and strength label."""
    for key in config["phase4b"]["events"]:
        assert key in cross.CASE_NARRATIVE
        assert cross.CASE_NARRATIVE[key]["inference_strength"]
        assert cross.CASE_NARRATIVE[key]["main_interpretation"]
