"""Tests for the Phase 4A-extension screening of two further candidate events."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import hormuz_common as hc

ext = import_module("11_phase4a_extension_screen")


class _NullLogger:
    """Logger stub."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# case roles and frozen specification
# ---------------------------------------------------------------------------
def test_extension_case_roles_are_fixed(config: dict) -> None:
    """Bab is a spillover case and Panama a contrast; neither is a control."""
    events = config["multievent"]["extension_events"]
    assert events["bab_mandeb_2026"]["case_role"] == "concurrent_spillover_case"
    assert events["panama_2023"]["case_role"] == "non_security_contrast"
    assert events["panama_2023"]["disruption_type"] == "climate_capacity_disruption"
    for event in events.values():
        joined = f"{event['case_role']} {event.get('case_role_note', '')}".lower()
        assert "control_group" not in joined.replace("never a control group", "")
        assert "negative_control" not in joined.replace("not a negative control", "")


def test_hormuz_specification_still_frozen(config: dict) -> None:
    """The extension must not move the primary AOI, event date or narrow AOI role."""
    assert hc.primary_aoi_key(config) == "hormuz_core"
    assert config["analysis"]["event_date"] == "2026-02-28"
    assert hc.resolve_aoi(config, "hormuz_narrow_strait")[1]["role"] == "sensitivity_nested_aoi"
    assert config["analysis"]["load_end"] == "2026-05-31"


def test_bab_uses_the_existing_aoi_and_hormuz_phases(config: dict) -> None:
    """Bab reuses the pre-specified Bab AOI and the frozen Hormuz calendar phases."""
    event = config["multievent"]["extension_events"]["bab_mandeb_2026"]
    assert event["aoi_key"] == "bab_el_mandeb_core"
    assert hc.resolve_aoi(config, "bab_el_mandeb_core")[1]["pre_specified"] is True
    assert event["phases"]["baseline"] == ["01-01", "02-27"]
    assert event["phases"]["acute"] == ["03-01", "03-27"]
    assert event["phases"]["persistent"] == ["03-28", "05-31"]
    assert event["primary_screen_end"] == "2026-05-31"
    assert event["pipeline_transition_at_onset"] is True


def test_panama_aoi_is_pre_specified_and_land_heavy(config: dict) -> None:
    """Panama's box is fixed, un-tuned, and must use water coverage."""
    definition = hc.resolve_aoi(config, "panama_canal_core")[1]
    assert definition["pre_specified"] is True
    assert definition["tuned_on_results"] is False
    bounds = hc.aoi_geometry(config, "panama_canal_core").bounds
    assert (round(bounds[0], 2), round(bounds[1], 2)) == (-79.95, 8.85)
    assert (round(bounds[2], 2), round(bounds[3], 2)) == (-79.50, 9.35)
    assert config["multievent"]["extension_events"]["panama_2023"][
        "pipeline_transition_at_onset"
    ] is False


def test_panama_onset_wording_avoids_absolute_claim(config: dict) -> None:
    """The November 2023 date is the regime onset, not the drought's beginning."""
    basis = config["multievent"]["extension_events"]["panama_2023"]["event_onset_basis"]
    assert "NOT the absolute beginning" in basis
    assert "restrictions before November 2023" in basis


# ---------------------------------------------------------------------------
# phase windows
# ---------------------------------------------------------------------------
def test_phase_window_clamps_feb_29_in_non_leap_years() -> None:
    """02-29 must become 02-28 in a non-leap placebo year instead of raising."""
    start, end_inclusive, end_exclusive = ext.phase_window(2021, "11-03", "02-29")
    assert start == pd.Timestamp("2021-11-03")
    assert end_inclusive == pd.Timestamp("2022-02-28")
    assert end_exclusive == pd.Timestamp("2022-03-01")


def test_phase_window_handles_the_year_boundary() -> None:
    """A phase whose end month-day precedes its start crosses into the next year."""
    start, end_inclusive, _ = ext.phase_window(2023, "11-03", "02-29")
    assert start == pd.Timestamp("2023-11-03")
    assert end_inclusive == pd.Timestamp("2024-02-29")  # 2024 is a leap year


def test_panama_adaptation_phase_uses_the_following_year(config: dict) -> None:
    """phase_year_offsets must put the adaptation phase in the year after onset."""
    event = config["multievent"]["extension_events"]["panama_2023"]
    windows = ext.event_windows(event, _NullLogger())
    pre = windows["event_year"]["pre"]
    restricted = windows["event_year"]["restricted"]
    adaptation = windows["event_year"]["adaptation"]
    assert pre[0] == pd.Timestamp("2023-09-01")
    assert restricted[0] == pd.Timestamp("2023-11-03")
    assert adaptation[0] == pd.Timestamp("2024-03-01")  # not 2023
    assert windows["placebo_2021"]["adaptation"][0] == pd.Timestamp("2022-03-01")
    # phases tile without gaps or overlap
    assert pre[2] == restricted[0]
    assert restricted[2] == adaptation[0]


# ---------------------------------------------------------------------------
# water-observable domain
# ---------------------------------------------------------------------------
def test_water_domain_is_the_union_and_is_stable() -> None:
    """Identical footprints give one domain area, and per-year robustness is exact."""
    import geopandas as gpd
    import shapely

    wkt = "POLYGON ((-79.95 8.85, -79.60 8.85, -79.60 9.10, -79.95 9.10, -79.95 8.85))"
    true_km2 = float(
        0.0
        or (
            gpd.GeoSeries([shapely.from_wkt(wkt)], crs="EPSG:4326")
            .to_crs("EPSG:6933")
            .area.iloc[0]
            / 1e6
        )
    )
    scenes = pd.DataFrame(
        {
            "scene_id": ["a", "b", "c", "d"],
            "date": pd.to_datetime(["2023-09-05", "2023-11-07", "2024-01-09", "2024-03-12"]),
            "intersection_wkt": [wkt] * 4,
            "coverage_km2": [true_km2] * 4,
            "coverage_fraction": [0.03] * 4,
        }
    )
    enriched, proxy_km2, robustness = ext.water_domain(scenes, _NullLogger())
    assert proxy_km2 == pytest.approx(true_km2, rel=1e-6)
    assert robustness["max_abs_relative_difference_vs_full_window"] == pytest.approx(0.0)
    assert enriched["coverage_fraction_water"].round(6).eq(1.0).all()
    assert (enriched["coverage_fraction_bbox"] < 0.1).all()


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _event_phase(**overrides) -> dict:
    base = {
        "usable_scenes": 19,
        "intersecting_scenes": 19,
        "detections": 205,
        "acquisition_days": 10,
        "acquisition_weeks": 10,
        "phase_days": 119,
        "median_coverage_water": 1.0,
        "ais_valid_days": 119,
        "ais_missing_unknown_days": 0,
        "ais_valid_day_share": 1.0,
        "pipeline_interpretable": True,
        "pipeline_pre": "20231026",
        "pipeline_event": "20231026",
        "pipeline_post": "20231026",
    }
    base.update(overrides)
    return base


def test_classification_does_not_depend_on_effect_size(config: dict) -> None:
    """A stable matching rate must still be STRONG feasibility."""
    screening = config["multievent"]["screening"]
    label, _ = ext.classify_sar(_event_phase(), True, screening)
    assert label == "STRONG"


def test_classification_conditional_on_thin_event_detections(config: dict) -> None:
    """50 detections (Bab) must be reported as CONDITIONAL with the count named."""
    screening = config["multievent"]["screening"]
    label, reason = ext.classify_sar(
        _event_phase(usable_scenes=7, detections=50, acquisition_days=5,
                     acquisition_weeks=5, phase_days=27),
        True,
        screening,
    )
    assert label == "CONDITIONAL"
    assert "50" in reason


def test_prolonged_phase_requires_acquisition_weeks(config: dict) -> None:
    """A multi-month phase is not judged on a single pass."""
    screening = config["multievent"]["screening"]
    label, reason = ext.classify_sar(
        _event_phase(acquisition_weeks=1), True, screening
    )
    assert label == "CONDITIONAL"
    assert "acquisition week" in reason


def test_pipeline_transition_at_onset_is_interpretable(config: dict) -> None:
    """A transition exactly at onset is a handled caveat, not a NO-GO."""
    screening = config["multievent"]["screening"]
    label, _ = ext.classify_sar(
        _event_phase(pipeline_interpretable=True), True, screening
    )
    assert label == "STRONG"
    label_no_go, reason = ext.classify_sar(
        _event_phase(pipeline_interpretable=False), True, screening
    )
    assert label_no_go == "CONDITIONAL"
    assert "pipeline" in reason


def test_ais_classification_uses_coverage_only(config: dict) -> None:
    """AIS availability is judged on report coverage, never on its values."""
    screening = config["multievent"]["screening"]
    assert ext.classify_ais(_event_phase(), screening)[0] == "STRONG"
    label, reason = ext.classify_ais(
        _event_phase(ais_valid_days=0, ais_missing_unknown_days=119,
                     ais_valid_day_share=0.0),
        screening,
    )
    assert label == "CONDITIONAL"
    assert "missing_unknown" in reason


def test_combined_status_is_the_weaker_of_the_two() -> None:
    """Combined feasibility can never be stronger than its weaker component."""
    assert ext.combine("STRONG", "STRONG") == "STRONG"
    assert ext.combine("STRONG", "CONDITIONAL") == "CONDITIONAL"
    assert ext.combine("CONDITIONAL", "NO-GO") == "NO-GO"
    assert ext.combine("NO-GO", "STRONG") == "NO-GO"


# ---------------------------------------------------------------------------
# matching screening helpers
# ---------------------------------------------------------------------------
def test_month_pair_change_matches_the_frozen_construction() -> None:
    """The month-pair helper reproduces a pooled pre/post rate difference."""
    cfg = {
        "stats": {"bootstrap_iterations": 200, "bootstrap_ci_level": 0.95,
                  "min_scenes_per_period": 3},
        "project": {"random_seed": 1},
    }
    detections = pd.DataFrame(
        {
            "scene_id": [f"f{i}" for i in range(10)] + [f"m{i}" for i in range(10)],
            "timestamp": pd.to_datetime(
                [f"2026-02-{i + 1:02d}" for i in range(10)]
                + [f"2026-03-{i + 1:02d}" for i in range(10)]
            ),
            "matched_flag": [True] * 9 + [False] + [True] * 5 + [False] * 5,
        }
    )
    row = ext.month_pair_change(cfg, detections, "2026-02", "2026-03", _NullLogger())
    assert row["pooled_rate_before"] == pytest.approx(0.9)
    assert row["pooled_rate_after"] == pytest.approx(0.5)
    assert row["delta_pp"] == pytest.approx(-40.0, abs=1e-6)
    assert row["ci_computed"]
