"""Tests for AOI scoping, the bootstrap estimators and the robustness tables."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import hormuz_common as hc
import hormuz_stats as hs

robustness = import_module("03b_robustness_tables")


# ---------------------------------------------------------------------------
# AOI selection and scoped paths
# ---------------------------------------------------------------------------
def test_configured_aois_are_primary_first(config: dict) -> None:
    """The primary AOI is listed first and both AOIs are configured."""
    keys = hc.list_aois(config)
    assert keys[0] == hc.primary_aoi_key(config)
    assert "hormuz_core" in keys
    assert "hormuz_narrow_strait" in keys


def test_narrow_aoi_is_the_configured_fixed_box(config: dict) -> None:
    """The narrow AOI must be exactly the pre-specified bbox, not a tuned shape."""
    geom = hc.aoi_geometry(config, "hormuz_narrow_strait")
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    assert (round(min_lon, 6), round(min_lat, 6)) == (56.25, 26.35)
    assert (round(max_lon, 6), round(max_lat, 6)) == (56.65, 26.70)


def test_narrow_aoi_is_nested_inside_primary_aoi(config: dict) -> None:
    """A nested sensitivity AOI must lie inside the primary corridor."""
    primary = hc.aoi_geometry(config, "hormuz_core")
    narrow = hc.aoi_geometry(config, "hormuz_narrow_strait")
    assert primary.contains(narrow)
    assert hc.aoi_area_km2(config, narrow) < hc.aoi_area_km2(config, primary)


def test_unknown_aoi_key_fails_loudly(config: dict) -> None:
    """An unknown AOI key must raise instead of silently using the primary AOI."""
    with pytest.raises(KeyError):
        hc.resolve_aoi(config, "not_an_aoi")
    with pytest.raises((KeyError, FileNotFoundError)):
        hc.aoi_geometry(config, "not_an_aoi")


def test_aoi_scoped_path_keeps_primary_names(config: dict, project_root) -> None:
    """Primary deliverable filenames stay stable; other AOIs get a suffix."""
    base = "outputs/tables/data_coverage.csv"
    primary = hc.aoi_scoped_path(base, "hormuz_core", config)
    narrow = hc.aoi_scoped_path(base, "hormuz_narrow_strait", config)
    assert primary.name == "data_coverage.csv"
    assert narrow.name == "data_coverage__hormuz_narrow_strait.csv"
    assert narrow.parent == primary.parent


def test_aoi_scoped_path_handles_directory_like_names(config: dict) -> None:
    """Names without an extension still get a deterministic suffix."""
    out = hc.aoi_scoped_path("outputs/figures", "hormuz_narrow_strait", config)
    assert out.name == "figures__hormuz_narrow_strait"


# ---------------------------------------------------------------------------
# bootstrap estimators
# ---------------------------------------------------------------------------
def test_pooled_rate_basic() -> None:
    """Pooled rate is a ratio of sums, not a mean of ratios."""
    assert hs.pooled_rate(np.array([1.0, 1.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)
    assert np.isnan(hs.pooled_rate(np.array([0.0]), np.array([0.0])))


def test_bootstrap_pooled_rate_is_deterministic_and_brackets_estimate() -> None:
    """The same seed gives the same interval, which brackets the point estimate."""
    num = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 8.0, 9.0, 7.0])
    den = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    first = hs.bootstrap_pooled_rate(num, den, iterations=500, level=0.95, seed=7)
    second = hs.bootstrap_pooled_rate(num, den, iterations=500, level=0.95, seed=7)
    assert first.low == second.low and first.high == second.high
    assert first.low <= first.estimate <= first.high
    assert first.ci_computed


def test_bootstrap_skips_interval_for_tiny_samples() -> None:
    """A two-scene period must not produce a confident-looking interval."""
    interval = hs.bootstrap_pooled_rate(
        np.array([1.0, 0.0]), np.array([1.0, 1.0]), iterations=200, level=0.95, seed=1,
        min_units_for_ci=3,
    )
    assert not interval.ci_computed
    assert np.isnan(interval.low)
    assert interval.estimate == pytest.approx(0.5)


def test_bootstrap_rate_difference_detects_shift() -> None:
    """A large pre/post shift must produce a negative interval."""
    pre_num = np.full(20, 8.0)
    pre_den = np.full(20, 10.0)
    post_num = np.full(20, 2.0)
    post_den = np.full(20, 10.0)
    interval = hs.bootstrap_rate_difference(
        pre_num, pre_den, post_num, post_den, iterations=500, level=0.95, seed=3
    )
    assert interval.estimate == pytest.approx(-0.6)
    assert interval.high < 0


def test_bootstrap_rate_difference_identical_periods_centred_on_zero() -> None:
    """No shift must leave zero inside the interval."""
    num = np.array([5.0, 4.0, 6.0, 5.0, 5.0, 4.0])
    den = np.full(6, 10.0)
    interval = hs.bootstrap_rate_difference(
        num, den, num, den, iterations=500, level=0.95, seed=11
    )
    assert interval.estimate == pytest.approx(0.0)
    assert interval.low <= 0 <= interval.high


def test_bootstrap_log_ratio_uses_epsilon() -> None:
    """Zero current activity must stay finite thanks to epsilon."""
    interval = hs.bootstrap_log_ratio(
        [0.0, 0.0, 0.0], [10.0, 10.0, 10.0], epsilon=1.0, iterations=200, level=0.95, seed=5
    )
    assert np.isfinite(interval.estimate)
    assert interval.estimate < 0


# ---------------------------------------------------------------------------
# robustness table wiring
# ---------------------------------------------------------------------------
def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two scenes before the event and two after, with known matching rates."""
    scenes = pd.DataFrame(
        {
            "scene_id": ["s1", "s2", "s3", "s4"],
            "scene_date_parsed": pd.to_datetime(
                ["2026-01-05", "2026-02-05", "2026-03-10", "2026-04-10"]
            ),
            "coverage_km2": [1000.0] * 4,
            "coverage_fraction": [0.5] * 4,
            "n_total": [10, 10, 10, 10],
            "n_matched": [8, 8, 2, 2],
            "n_unmatched": [2, 2, 8, 8],
            "n_unknown": [0] * 4,
            "n_matched_mmsi": [9, 9, 4, 4],
            "sar_density": [10.0] * 4,
            "sar_density_matched": [8.0, 8.0, 2.0, 2.0],
            "sar_density_unmatched": [2.0, 2.0, 8.0, 8.0],
            "median_length_m": [100.0] * 4,
            "median_matching_score": [1.0] * 4,
            "passes_coverage_threshold": [True] * 4,
            "platform_has_detections": [True] * 4,
            "usable_for_primary_metrics": [True] * 4,
            "count_identity_ok": [True] * 4,
            "matching_rate": [0.8, 0.8, 0.2, 0.2],
            "matching_rate_mmsi": [0.9, 0.9, 0.4, 0.4],
            "has_detections": [True] * 4,
        }
    )
    scenes["week_key"] = pd.to_datetime(scenes["scene_date_parsed"]).dt.isocalendar().week.astype(str)
    detections = pd.DataFrame(
        {
            "scene_id": ["s1", "s2", "s3", "s4"],
            "timestamp": pd.to_datetime(
                ["2026-01-05", "2026-02-05", "2026-03-10", "2026-04-10"]
            ),
            "lat": [26.4, 26.5, 26.45, 26.45],
            "lon": [56.4, 56.5, 56.42, 56.42],
            "length_m": [150.0, 150.0, 40.0, 40.0],
            "matching_score": [5.0, 5.0, 0.0, 0.0],
            "matched_flag": [True, True, False, False],
            "unmatched_flag": [False, False, True, True],
            "unknown_flag": [False] * 4,
            "mmsi_matched_flag": [True, True, False, False],
        }
    )
    return scenes, detections


def test_length_bin_matching_flags_small_samples(config: dict) -> None:
    """Cells below the configured threshold are flagged and get no interval."""
    scenes, detections = _toy_inputs()
    frame = robustness.build_length_bin_matching(
        config, {"hormuz_core": (scenes, detections)}, logger=_NullLogger()
    )
    assert set(frame["phase"]) <= {"baseline", "acute_shock", "persistent_disruption"}
    assert frame["small_sample"].any()
    small = frame[frame["small_sample"]]
    assert not small["ci_computed"].any()


def test_aoi_sensitivity_reports_delta_for_event_year(config: dict) -> None:
    """The 2026 pre/post delta equals the realised rate difference."""
    scenes, detections = _toy_inputs()
    scenes = scenes.copy()
    scenes["aoi_key"] = "hormuz_core"
    frame = robustness.build_aoi_sensitivity(
        config, {"hormuz_core": (scenes, detections)}, logger=_NullLogger()
    )
    row = frame[(frame["row_type"] == "pre_post_split") & (frame["year_group"] == "2026")].iloc[0]
    assert row["delta_pre_post_pp"] == pytest.approx(-60.0, abs=1e-6)  # 0.2 - 0.8
    assert row["n_scenes_pre"] == 2 and row["n_scenes_post"] == 2
    assert not row["ci_computed"]  # only two scenes per side


def test_matching_definition_sensitivity_contains_both_definitions(config: dict) -> None:
    """Both the primary and the MMSI definition must appear, per period."""
    scenes, detections = _toy_inputs()
    frame = robustness.build_matching_definition_sensitivity(
        config, {"hormuz_core": (scenes, detections)}, logger=_NullLogger()
    )
    assert set(frame["definition"]) == {"primary_matched_category", "secondary_mmsi_notnull"}
    deltas = frame[frame["row_type"] == "pre_post_delta"]
    assert len(deltas) == 2 * len(config["analysis"]["years"])  # two definitions x three years
    event = deltas[deltas["year_group"] == "2026"]
    assert (event["delta_pre_post_pp"] < 0).all()
    # years with no toy data must be NaN, never silently zero
    empty_years = deltas[deltas["year_group"] != "2026"]
    assert empty_years["delta_pre_post_pp"].isna().all()


def test_dispersion_is_zero_for_single_detection_scenes(config: dict) -> None:
    """One detection per scene cannot have measurable dispersion."""
    scenes, detections = _toy_inputs()
    frame = robustness.build_detection_dispersion(
        config, {"hormuz_core": (scenes, detections)}, logger=_NullLogger()
    )
    month = frame[(frame["year"] == 2026) & (frame["month"] == 3)].iloc[0]
    assert month["median_standard_distance_km"] == 0.0
    assert month["n_detections"] == 1


class _NullLogger:
    """Minimal logger stub so table builders can be tested without log files."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass
