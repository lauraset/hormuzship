"""Tests for the Phase 3 identification machinery."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

import hormuz_common as hc

ais = import_module("02_fetch_ais")
matching = import_module("04_matching_analysis")
models = import_module("06_statistical_tests")
identification = import_module("09_identification_checks")


class _NullLogger:
    """Logger stub."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# pre-specified technical reference corridors
# ---------------------------------------------------------------------------
def test_reference_corridors_are_pre_specified_and_independent(config: dict) -> None:
    """Corridors must be pre-specified, labelled as technical, and not in Hormuz."""
    roles = {
        key: hc.resolve_aoi(config, key)[1]
        for key in hc.list_aois(config)
    }
    corridors = {k: v for k, v in roles.items() if v.get("role") == "technical_reference_corridor"}
    assert set(corridors) == {"gibraltar_strait", "singapore_strait", "english_channel_dover"}
    for key, definition in corridors.items():
        assert definition.get("pre_specified") is True
        assert definition.get("tuned_on_results") is False
        bounds = hc.aoi_geometry(config, key).bounds
        # Hormuz sits between 54 and 58 E; none of the reference boxes may touch it.
        assert bounds[2] < 30 or bounds[0] > 60, f"{key} overlaps the Hormuz region"


def test_primary_and_narrow_aoi_are_not_corridors(config: dict) -> None:
    """The Hormuz AOIs keep their own roles; corridors never reclassify them."""
    assert hc.resolve_aoi(config, "hormuz_core")[1]["role"] == "primary"
    assert hc.resolve_aoi(config, "hormuz_narrow_strait")[1]["role"] == "sensitivity_nested_aoi"
    assert hc.primary_aoi_key(config) == "hormuz_core"


# ---------------------------------------------------------------------------
# AIS observation status
# ---------------------------------------------------------------------------
def _daily_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "classification": "all",
            "class_value": "__ALL__",
            "date": pd.to_datetime(dates),
            "presence_hours": 10.0,
            "n_vessels": 3,
        }
    )


def test_observation_status_splits_zero_from_missing() -> None:
    """Covered days with no entry are structural zeros; uncovered days are missing."""
    daily = _daily_frame(["2026-03-01"])
    status = ais.build_observation_status(
        daily,
        ["ais_presence_x_ds_2026-03-01_2026-03-03.zip"],
        "x",
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-03-05"),
    )
    mapping = dict(zip(status["date"].dt.strftime("%Y-%m-%d"), status["ais_observation_status"]))
    assert mapping["2026-03-01"] == "observed_positive"
    assert mapping["2026-03-02"] == "structural_zero"   # covered, no entry
    assert mapping["2026-03-03"] == "structural_zero"   # covered, no entry
    assert mapping["2026-03-04"] == "missing_unknown"   # not covered by any report
    assert mapping["2026-03-05"] == "missing_unknown"


def test_observation_status_never_calls_an_uncovered_day_zero() -> None:
    """With no parsed report at all, every day must be missing_unknown."""
    status = ais.build_observation_status(
        pd.DataFrame(), [], "x", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-03")
    )
    assert set(status["ais_observation_status"]) == {"missing_unknown"}


def test_cache_interval_round_trip() -> None:
    """The cache filename is the auditable record of a completed report interval."""
    interval = ais.parse_cache_interval(
        "ais_presence_hormuz_core_public-global-presence_v4.0_2026-03-01_2026-03-31.zip"
    )
    assert interval == (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31"))
    assert ais.parse_cache_interval("not-a-cache-file.csv") is None


# ---------------------------------------------------------------------------
# matching terminology and the MMSI-present cross-tab
# ---------------------------------------------------------------------------
def _detections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_id": ["s1", "s1", "s1", "s2"],
            "timestamp": pd.to_datetime(
                ["2026-02-01", "2026-02-01", "2026-03-05", "2026-03-05"]
            ),
            "matched_flag": [True, False, False, False],
            "unmatched_flag": [False, True, True, True],
            "mmsi": [422000001.0, 422000002.0, np.nan, 422000004.0],
            "matching_score": [5.0, 0.001, 0.0, 0.002],
            "length_m": [150.0, 40.0, 60.0, 200.0],
            "match_label": ["cargo", "unmatched", "unmatched", "unmatched"],
        }
    )


def test_crosstab_separates_the_three_matching_groups(config: dict) -> None:
    """matched+MMSI, unmatched+MMSI and unmatched+no-MMSI must be distinguished."""
    cfg = dict(config)
    table = matching.definition_crosstab(_detections(), cfg, "hormuz_core", _NullLogger())
    groups = set(table["definition_group"])
    assert "matched_category_matched__mmsi_present" in groups
    assert "matched_category_unmatched__mmsi_present" in groups
    assert "matched_category_unmatched__mmsi_absent" in groups


def test_mmsi_present_unmatched_group_has_near_zero_match_quality(config: dict) -> None:
    """The diagnostic group must be identifiable as low-confidence candidates."""
    table = matching.definition_crosstab(_detections(), config, "hormuz_core", _NullLogger())
    row = table[
        (table["definition_group"] == "matched_category_unmatched__mmsi_present")
        & (table["grouping"] == "year")
    ].iloc[0]
    assert row["matching_score_p95"] < 1.0
    assert row["share_matching_score_below_1"] == pytest.approx(1.0)


def test_terminology_is_recorded_as_mmsi_present_diagnostic(config: dict) -> None:
    """The MMSI definition must never be labelled an alternative match definition."""
    text = (hc.PROJECT_ROOT / "src" / "04_matching_analysis.py").read_text(encoding="utf-8")
    assert "MMSI-present diagnostic" in text
    # the term must not survive as a label or an identifier anywhere in the code
    assert "alternative_matching_definition" not in text
    assert "alternative_matching" not in text


# ---------------------------------------------------------------------------
# detection-level model and standardization
# ---------------------------------------------------------------------------
def _model_frame() -> pd.DataFrame:
    """Synthetic detections with a strong period effect and a length gradient."""
    rng = np.random.default_rng(0)
    rows = []
    for period, base in (
        ("2024_reference", 0.6),
        ("2025_reference", 0.6),
        ("2026_baseline", 0.6),
        ("2026_acute", 0.1),
        ("2026_persistent", 0.2),
    ):
        for scene in range(6):
            for _ in range(40):
                large = rng.random() < 0.5
                probability = base + (0.3 if large else 0.0)
                rows.append(
                    {
                        "scene_id": f"{period}_s{scene}",
                        "matched_flag": int(rng.random() < probability),
                        "length_m": 120.0 if large else 30.0,
                        "length_bin": "[100,150)" if large else "[0,50)",
                        "cell": "1_1" if large else "2_2",
                        "period": period,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["period"] = pd.Categorical(frame["period"], categories=list(models.PERIOD_LABELS))
    return frame


def test_detection_logit_clusters_by_scene(config: dict) -> None:
    """The fitted model must report one cluster per scene."""
    frame = _model_frame()
    result, _design = models.fit_detection_logit(
        frame, config["models"]["primary_formula"], "scene_id", 200, _NullLogger()
    )
    assert result.cov_type == "cluster"
    assert frame["scene_id"].nunique() == 30
    assert float(result.nobs) == len(frame)


def test_standardization_keeps_the_event_decline(config: dict) -> None:
    """Standardizing composition must not remove a genuine period effect."""
    frame = _model_frame()
    result, design_info = models.fit_detection_logit(
        frame, config["models"]["primary_formula"], "scene_id", 200, _NullLogger()
    )
    standardized = models.standardized_probabilities(result, design_info, frame, config, _NullLogger())
    values = dict(zip(standardized["period"], standardized["standardized_matching_probability"]))
    assert values["2026_baseline"] > values["2026_acute"]
    assert values["2026_baseline"] > values["2026_persistent"]
    acute = standardized[standardized["period"] == "2026_acute"].iloc[0]
    assert acute["standardized_difference_vs_baseline_pp"] < 0
    assert acute["standardized_difference_ci_high_pp"] < 0


def test_standardization_reports_the_baseline_composition(config: dict) -> None:
    """The standardization target must be the configured pre-event baseline."""
    frame = _model_frame()
    result, design_info = models.fit_detection_logit(
        frame, config["models"]["primary_formula"], "scene_id", 200, _NullLogger()
    )
    standardized = models.standardized_probabilities(result, design_info, frame, config, _NullLogger())
    assert set(standardized["standardization_baseline"]) == {"2026_baseline"}
    baseline = standardized[standardized["period"] == "2026_baseline"].iloc[0]
    assert baseline["standardized_difference_vs_baseline_pp"] == pytest.approx(0.0)


def test_model_formula_excludes_endogenous_predictors(config: dict) -> None:
    """matching_score and MMSI presence must not appear in any model formula."""
    for key in ("primary_formula", "length_only_formula", "grouped_formula"):
        formula = str(config["models"][key])
        assert "matching_score" not in formula
        assert "mmsi" not in formula.lower()


# ---------------------------------------------------------------------------
# composition and falsification helpers
# ---------------------------------------------------------------------------
def test_total_variation_bounds() -> None:
    """TVD is 0 for identical distributions and 1 for disjoint ones."""
    a = pd.Series({"x": 3, "y": 1})
    assert identification.total_variation(a, a) == pytest.approx(0.0)
    assert identification.total_variation(pd.Series({"x": 4}), pd.Series({"y": 4})) == pytest.approx(1.0)


def test_standardised_change_uses_the_before_mix_and_flags_thin_bins() -> None:
    """Direct standardization reweights to the before mix and drops thin bins."""
    before = pd.DataFrame(
        {
            "length_m": [30.0] * 100 + [120.0] * 100,
            "matched_flag": [True] * 50 + [False] * 50 + [True] * 20 + [False] * 80,
        }
    )
    after = pd.DataFrame(
        {
            "length_m": [30.0] * 10 + [120.0] * 100,
            "matched_flag": [True] * 5 + [False] * 5 + [True] * 10 + [False] * 90,
        }
    )
    result = identification.standardised_rate_change(
        before, after, [0, 50, 100, 200], min_per_bin=30, logger=_NullLogger()
    )
    # only the [100,200) bin reaches 30 observations in both periods
    assert "[0,50)" in result["standardised_bins_excluded"]
    assert "[50,100)" in result["standardised_bins_excluded"]
    assert "[100,200)" in result["standardised_bins_used"]
    # only the [100,200) bin is usable, so the standardised change equals that bin's change
    assert result["standardised_delta_pp"] == pytest.approx(100 * (0.10 - 0.20), abs=1e-6)


def test_composition_shift_uses_the_event_year_only(config: dict) -> None:
    """Pre/post composition must be measured inside the event year."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-06-01", "2026-02-01", "2026-03-10", "2026-05-01"]
            ),
            "length_m": [100.0, 120.0, 60.0, 55.0],
            "presence_score": [1.0, 1.0, 1.0, 1.0],
            "fishing_score": [0.1, 0.1, 0.2, 0.2],
            "lat": [26.4, 26.4, 26.4, 26.4],
            "lon": [56.4, 56.4, 56.4, 56.4],
            "match_label": ["cargo", "cargo", "unmatched", "unmatched"],
        }
    )
    table = identification.composition_shift(frame, config, "hormuz_core", _NullLogger())
    distance = table[table["level"] == "__distribution_distance__"]
    lengths = table[(table["variable"] == "length_m") & (table["level"] == "median")]
    assert distance["n_detections"].max() == 3  # the 2024 row is excluded
    assert set(lengths["period"]) == {"pre_2026_baseline", "post_2026_event"}
