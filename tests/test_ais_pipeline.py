"""Tests for the Phase 2 AIS ingestion path (no network, synthetic fixtures)."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

ais = import_module("02_fetch_ais")


class _NullLogger:
    """Logger stub for table builders."""

    def info(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: D102
        pass

    def error(self, *args, **kwargs) -> None:  # noqa: D102
        pass


def _raw_frame() -> pd.DataFrame:
    """A GFW-shaped activity frame, using the real returned column names."""
    return pd.DataFrame(
        {
            "Lat": [26.43, 26.44, 26.45, 26.46, 26.47],
            "Lon": [56.43, 56.44, 56.45, 56.46, 56.47],
            "Time Range": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
                "2025-01-02",
                "2025-01-02",
            ],
            "Vessel ID": ["1", "2", "1", "3", "4"],
            "Flag": ["IRN", "PAN", "IRN", "UAE", "SGP"],
            "Gear Type": ["", "", "", "", ""],
            "Vessel Type": ["CARGO", "FISHING", "CARGO", "FISHING", "TANKER"],
            "MMSI": ["422000001", "353000002", "422000001", "470000003", "563000004"],
            "Vessel Presence Hours": [10.0, 4.0, 8.0, 2.0, 6.0],
        }
    )


# ---------------------------------------------------------------------------
# request planning
# ---------------------------------------------------------------------------
def test_month_chunks_cover_the_range_without_gaps() -> None:
    """Monthly chunking must be contiguous and inclusive at both ends."""
    chunks = ais.month_chunks(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-15"), 1)
    assert chunks[0][0] == pd.Timestamp("2025-01-01")
    assert chunks[-1][1] == pd.Timestamp("2025-03-15")
    for (_, previous_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == previous_end + pd.Timedelta(days=1)


def test_request_query_uses_pinned_dataset_and_upper_case_format(config: dict) -> None:
    """The query must pin the dataset version and use the accepted format value."""
    query = ais.build_request_query(
        config, config["ais"]["dataset"], pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")
    )
    assert query["datasets[0]"] == "public-global-presence:v4.0"
    assert "latest" not in query["datasets[0]"]
    assert query["date-range"] == "2026-03-01,2026-04-01"  # end is exclusive upstream
    assert query["temporal-resolution"] == "DAILY"
    assert query["format"] == "CSV"


def test_geojson_goes_in_the_body_not_the_query(config: dict) -> None:
    """geojson is rejected as a query parameter, so it must be in the body."""
    geometry = {"type": "Polygon", "coordinates": [[[56.25, 26.35], [56.65, 26.35], [56.65, 26.7], [56.25, 26.35]]]}
    body = ais.build_request_body(geometry)
    assert body["geojson"]["type"] == "Feature"
    assert body["geojson"]["geometry"] == geometry
    query = ais.build_request_query(
        config, config["ais"]["dataset"], pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")
    )
    assert "geojson" not in query


def test_api_date_range_compensates_for_exclusive_end() -> None:
    """An inclusive window must be sent as end+1 day."""
    assert ais.api_date_range(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31"), True) == "2026-03-01,2026-04-01"
    assert ais.api_date_range(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31"), False) == "2026-03-01,2026-03-31"


def test_group_by_rejects_values_the_api_does_not_accept(config: dict) -> None:
    """An invalid group-by must fail loudly instead of silently degrading."""
    bad = dict(config)
    bad["ais"] = dict(config["ais"], group_by=["vessel_type"])
    with pytest.raises(ValueError, match="not accepted"):
        ais.build_request_query(bad, "public-global-presence:v4.0", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02"))


def test_cache_path_is_deterministic(config: dict) -> None:
    """The same request maps to the same cache file."""
    first = ais.cache_path_for(config, "hormuz_core", "public-global-presence:v4.0",
                               pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31"))
    second = ais.cache_path_for(config, "hormuz_core", "public-global-presence:v4.0",
                                pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31"))
    assert first == second
    assert ":" not in first.name  # dataset separator must not leak into the filename


# ---------------------------------------------------------------------------
# schema detection
# ---------------------------------------------------------------------------
def test_role_detection_finds_documented_gfw_columns() -> None:
    """The role detector must resolve the real column names, not assumed ones."""
    roles = ais.detect_ais_roles(_raw_frame())
    assert roles["date"] == "Time Range"
    assert roles["presence_hours"] == "Vessel Presence Hours"
    assert roles["vessel_type"] == "Vessel Type"
    assert roles["vessel_id"] == "Vessel ID"
    assert roles["flag"] == "Flag"
    assert roles["mmsi"] == "MMSI"
    assert roles["lat"] == "Lat"
    assert roles["lon"] == "Lon"


def test_role_detection_handles_renamed_columns() -> None:
    """Alternative GFW spellings must still resolve."""
    frame = pd.DataFrame({"day": ["2025-01-01"], "geartype": ["cargo"], "presence": [3.0]})
    roles = ais.detect_ais_roles(frame)
    assert roles["date"] == "day"
    assert roles["vessel_type"] == "geartype"
    assert roles["presence_hours"] == "presence"


def test_missing_required_role_raises_clear_error(config: dict) -> None:
    """An unresolvable schema must stop processing with an actionable message."""
    frame = pd.DataFrame({"something": [1]})
    roles = ais.detect_ais_roles(frame)
    with pytest.raises(ValueError, match="does not resolve required role"):
        ais.build_daily_ais_table([frame], roles, "hormuz_core", _NullLogger())


# ---------------------------------------------------------------------------
# aggregation and validation
# ---------------------------------------------------------------------------
def test_daily_table_sums_hours_per_day_and_type() -> None:
    """Presence hours are summed per day and type, plus an all-types total."""
    roles = ais.detect_ais_roles(_raw_frame())
    daily = ais.build_daily_ais_table([_raw_frame()], roles, "hormuz_core", _NullLogger())
    cargo_first = daily[
        (daily["classification"] == "vessel_type")
        & (daily["class_value"] == "CARGO")
        & (daily["date"] == pd.Timestamp("2025-01-01"))
    ]
    assert cargo_first["presence_hours"].iloc[0] == pytest.approx(10.0)
    total_first = daily[
        (daily["classification"] == "all") & (daily["date"] == pd.Timestamp("2025-01-01"))
    ]
    assert total_first["presence_hours"].iloc[0] == pytest.approx(14.0)
    vessel_types = daily[daily["classification"] == "vessel_type"]["class_value"].nunique()
    assert vessel_types == 3  # CARGO, FISHING, TANKER


def test_daily_table_drops_rows_without_date_or_hours_but_reports_it() -> None:
    """Rows missing a date or a value are excluded, never filled with zero."""
    frame = pd.concat(
        [
            _raw_frame(),
            pd.DataFrame(
                {
                    "Time Range": ["2025-01-03", None],
                    "Vessel Type": ["CARGO", "CARGO"],
                    "Vessel Presence Hours": [None, 5.0],
                    "Vessel ID": ["9", "9"],
                }
            ),
        ],
        ignore_index=True,
    )
    roles = ais.detect_ais_roles(frame)
    daily = ais.build_daily_ais_table([frame], roles, "hormuz_core", _NullLogger())
    assert daily["date"].notna().all()
    assert pd.Timestamp("2025-01-03") not in set(daily["date"])


def test_day_coverage_without_a_report_marks_everything_missing_unknown() -> None:
    """With no parsed report interval, no day may be called a structural zero."""
    roles = ais.detect_ais_roles(_raw_frame())
    daily = ais.build_daily_ais_table([_raw_frame()], roles, "hormuz_core", _NullLogger())
    coverage = ais.validate_day_coverage(
        daily, "hormuz_core", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")
    )
    assert len(coverage) == 5
    assert set(coverage["ais_observation_status"]) == {"missing_unknown"}


def test_day_coverage_with_a_report_separates_zero_from_positive() -> None:
    """A covered day with an entry is observed; a covered day without one is a zero."""
    roles = ais.detect_ais_roles(_raw_frame())
    daily = ais.build_daily_ais_table([_raw_frame()], roles, "hormuz_core", _NullLogger())
    coverage = ais.build_observation_status(
        daily,
        ["ais_presence_hormuz_core_ds_2025-01-01_2025-01-05.zip"],
        "hormuz_core",
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-05"),
    )
    mapping = dict(zip(
        coverage["date"].dt.strftime("%Y-%m-%d"), coverage["ais_observation_status"]
    ))
    assert mapping["2025-01-01"] == "observed_positive"
    assert mapping["2025-01-03"] == "structural_zero"


def test_vessel_type_summary_is_a_share_of_the_total() -> None:
    """Relative contribution must sum to one per AOI."""
    roles = ais.detect_ais_roles(_raw_frame())
    daily = ais.build_daily_ais_table([_raw_frame()], roles, "hormuz_core", _NullLogger())
    summary = ais.build_classification_summary(daily, "vessel_type")
    assert summary["relative_contribution"].sum() == pytest.approx(1.0)
    assert "__ALL__" not in set(summary["class_value"])
    assert summary.iloc[0]["class_value"] == "CARGO"  # 18 h vs 6 h FISHING vs 6 h TANKER


def test_daily_table_keeps_three_classifications() -> None:
    """Vessel Type, Gear Type and the all-types total must all be available."""
    roles = ais.detect_ais_roles(_raw_frame())
    daily = ais.build_daily_ais_table([_raw_frame()], roles, "hormuz_core", _NullLogger())
    assert set(daily["classification"]) == {"vessel_type", "gear_type", "all"}
    all_rows = daily[daily["classification"] == "all"]
    assert set(all_rows["class_value"]) == {"__ALL__"}
    # the all-types total must equal the sum of the vessel-type rows per day
    vessel_rows = daily[daily["classification"] == "vessel_type"]
    per_day = vessel_rows.groupby("date")["presence_hours"].sum()
    for date, value in per_day.items():
        total = all_rows[all_rows["date"] == date]["presence_hours"].iloc[0]
        assert value == pytest.approx(total)


def test_extract_activity_csv_reads_the_largest_csv_member() -> None:
    """The report is a ZIP; the activity table is the largest .csv member."""
    import io as _io
    import zipfile as _zipfile

    buffer = _io.BytesIO()
    with _zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("layer-activity-data-0/public-global-presence-v4.0.csv",
                         _raw_frame().to_csv(index=False))
        archive.writestr("layer-activity-data-0/readme.md", "readme")
        archive.writestr("notes.csv", "a,b\n1,2\n")
    frame = ais.extract_activity_csv(buffer.getvalue(), _NullLogger())
    assert len(frame) == 5
    assert "Vessel Presence Hours" in frame.columns


def test_extract_activity_csv_rejects_non_zip_payloads() -> None:
    """A JSON error body must fail loudly, not be parsed as data."""
    with pytest.raises(ValueError, match="expected a ZIP"):
        ais.extract_activity_csv(b'{"statusCode":422}', _NullLogger())
