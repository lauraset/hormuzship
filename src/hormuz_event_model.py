"""Shared Phase 4B machinery: event windows, activity metrics and the matching model.

This module generalises what the frozen Hormuz steps did, without touching them:

* phase windows defined as calendar month-day windows per year role, with
  cross-year phases and leap-day clamping;
* SAR activity (scene-level and weekly density, historical-normalized anomaly);
* the AIS metric family kept deliberately separate - presence hours, distinct
  vessels, the same two restricted to SAR acquisition days, and a dwell proxy;
* a detection-level logistic model of confirmed AIS-SAR matching
  (``matched_category != "unmatched"``), with standard errors clustered by SAR
  scene and standardization to a chosen phase's covariate distribution.

Design choices that are fixed for Phase 4B
------------------------------------------
* ``matching_score`` and MMSI presence are never predictors: both come out of the
  matching process and are endogenous to the outcome.
* The period factor contains the event year's phases plus a single pooled
  ``placebo_reference`` level. A small AOI cannot support one level per
  placebo-year per phase, and the seasonal benchmark is obtained separately by
  fitting the same model on placebo years only.
* Coverage uses the geography rule from the screening phase: open-water
  chokepoints keep the bounding-box coverage, land-heavy corridors use the
  footprint-derived water-observable domain.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from hormuz_common import (
    PROJECT_ROOT,
    add_calendar_fields,
    aoi_scoped_path,
    read_csv_fast,
    resolve_path,
    safe_divide,
)

PLACEBO_LEVEL = "placebo_reference"


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def phase_window(
    year: int,
    start_md: str,
    end_md: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Resolve a month-day phase to ``(start, end_inclusive, end_exclusive)``.

    An end month-day earlier than the start month-day means the phase crosses the
    year boundary (Red Sea event phase 12-15 -> 02-15, Panama restricted phase
    11-03 -> 02-29). The end day is clamped to the last day of its month so
    ``02-29`` becomes ``02-28`` in a non-leap placebo year.
    """
    start = pd.Timestamp(year=int(year), month=int(start_md[:2]), day=int(start_md[3:]))
    end_year = int(year) + 1 if end_md < start_md else int(year)
    month, day = int(end_md[:2]), int(end_md[3:])
    last_day = pd.Period(f"{end_year}-{month:02d}").days_in_month
    end = pd.Timestamp(year=end_year, month=month, day=min(day, last_day))
    return start, end, end + pd.Timedelta(days=1)


def phase_windows(
    event: dict[str, Any],
    logger=None,
) -> dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]]:
    """All phase windows for the event year and every placebo year."""
    onset_year = int(pd.Timestamp(event["event_onset"]).year)
    roles: list[tuple[str, int]] = [(f"{onset_year}_event_year", onset_year)]
    for year in event.get("placebo_years", []):
        roles.append((f"{int(year)}_placebo", int(year)))
    offsets = event.get("phase_year_offsets", {}) or {}
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = {}
    for label, year in roles:
        windows[label] = {
            phase: phase_window(int(year) + int(offsets.get(phase, 0)), bounds[0], bounds[1])
            for phase, bounds in event["phases"].items()
        }
    if logger is not None:
        logger.info(
            "phases for %s: %s",
            event.get("event_label", event.get("case_role", "?")),
            {
                phase: f"{window[0].date()}..{window[1].date()}"
                for phase, window in windows[f"{onset_year}_event_year"].items()
            },
        )
    return windows


def event_year_label(event: dict[str, Any]) -> str:
    """Canonical label of the event-year role inside a ``phase_windows`` dict."""
    return f"{int(pd.Timestamp(event['event_onset']).year)}_event_year"


def phase_in_window(
    timestamps: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
) -> pd.Series:
    """Membership mask for a ``(start, end_inclusive, end_exclusive)`` window."""
    start, _end_inclusive, end_exclusive = window
    return (timestamps >= start) & (timestamps < end_exclusive)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_scene_table(cfg: dict[str, Any], aoi_key: str, tag: str | None) -> pd.DataFrame:
    """Load a (possibly window-tagged) scene metric table from step 03."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run steps 01 and 03 for AOI {aoi_key}")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    return frame


def load_detections(cfg: dict[str, Any], aoi_key: str, tag: str | None) -> pd.DataFrame:
    """Load AOI detections for the same window as the scene table."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet",
        aoi_key,
        cfg,
        tag=tag,
    )
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_ais_daily(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Load the per-day AIS table written by step 02 (may be empty)."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily.parquet", aoi_key, cfg
    )
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def load_ais_status(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Load the per-day AIS observation status (never treat missing as zero)."""
    path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily_status.parquet", aoi_key, cfg
    )
    if not path.exists():
        return pd.DataFrame(columns=["date", "ais_observation_status"])
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def water_domain_geometry(scenes: pd.DataFrame) -> tuple[Any, float]:
    """Return the footprint-derived water-observable polygon and its area in km2."""
    geometries = gpd.GeoSeries(
        shapely.from_wkt(scenes["intersection_wkt"].fillna("").to_numpy(), on_invalid="ignore"),
        crs="EPSG:4326",
    )
    usable = geometries[geometries.notna() & ~geometries.is_empty]
    if usable.empty:
        return None, 0.0
    union = usable.union_all()
    area = float(gpd.GeoSeries([union], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6)
    return union, area


def ensure_water_coverage(scenes: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add ``coverage_fraction_water`` from the footprint-derived water domain."""
    frame = scenes.copy()
    union, water_km2 = water_domain_geometry(frame)
    if union is None:
        frame["coverage_fraction_water"] = np.nan
        return frame, 0.0
    frame["coverage_fraction_water"] = frame["coverage_km2"] / water_km2 if water_km2 else np.nan
    return frame, water_km2


def load_ais_daily_in_domain(
    cfg: dict[str, Any],
    aoi_key: str,
    inside_geom,
    logger=None,
) -> pd.DataFrame:
    """Daily AIS table restricted to a geometric domain (the SAR-observable water).

    The AOI-level daily table produced by step 02 is already scoped to the API
    polygon, which for a land-heavy corridor contains water that Sentinel-1 does
    not image (Panama's Gatun Lake section holds ~20 % of AIS presence hours but
    has no SAR-observable water at all). Comparing an AIS series that includes
    those hours with a SAR series that cannot see them would be a spatial-domain
    artefact, so the Phase 4B AIS series are recomputed inside the same water
    domain the footprints define.
    """
    if inside_geom is None:
        return load_ais_daily(cfg, aoi_key)
    cache_dir = resolve_path(cfg["paths"]["ais_presence_dir"])
    records: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob(f"ais_presence_{aoi_key}_*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not members:
                    continue
                largest = max(members, key=lambda name: archive.getinfo(name).file_size)
                frame = pd.read_csv(io.StringIO(archive.read(largest).decode("utf-8")))
        except Exception as error:  # noqa: BLE001
            if logger is not None:
                logger.warning("could not read cached AIS report %s: %s", path.name, error)
            continue
        if frame.empty:
            continue
        date_col = next((c for c in frame.columns if str(c).strip().lower() in {"time range", "date"}), None)
        hours_col = next((c for c in frame.columns if "presence hours" in str(c).lower()), None)
        mmsi_col = next((c for c in frame.columns if str(c).strip().lower() in {"mmsi", "vessel id"}), None)
        if date_col is None or hours_col is None:
            continue
        inside = shapely.intersects(
            shapely.points(
                pd.to_numeric(frame["Lon"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(frame["Lat"], errors="coerce").to_numpy(dtype=float),
            ),
            inside_geom,
        )
        block = frame.loc[inside].copy()
        block["_date"] = pd.to_datetime(block[date_col], errors="coerce").dt.normalize()
        block["_hours"] = pd.to_numeric(block[hours_col], errors="coerce")
        block["_mmsi"] = block[mmsi_col].astype("string") if mmsi_col else pd.NA
        records.append(block)
    if not records:
        return pd.DataFrame(columns=["date", "presence_hours", "n_vessels"])
    # Cached report windows can overlap (a longer re-download plus the original
    # 12-month windows), so byte-identical rows must be collapsed before
    # aggregating. Deduplication must use the FULL row: a vessel can contribute
    # several rows for the same day with the same hourly value in different
    # presence cells, and collapsing on (date, hours, mmsi) alone would delete
    # those legitimate hours.
    combined = (
        pd.concat(records, ignore_index=True)
        .dropna(subset=["_date"])
        .drop_duplicates()
    )
    daily = combined.groupby("_date", as_index=False).agg(
        presence_hours=("_hours", "sum"), n_vessels=("_mmsi", "nunique")
    )
    daily = daily.rename(columns={"_date": "date"})
    daily["classification"] = "all"
    daily["class_value"] = "__ALL__"
    if logger is not None:
        logger.info(
            "AIS domain-restricted daily table for %s: %d days, %.0f hours",
            aoi_key,
            len(daily),
            float(daily["presence_hours"].sum()),
        )
    return daily


def load_vessel_days(
    cfg: dict[str, Any],
    aoi_key: str,
    logger=None,
    inside_geom: shapely.geometry.base.BaseGeometry | None = None,
    cache_suffix: str = "",
) -> pd.DataFrame:
    """Per (date, MMSI) presence extracted from the cached raw AIS reports.

    The daily table already carries a distinct-vessel count, but a *weekly*
    distinct-vessel series cannot be obtained by summing daily counts (a vessel
    present on five days would be counted five times). The raw cached reports
    contain the MMSI per vessel-day, so the exact weekly unique count is rebuilt
    here and cached as a parquet to avoid re-parsing on every run.

    ``inside_geom`` optionally restricts vessel-days to the SAR-observable water
    domain, so that the AIS and SAR dimensions describe the same water.
    """
    cache_path = (
        resolve_path(cfg["paths"]["interim_dir"])
        / f"ais_vessel_days__{aoi_key}{cache_suffix}.parquet"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    cache_dir = resolve_path(cfg["paths"]["ais_presence_dir"])
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob(f"ais_presence_{aoi_key}_*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not members:
                    continue
                largest = max(members, key=lambda name: archive.getinfo(name).file_size)
                raw = archive.read(largest).decode("utf-8")
            frame = pd.read_csv(io.StringIO(raw))
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            if logger is not None:
                logger.warning("could not read cached AIS report %s: %s", path.name, error)
            continue
        if frame.empty:
            continue
        date_col = next((c for c in frame.columns if str(c).strip().lower() in {"time range", "date"}), None)
        mmsi_col = next((c for c in frame.columns if str(c).strip().lower() in {"mmsi", "vessel id"}), None)
        if date_col is None or mmsi_col is None:
            continue
        block = pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
                "mmsi": frame[mmsi_col].astype("string"),
            }
        )
        lat_col = next((c for c in frame.columns if str(c).strip().lower() in {"lat", "latitude"}), None)
        lon_col = next((c for c in frame.columns if str(c).strip().lower() in {"lon", "lng", "longitude"}), None)
        if inside_geom is not None and lat_col and lon_col:
            inside = shapely.intersects(
                shapely.points(
                    pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(dtype=float),
                ),
                inside_geom,
            )
            block = block.loc[inside]
        frames.append(block)
    if not frames:
        out = pd.DataFrame(columns=["date", "mmsi"])
    else:
        out = pd.concat(frames, ignore_index=True).dropna(subset=["date", "mmsi"]).drop_duplicates()
    out.to_parquet(cache_path, index=False)
    if logger is not None:
        logger.info(
            "vessel-day table for %s: %d rows, %s..%s, %d distinct MMSI",
            aoi_key,
            len(out),
            out["date"].min().date() if len(out) else "-",
            out["date"].max().date() if len(out) else "-",
            out["mmsi"].nunique() if len(out) else 0,
        )
    return out


# ---------------------------------------------------------------------------
# SAR activity
# ---------------------------------------------------------------------------
def sar_activity(
    scenes: pd.DataFrame,
    event: dict[str, Any],
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]],
    coverage_threshold: float,
    density_scale: float,
    epsilon: float,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scene-level and weekly SAR density with historical-normalized anomalies.

    Returns ``(scene_table, weekly_table)``. Density is always normalised by the
    covered area, so partially imaging scenes stay comparable; a week without an
    acquisition is simply absent, never zero.
    """
    metric = str(event.get("coverage_metric", "coverage_fraction"))
    frame = scenes[scenes[metric] >= coverage_threshold].copy()
    frame["sar_density"] = safe_divide(frame["n_total"], frame["coverage_km2"]) * density_scale
    frame = add_calendar_fields(frame, "date")
    roles = list(windows)
    event_role = event_year_label(event)
    frame["year_role"] = "outside_windows"
    for role in roles:
        mask = pd.Series(False, index=frame.index)
        for window in windows[role].values():
            mask |= phase_in_window(frame["date"], window)
        frame.loc[mask, "year_role"] = role
    frame["phase"] = "outside"
    for role in roles:
        for phase, window in windows[role].items():
            mask = phase_in_window(frame["date"], window)
            frame.loc[mask, "phase"] = phase

    weekly = (
        frame.groupby(["year_role", "phase", "iso_year", "iso_week", "week_key", "week_start"],
                      as_index=False)
        .agg(
            n_scenes=("scene_id", "size"),
            sar_density_median=("sar_density", "median"),
            sar_density_mean=("sar_density", "mean"),
            sar_detections=("n_total", "sum"),
            coverage_km2_sum=("coverage_km2", "sum"),
        )
    )
    # historical expectation per ISO week from the placebo roles only
    placebo_weeks = weekly[weekly["year_role"] != event_role]
    expected = (
        placebo_weeks.groupby("iso_week", as_index=False)["sar_density_median"]
        .median()
        .rename(columns={"sar_density_median": "expected_sar_density"})
    )
    weekly = weekly.merge(expected, how="left", on="iso_week")
    weekly["sar_activity_anomaly"] = np.log(
        (weekly["sar_density_median"] + epsilon) / (weekly["expected_sar_density"] + epsilon)
    )
    weekly["is_event_year"] = weekly["year_role"] == event_role
    if logger is not None:
        logger.info(
            "SAR activity: %d usable scenes, %d weeks (%d event-year)",
            len(frame),
            len(weekly),
            int(weekly["is_event_year"].sum()),
        )
    return frame, weekly


def phase_summary(
    weekly: pd.DataFrame,
    event: dict[str, Any],
    value_columns: Iterable[str],
    phase_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean of each weekly metric per (year role, phase), event year labelled."""
    event_role = event_year_label(event)
    order = list(phase_order or event["phases"].keys())
    rows: list[dict[str, Any]] = []
    for role, group in weekly.groupby("year_role"):
        for phase in order:
            subset = group[group["phase"] == phase]
            if subset.empty:
                continue
            row: dict[str, Any] = {
                "year_role": role,
                "phase": phase,
                "is_event_year": role == event_role,
                "n_weeks": int(len(subset)),
                "n_scenes": int(subset["n_scenes"].sum()) if "n_scenes" in subset else 0,
            }
            for column in value_columns:
                if column in subset:
                    row[column] = float(subset[column].mean(skipna=True))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# matching model: detection-level logistic, scene-clustered, standardized
# ---------------------------------------------------------------------------
def cell_labels(lat: pd.Series, lon: pd.Series, degrees: float) -> pd.Series:
    """Fixed 0.05-degree style cell label from a pre-specified grid size."""
    lat_idx = np.floor(pd.to_numeric(lat, errors="coerce") / degrees).astype("Int64")
    lon_idx = np.floor(pd.to_numeric(lon, errors="coerce") / degrees).astype("Int64")
    return lat_idx.astype("string") + "_" + lon_idx.astype("string")


def build_model_frame(
    scenes: pd.DataFrame,
    detections: pd.DataFrame,
    event: dict[str, Any],
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]],
    cfg: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detection-level modelling frame with the event's period structure.

    ``period`` labels the event-year phases and pools every placebo year into one
    ``placebo_reference`` level, because a small AOI cannot support one level per
    placebo-year per phase. ``placebo_period`` keeps the finer year x phase labels
    so the placebo-only seasonal benchmark can be fitted separately.
    """
    model_cfg = cfg["phase4b"]["analysis"]
    metric = str(event.get("coverage_metric", "coverage_fraction"))
    threshold = float(cfg["sar"]["min_scene_coverage_fraction"])
    usable = scenes[scenes[metric] >= threshold]
    frame = detections[detections["scene_id"].isin(set(usable["scene_id"]))].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    n_before = len(frame)

    # outcome: confirmed AIS-SAR match, unknown labels excluded and counted
    known = frame["matched_flag"].notna() | frame["unmatched_flag"].notna()
    n_unknown = int((~known).sum())
    frame = frame[known].copy()
    frame["matched_flag"] = frame["matched_flag"].astype(bool).astype(int)

    bins = [float(value) for value in cfg["identification"]["length_bins_m"]]
    length = pd.to_numeric(frame["length_m"], errors="coerce")
    n_missing_length = int(length.isna().sum())
    frame = frame[length.notna()].copy()
    frame["length_bin"] = pd.cut(
        pd.to_numeric(frame["length_m"], errors="coerce"), bins=bins, right=False
    ).astype(str)

    degrees = float(model_cfg["spatial_cell_degrees"])
    frame["cell_raw"] = cell_labels(frame["lat"], frame["lon"], degrees)
    counts = frame["cell_raw"].value_counts()
    keep_cells = set(counts[counts >= int(model_cfg["min_detections_per_cell"])].index)
    frame["cell"] = np.where(frame["cell_raw"].isin(keep_cells), frame["cell_raw"], "cell_other")

    event_role = event_year_label(event)
    frame["period"] = PLACEBO_LEVEL
    frame["placebo_period"] = None
    frame["phase"] = "outside"
    frame["year_role"] = "outside_windows"
    for role, per_phase in windows.items():
        role_mask = pd.Series(False, index=frame.index)
        for phase, window in per_phase.items():
            mask = phase_in_window(frame["timestamp"], window)
            role_mask |= mask
            if role == event_role:
                frame.loc[mask, "period"] = phase
                frame.loc[mask, "phase"] = phase
            else:
                year = int(role.split("_")[0])
                frame.loc[mask, "placebo_period"] = f"{year}_{phase}"
                frame.loc[mask, "phase"] = phase
        frame.loc[role_mask, "year_role"] = role
    n_outside = int((frame["period"] == PLACEBO_LEVEL).sum() - frame["placebo_period"].notna().sum())
    frame = frame[frame["year_role"] != "outside_windows"].copy()

    diagnostics = {
        "aoi_key": event["aoi_key"],
        "coverage_metric": metric,
        "coverage_threshold": threshold,
        "n_detections_usable_scenes": int(n_before),
        "n_dropped_unknown_match_label": n_unknown,
        "n_dropped_missing_length": n_missing_length,
        "n_dropped_outside_windows": int(n_outside),
        "n_model_rows": int(len(frame)),
        "n_scenes": int(frame["scene_id"].nunique()),
        "n_cells_kept": int(len(keep_cells)),
        "n_cells_pooled": int(len(counts) - len(keep_cells)),
        "period_counts": frame["period"].value_counts().to_dict(),
        "phase_scene_counts": frame.groupby("phase")["scene_id"].nunique().to_dict(),
    }
    if logger is not None:
        logger.info(
            "model frame %s/%s: %d detections, %d scenes, %d cells, periods %s",
            event["aoi_key"],
            event["event_onset"],
            len(frame),
            diagnostics["n_scenes"],
            diagnostics["n_cells_kept"],
            diagnostics["period_counts"],
        )
    return frame, diagnostics


def fit_detection_logit(
    frame: pd.DataFrame,
    formula: str,
    cluster_variable: str,
    max_iterations: int,
    logger=None,
):
    """Logistic GLM with standard errors clustered by SAR scene."""
    import warnings

    import patsy
    import statsmodels.api as sm

    y_matrix, design = patsy.dmatrices(formula, frame, return_type="dataframe")
    groups = frame[cluster_variable].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(np.asarray(y_matrix, dtype=float).ravel(), design, family=sm.families.Binomial())
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, maxiter=max_iterations)
    if logger is not None:
        logger.info(
            "fitted %s: n=%d, clusters=%d, converged=%s",
            formula,
            int(result.nobs),
            int(groups.nunique()),
            bool(result.converged),
        )
    return result, design.design_info


def standardized_probabilities(
    result,
    design_info,
    frame: pd.DataFrame,
    labels: Sequence[str],
    target_label: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Standardized matching probabilities per period with simulation intervals.

    The covariate mix (length bin x spatial cell) is held at the ``target_label``
    phase; each period's factor level is then applied to that fixed mix. Intervals
    come from draws out of the cluster-robust covariance matrix.
    """
    import patsy

    baseline = frame[frame["period"] == target_label]
    if baseline.empty:
        return pd.DataFrame()
    combinations = (
        baseline.groupby(["length_bin", "cell"], observed=True).size().rename("weight").reset_index()
    )
    combinations["weight"] = combinations["weight"] / combinations["weight"].sum()
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    rng = np.random.default_rng(seed)
    beta_draws = rng.multivariate_normal(params, covariance, size=int(draws))

    def design_for(label: str):
        new_data = combinations[["length_bin", "cell"]].copy()
        new_data["period"] = label
        return np.asarray(
            patsy.build_design_matrices([design_info], new_data, return_type="dataframe")[0],
            dtype=float,
        )

    weights = combinations["weight"].to_numpy(dtype=float)
    reference_design = design_for(target_label)
    reference_draws = (
        1.0 / (1.0 + np.exp(-(reference_design @ beta_draws.T)))
    ).T @ weights
    records: list[dict[str, Any]] = []
    for label in labels:
        design = design_for(label)
        point = float(np.sum(weights * (1.0 / (1.0 + np.exp(-(design @ params))))))
        simulated = (1.0 / (1.0 + np.exp(-(design @ beta_draws.T)))).T @ weights
        difference = 100.0 * (simulated - reference_draws)
        subset = frame[frame["period"] == label]
        records.append(
            {
                "period": label,
                "n_detections": int(len(subset)),
                "n_scenes": int(subset["scene_id"].nunique()) if len(subset) else 0,
                "observed_matching_rate": float(subset["matched_flag"].mean())
                if len(subset) else np.nan,
                "standardized_matching_probability": point,
                "standardized_ci_low": float(np.percentile(simulated, 2.5)),
                "standardized_ci_high": float(np.percentile(simulated, 97.5)),
                "standardized_change_vs_target_pp": 100.0 * (point - float(np.mean(reference_draws))),
                "standardized_change_ci_low_pp": float(np.percentile(difference, 2.5)),
                "standardized_change_ci_high_pp": float(np.percentile(difference, 97.5)),
                "standardization_target": target_label,
                "coefficient_draws": int(draws),
            }
        )
    return pd.DataFrame(records)


def coefficients_to_frame(
    result,
    aoi_key: str,
    event_key: str,
    model_name: str,
    formula: str,
    note: str,
) -> pd.DataFrame:
    """Coefficient table (log-odds, odds ratios, cluster-robust intervals)."""
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    conf = np.asarray(result.conf_int(), dtype=float)
    return pd.DataFrame(
        {
            "event": event_key,
            "aoi_key": aoi_key,
            "model": model_name,
            "formula": formula,
            "term": list(result.params.index),
            "log_odds": params,
            "std_error": bse,
            "odds_ratio": np.exp(params),
            "odds_ratio_ci_low": np.exp(conf[:, 0]),
            "odds_ratio_ci_high": np.exp(conf[:, 1]),
            "p_value": np.asarray(result.pvalues, dtype=float),
            "n_observations": float(result.nobs),
            "converged": bool(result.converged),
            "note": note,
        }
    )


# ---------------------------------------------------------------------------
# AIS metric family (kept deliberately separate; never combined into an index)
# ---------------------------------------------------------------------------
def ais_metrics(
    ais_daily: pd.DataFrame,
    status: pd.DataFrame,
    vessel_days: pd.DataFrame,
    sar_day_table: pd.DataFrame,
    event: dict[str, Any],
    windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]],
    epsilon: float,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly AIS metric family, kept deliberately separate (never one index).

    Metrics: ``presence_hours``, ``distinct_vessels``,
    ``presence_hours_on_sar_days`` (mean per observed acquisition day),
    ``distinct_vessels_on_sar_days`` and the dwell proxy
    ``presence_hours_per_vessel``.

    Missing days are handled explicitly: ``structural_zero`` days count as
    observed days with zero activity; ``missing_unknown`` days are excluded from
    both numerator and denominator.
    """
    event_role = event_year_label(event)
    if ais_daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    totals = (
        ais_daily[ais_daily["classification"] == "all"][
            ["date", "presence_hours", "n_vessels"]
        ]
        .groupby("date", as_index=False)
        .agg(presence_hours=("presence_hours", "sum"), n_vessels=("n_vessels", "max"))
    )
    if len(status):
        usable_status = status[
            status["ais_observation_status"].isin(["observed_positive", "structural_zero"])
        ]
        totals = totals[totals["date"].isin(set(usable_status["date"]))]
        zeros = sorted(
            set(usable_status.loc[usable_status["ais_observation_status"] == "structural_zero", "date"])
            - set(totals["date"])
        )
        if zeros:
            totals = pd.concat(
                [
                    totals,
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(zeros),
                            "presence_hours": 0.0,
                            "n_vessels": 0.0,
                        }
                    ),
                ],
                ignore_index=True,
            )
    acquisition_days = (
        set(pd.to_datetime(sar_day_table["date"]).dt.normalize())
        if len(sar_day_table) else set()
    )
    totals["is_sar_day"] = totals["date"].isin(acquisition_days)
    totals = add_calendar_fields(totals, "date")
    totals["year_role"] = "outside_windows"
    totals["phase"] = "outside"
    for role, per_phase in windows.items():
        for phase, window in per_phase.items():
            mask = phase_in_window(totals["date"], window)
            totals.loc[mask, "phase"] = phase
            totals.loc[mask, "year_role"] = role
    week_keys = ["iso_year", "iso_week", "week_key", "week_start"]

    vessels = (
        add_calendar_fields(vessel_days.copy(), "date") if len(vessel_days) else vessel_days
    )
    if len(vessels):
        vessels["is_sar_day"] = vessels["date"].isin(acquisition_days)
        vessels["year_role"] = "outside_windows"
        vessels["phase"] = "outside"
        for role, per_phase in windows.items():
            for phase, window in per_phase.items():
                mask = phase_in_window(vessels["date"], window)
                vessels.loc[mask, "phase"] = phase
                vessels.loc[mask, "year_role"] = role

    weekly: list[pd.DataFrame] = []

    # all days: hours and distinct vessels are different metrics, never merged
    hours_all = totals.groupby(week_keys, as_index=False).agg(
        presence_hours=("presence_hours", "sum"), days_observed=("date", "nunique")
    )
    hours_all["metric"] = "presence_hours"
    hours_all["value"] = hours_all["presence_hours"]
    weekly.append(hours_all)

    if len(vessels):
        counts_all = vessels.groupby(week_keys, as_index=False)["mmsi"].nunique().rename(
            columns={"mmsi": "distinct_vessels"}
        )
        counts_all["metric"] = "distinct_vessels"
        counts_all["value"] = counts_all["distinct_vessels"]
        counts_all = counts_all.merge(
            hours_all[week_keys + ["presence_hours", "days_observed"]], how="left", on=week_keys
        )
        counts_all["presence_hours_per_vessel"] = safe_divide(
            counts_all["presence_hours"], counts_all["distinct_vessels"]
        )
        weekly.append(counts_all)

    sar_hours = totals[totals["is_sar_day"]]
    if len(sar_hours):
        sar_group = sar_hours.groupby(week_keys, as_index=False).agg(
            presence_hours=("presence_hours", "sum"),
            days_observed=("date", "nunique"),
        )
        sar_group["metric"] = "presence_hours_on_sar_days"
        sar_group["value"] = safe_divide(sar_group["presence_hours"], sar_group["days_observed"])
        if len(vessels):
            counts_sar = vessels[vessels["is_sar_day"]].groupby(week_keys, as_index=False)[
                "mmsi"
            ].nunique().rename(columns={"mmsi": "distinct_vessels"})
            sar_group = sar_group.merge(counts_sar, how="left", on=week_keys)
        weekly.append(sar_group)
        if len(vessels):
            vessels_only = counts_sar.copy()
            vessels_only["metric"] = "distinct_vessels_on_sar_days"
            vessels_only["value"] = vessels_only["distinct_vessels"]
            weekly.append(vessels_only)

    combined = pd.concat(weekly, ignore_index=True)
    combined["year_role"] = "outside_windows"
    combined["phase"] = "outside"
    for role in windows:
        role_mask = pd.Series(False, index=combined.index)
        for phase, window in windows[role].items():
            mask = phase_in_window(combined["week_start"], window)
            combined.loc[mask, "phase"] = phase
            role_mask |= mask
        combined.loc[role_mask, "year_role"] = role
    combined["is_event_year"] = combined["year_role"] == event_role
    combined = combined[combined["year_role"] != "outside_windows"].copy()

    placebo = combined[~combined["is_event_year"]]
    expected = (
        placebo.groupby(["metric", "iso_week"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "expected_value"})
    )
    combined = combined.merge(expected, how="left", on=["metric", "iso_week"])
    combined["ais_activity_anomaly"] = np.log(
        (combined["value"] + epsilon) / (combined["expected_value"] + epsilon)
    )

    phase_rows: list[dict[str, Any]] = []
    for (role, phase), group in combined.groupby(["year_role", "phase"]):
        by_metric = group.groupby("metric")["value"].mean()
        hours = float(group.loc[group["metric"] == "presence_hours", "value"].sum())
        distinct = by_metric.get("distinct_vessels", np.nan)
        # the dwell proxy lives on the distinct-vessel rows, not on `value`
        dwell_rows = group[group["metric"] == "distinct_vessels"]
        dwell = (
            float(dwell_rows["presence_hours_per_vessel"].mean(skipna=True))
            if len(dwell_rows) and dwell_rows["presence_hours_per_vessel"].notna().any()
            else np.nan
        )
        # phase-level dwell: total hours divided by vessels counted once over the
        # whole phase, which avoids both week-over-week double counting and the
        # bias of averaging weekly ratios.
        if len(vessels):
            phase_vessels = vessels[(vessels["year_role"] == role) & (vessels["phase"] == phase)]
            unique_vessels = float(phase_vessels["mmsi"].nunique())
        else:
            unique_vessels = np.nan
        phase_hours = float(group.loc[group["metric"] == "presence_hours", "presence_hours"].sum())
        phase_dwell = phase_hours / unique_vessels if unique_vessels else np.nan
        phase_window = windows[role][phase]
        phase_days = int((phase_window[2] - phase_window[0]).days)
        daily_slice = totals[
            (totals["year_role"] == role) & (totals["phase"] == phase)
        ] if "year_role" in totals else pd.DataFrame()
        phase_rows.append(
            {
                "year_role": role,
                "phase": phase,
                "is_event_year": role == event_role,
                "n_weeks": int(group["iso_week"].nunique()),
                "ais_presence_hours_weekly_mean": by_metric.get("presence_hours", np.nan),
                "ais_distinct_vessels_weekly_mean": distinct,
                "ais_presence_hours_on_sar_days": by_metric.get(
                    "presence_hours_on_sar_days", np.nan
                ),
                "ais_distinct_vessels_on_sar_days": by_metric.get(
                    "distinct_vessels_on_sar_days", np.nan
                ),
                "ais_presence_hours_per_vessel": dwell,
                "ais_distinct_vessels_phase_unique": unique_vessels,
                "ais_presence_hours_per_vessel_phase": phase_dwell,
                "phase_days": phase_days,
                # length-normalised breadth: a longer phase can only accumulate
                # more distinct vessels, so the per-day rate is the comparable one
                "ais_distinct_vessels_per_day": unique_vessels / phase_days if phase_days else np.nan,
                "ais_presence_hours_daily_mean": float(daily_slice["presence_hours"].mean())
                if len(daily_slice) else np.nan,
                # date-exact snapshot: how many vessels are present at a time.
                # This is the measure to use for phases shorter than two weeks,
                # where an ISO-week grid cannot resolve the phase boundary.
                "ais_distinct_vessels_daily_mean": float(daily_slice["n_vessels"].mean())
                if len(daily_slice) and "n_vessels" in daily_slice else np.nan,
                "ais_presence_hours_phase_total": hours,
                "ais_presence_hours_anomaly": float(
                    group.loc[group["metric"] == "presence_hours", "ais_activity_anomaly"].mean()
                ),
                "ais_distinct_vessels_anomaly": float(
                    group.loc[group["metric"] == "distinct_vessels", "ais_activity_anomaly"].mean()
                ) if (group["metric"] == "distinct_vessels").any() else np.nan,
                "ais_distinct_vessels_sar_days_anomaly": float(
                    group.loc[
                        group["metric"] == "distinct_vessels_on_sar_days", "ais_activity_anomaly"
                    ].mean()
                ) if (group["metric"] == "distinct_vessels_on_sar_days").any() else np.nan,
            }
        )
    phases = pd.DataFrame(phase_rows)
    if logger is not None:
        logger.info(
            "AIS metrics: %d weekly rows across metrics %s",
            len(combined),
            sorted(combined["metric"].unique()),
        )
    return combined, phases


def raw_phase_change(
    frame: pd.DataFrame,
    pre_label: str,
    event_label: str,
    cluster: str,
    iterations: int,
    level: float,
    seed: int,
    min_units: int,
) -> dict[str, Any]:
    """Raw (unstandardized) pre->event matching change with a scene-bootstrap CI."""
    from hormuz_stats import bootstrap_rate_difference, pooled_rate

    pre = frame[frame["period"] == pre_label]
    event = frame[frame["period"] == event_label]
    pre_scene = pre.groupby(cluster)["matched_flag"].agg(["sum", "size"])
    event_scene = event.groupby(cluster)["matched_flag"].agg(["sum", "size"])
    interval = bootstrap_rate_difference(
        pre_scene["sum"].to_numpy(dtype=float),
        pre_scene["size"].to_numpy(dtype=float),
        event_scene["sum"].to_numpy(dtype=float),
        event_scene["size"].to_numpy(dtype=float),
        iterations=iterations,
        level=level,
        seed=seed,
        min_units_for_ci=min_units,
    )
    return {
        "raw_rate_pre": pooled_rate(
            pre_scene["sum"].to_numpy(dtype=float), pre_scene["size"].to_numpy(dtype=float)
        ),
        "raw_rate_event": pooled_rate(
            event_scene["sum"].to_numpy(dtype=float), event_scene["size"].to_numpy(dtype=float)
        ),
        "raw_change_pp": 100.0 * interval.estimate,
        "raw_change_ci_low_pp": 100.0 * interval.low,
        "raw_change_ci_high_pp": 100.0 * interval.high,
        "raw_ci_computed": interval.ci_computed,
        "n_pre_detections": int(len(pre)),
        "n_event_detections": int(len(event)),
        "n_pre_scenes": int(pre[cluster].nunique()),
        "n_event_scenes": int(event[cluster].nunique()),
    }


def placebo_benchmark(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    event: dict[str, Any],
    pre_label: str,
    event_label: str,
    logger=None,
) -> dict[str, Any]:
    """Seasonal benchmark: the same pre->event change measured in placebo years.

    The placebo years are pooled and fitted with a plain pre/event/post phase
    factor, then standardized to their own pre-phase distribution — the direct
    analogue of the frozen Hormuz reference-year comparison. Per-year raw changes
    give the descriptive placebo range.
    """
    model_cfg = cfg["phase4b"]["analysis"]
    placebo = frame[frame["placebo_period"].notna()].copy()
    if placebo.empty:
        return {"placebo_available": False}
    placebo["period"] = placebo["phase"]
    result: dict[str, Any] = {"placebo_available": True, "n_placebo_detections": int(len(placebo))}
    try:
        fitted, design_info = fit_detection_logit(
            placebo,
            model_cfg["formula"],
            str(model_cfg["cluster_variable"]),
            int(model_cfg["max_iterations"]),
            logger,
        )
        standardized = standardized_probabilities(
            fitted,
            design_info,
            placebo,
            labels=[pre_label, event_label],
            target_label=pre_label,
            draws=int(model_cfg["simulation_draws"]),
            seed=int(cfg["project"]["random_seed"]),
        )
        row = standardized[standardized["period"] == event_label]
        if len(row):
            result.update(
                {
                    "placebo_standardized_change_pp": float(
                        row["standardized_change_vs_target_pp"].iloc[0]
                    ),
                    "placebo_standardized_ci_low_pp": float(
                        row["standardized_change_ci_low_pp"].iloc[0]
                    ),
                    "placebo_standardized_ci_high_pp": float(
                        row["standardized_change_ci_high_pp"].iloc[0]
                    ),
                }
            )
    except Exception as error:  # noqa: BLE001 - reported as a limitation, not fatal
        result["placebo_model_error"] = f"{type(error).__name__}: {error}"
        if logger is not None:
            logger.warning("placebo model failed (%s); falling back to raw per-year changes", error)

    per_year: list[float] = []
    for year in event.get("placebo_years", []):
        subset = frame[frame["placebo_period"].astype("string").str.startswith(f"{int(year)}_")]
        if subset.empty:
            continue
        pre = subset[subset["phase"] == pre_label]
        event_rows = subset[subset["phase"] == event_label]
        if len(pre) < 10 or len(event_rows) < 10:
            continue
        per_year.append(
            100.0
            * (float(event_rows["matched_flag"].mean()) - float(pre["matched_flag"].mean()))
        )
    if per_year:
        result["placebo_raw_change_pp_by_year"] = per_year
        result["placebo_raw_range_pp"] = [min(per_year), max(per_year)]
    return result


