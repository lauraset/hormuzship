"""STEP 1 - ingest GFW Sentinel-1 SAR footprints and detections for the AOI.

What this script produces (all inside ``data/interim``)
-----------------------------------------------------
``sar_scenes.parquet``
    One row per SAR scene (acquisition) whose footprint intersects the AOI:
    ``scene_id``, ``date``, ``acquisition_start/end``, ``pipeline``,
    ``satellite_platform``, ``footprint_area_km2``, ``intersection_km2``,
    ``coverage_km2``, ``coverage_fraction``, ``geometry_valid`` and the
    AOI-intersection geometry as WKT (for the coverage map).

``sar_detections_aoi.parquet``
    Every detection that falls inside the AOI, with explicit match flags
    (``match_status`` in {matched, unmatched, unknown}) plus the two match
    definitions kept side by side (GFW label vs MMSI presence).

``sar_daily_aoi_coverage.parquet``
    Per calendar date: how much of the AOI was actually imaged (union of all
    scene intersections) and by how many scenes. This is the denominator used
    for coverage-normalised activity, and it is what makes "no acquisition"
    days distinguishable from "zero vessels" days.

``outputs/diagnostics/*``
    Filter ledgers (before/after/removed/reason for **every** filter), the
    coverage-fraction distribution used to calibrate the scene threshold, and
    orphan-detection diagnostics (detections whose scene is missing from the
    footprint table).

Notes
-----
* Dates with no SAR acquisition are simply absent from these tables. They are
  never written as zero vessels; downstream code must treat them as NA.
* Only ``int64`` counts and geometry-derived areas are computed here; ratios,
  matching rates and anomalies are computed in step 03.

Usage
-----
    python src/01_prepare_sar.py
    python src/01_prepare_sar.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    FilterLedger,
    add_calendar_fields,
    aoi_area_km2,
    aoi_geometry,
    aoi_scoped_path,
    banner,
    bbox_of,
    discover_sar_files,
    ensure_dirs,
    files_in_window,
    get_logger,
    load_config,
    load_schema_mapping,
    label_or_na,
    list_aois,
    measure_intersections,
    missing_tokens_mask,
    parse_gfw_datetimes,
    parse_scene_platform,
    points_within,
    primary_aoi_key,
    read_csv_fast,
    resolve_path,
    write_json,
)

FOOTPRINT_COLUMNS = ["scene_id", "date", "start_time", "end_time", "footprint_wkt"]
# The documented GFW detection column order, used only as a fallback when a file
# has lost its header row (always logged; see hormuz_common.read_csv_fast).
DETECTION_COLUMNS = [
    "scene_id",
    "timestamp",
    "lat",
    "lon",
    "presence_score",
    "length_m",
    "mmsi",
    "matching_score",
    "fishing_score",
    "matched_category",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest GFW SAR data for the Hormuz AOI")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aoi", default=None,
                        help="configured AOI key (default: the primary AOI)")
    parser.add_argument("--start", default=None,
                        help="override analysis.load_start (for historical event windows)")
    parser.add_argument("--end", default=None,
                        help="override analysis.load_end and the analysis cut-off")
    parser.add_argument("--limit-months", type=int, default=None,
                        help="debug helper: only process the first N months")
    parser.add_argument("--window-tag", default=None,
                        help="optional suffix so a second calendar window for the "
                             "same AOI does not overwrite the first run's artifacts")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# footprints
# ---------------------------------------------------------------------------
def process_footprint_file(
    path: Path,
    pipeline: str,
    aoi_geom,
    cfg: dict[str, Any],
    ledger: FilterLedger,
    logger,
) -> pd.DataFrame:
    """Read one monthly footprint file and return AOI-intersecting scene rows."""
    table = read_csv_fast(
        path, FOOTPRINT_COLUMNS, expected_columns=FOOTPRINT_COLUMNS, logger=logger
    )
    n_read = len(table)
    if table.attrs.get("header_missing"):
        ledger.record(
            step=f"source_file_header_missing:{path.name}",
            before=n_read,
            after=n_read,
            reason=(
                "file has no header row in the archive; read positionally with the "
                "documented GFW column order - verify the date coverage of this file"
            ),
        )

    table = table.rename(
        columns={
            "date": "scene_date",
            "start_time": "acquisition_start",
            "end_time": "acquisition_end",
            "footprint_wkt": "footprint_wkt",
        }
    )

    geometries = shapely.from_wkt(table["footprint_wkt"].to_numpy(), on_invalid="ignore")
    valid_mask = np.array([geom is not None and not geom.is_empty for geom in geometries])
    table = table.loc[valid_mask].reset_index(drop=True)
    geometries = geometries[valid_mask]
    n_after_wkt = len(table)
    ledger.record(
        step=f"footprint_wkt_parse:{path.name}",
        before=n_read,
        after=n_after_wkt,
        reason="could not be parsed into a non-empty polygon (invalid/blank WKT)",
    )
    if n_after_wkt == 0:
        return pd.DataFrame()

    gdf = gpd.GeoDataFrame(table, geometry=gpd.GeoSeries(geometries, crs="EPSG:4326"))
    geoms = gdf.geometry

    min_lon, min_lat, max_lon, max_lat = bbox_of(aoi_geom)
    bbox_mask = (
        (geoms.bounds["minx"] <= max_lon)
        & (geoms.bounds["maxx"] >= min_lon)
        & (geoms.bounds["miny"] <= max_lat)
        & (geoms.bounds["maxy"] >= min_lat)
    )
    gdf = gdf.loc[bbox_mask].reset_index(drop=True)
    ledger.record(
        step=f"footprint_bbox_filter:{path.name}",
        before=n_after_wkt,
        after=len(gdf),
        reason="footprint bounding box does not overlap the AOI bounding box",
    )
    if gdf.empty:
        return pd.DataFrame()

    metrics = measure_intersections(
        gdf.geometry,
        aoi_geom,
        equal_area_crs=cfg["aoi"]["equal_area_crs"],
        source_crs="EPSG:4326",
    )
    before_intersection = len(gdf)
    intersects = metrics["intersection_km2"].fillna(0.0) > 0
    ledger.record(
        step=f"footprint_intersection_filter:{path.name}",
        before=before_intersection,
        after=int(intersects.sum()),
        reason="footprint polygon does not intersect the AOI polygon (bbox overlap only)",
    )

    out = pd.concat([gdf.drop(columns="geometry").reset_index(drop=True), metrics], axis=1)
    out = out.loc[intersects.to_numpy()].reset_index(drop=True)
    invalid_count = int((~out["geometry_valid"]).sum())
    if invalid_count:
        logger.warning("%s: %d invalid footprint geometries were repaired or dropped", path.name, invalid_count)

    out["coverage_km2"] = out["intersection_km2"]
    out["pipeline"] = pipeline
    out["source_file"] = path.name
    out["satellite_platform"] = out["scene_id"].map(parse_scene_platform)
    out["scene_date_parsed"] = pd.to_datetime(out["scene_date"], errors="coerce")
    out["acquisition_start_parsed"] = parse_gfw_datetimes(out["acquisition_start"], logger)
    out["acquisition_end_parsed"] = parse_gfw_datetimes(out["acquisition_end"], logger)
    return out


# ---------------------------------------------------------------------------
# detections
# ---------------------------------------------------------------------------
def process_detection_file(
    path: Path,
    pipeline: str,
    aoi_geom,
    mapping: dict[str, Any],
    cfg: dict[str, Any],
    ledger: FilterLedger,
    logger,
) -> pd.DataFrame:
    """Read one monthly detection file and keep only detections inside the AOI."""
    datetime_col = mapping["datetime"]
    lat_col = mapping["latitude"]
    lon_col = mapping["longitude"]
    scene_col = mapping["scene_id"]
    match_col = mapping.get("match_label")
    unmatched_value = mapping.get("match_value_unmatched")
    mmsi_col = mapping.get("mmsi")

    table = read_csv_fast(
        path,
        expected_columns=DETECTION_COLUMNS,
        logger=logger,
    )
    n_read = len(table)
    if table.attrs.get("header_missing"):
        ledger.record(
            step=f"source_file_header_missing:{path.name}",
            before=n_read,
            after=n_read,
            reason=(
                "file has no header row in the archive; read positionally with the "
                "documented GFW column order"
            ),
        )
    missing = [column for column in [scene_col, datetime_col, lat_col, lon_col] if column not in table.columns]
    if missing:
        raise KeyError(
            f"{path.name}: missing required column(s) {missing} according to the schema mapping; "
            "re-run src/00_inspect_data.py"
        )

    lat = pd.to_numeric(table[lat_col], errors="coerce")
    lon = pd.to_numeric(table[lon_col], errors="coerce")
    coordinate_ok = lat.notna() & lon.notna()
    ledger.record(
        step=f"detection_coordinate_validity:{path.name}",
        before=n_read,
        after=int(coordinate_ok.sum()),
        reason="latitude/longitude missing or non-numeric (kept as excluded, not filled)",
    )
    table = table.loc[coordinate_ok].reset_index(drop=True)
    lat = lat.loc[coordinate_ok].reset_index(drop=True)
    lon = lon.loc[coordinate_ok].reset_index(drop=True)

    min_lon, min_lat, max_lon, max_lat = bbox_of(aoi_geom, cfg["sar"]["detection_bbox_buffer_deg"])
    bbox_mask = lat.between(min_lat, max_lat) & lon.between(min_lon, max_lon)
    table = table.loc[bbox_mask].reset_index(drop=True)
    lat = lat.loc[bbox_mask].reset_index(drop=True)
    lon = lon.loc[bbox_mask].reset_index(drop=True)
    ledger.record(
        step=f"detection_bbox_filter:{path.name}",
        before=int(coordinate_ok.sum()),
        after=len(table),
        reason="detection outside the AOI bounding box (buffered)",
        buffer_deg=cfg["sar"]["detection_bbox_buffer_deg"],
    )
    if table.empty:
        return pd.DataFrame()

    inside = points_within(lon, lat, aoi_geom)
    before_poly = len(table)
    table = table.loc[inside].reset_index(drop=True)
    ledger.record(
        step=f"detection_aoi_polygon_filter:{path.name}",
        before=before_poly,
        after=len(table),
        reason="point falls inside the buffered bbox but outside the AOI polygon",
    )
    if table.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "scene_id": table[scene_col].astype(str),
            "timestamp": parse_gfw_datetimes(table[datetime_col], logger),
            "lat": pd.to_numeric(table[lat_col], errors="coerce"),
            "lon": pd.to_numeric(table[lon_col], errors="coerce"),
        }
    )
    for role in ("presence_score", "length_m", "matching_score", "fishing_score"):
        column = mapping.get(role)
        out[role] = pd.to_numeric(table[column], errors="coerce") if column in table.columns else np.nan
    out["mmsi"] = pd.to_numeric(table[mmsi_col], errors="coerce") if mmsi_col in table.columns else np.nan

    if match_col and match_col in table.columns:
        labels = table[match_col].astype("string").str.strip()
    else:
        labels = pd.Series([pd.NA] * len(table), dtype="string")
    out["match_label"] = labels

    # Two independent match definitions, kept side by side and never conflated.
    if match_col and unmatched_value is not None:
        normalised = labels.str.lower()
        is_unmatched = normalised.eq(str(unmatched_value).lower())
        is_missing = missing_tokens_mask(labels, cfg["sar"]["null_tokens"])
        status = np.where(is_unmatched, "unmatched", np.where(is_missing, "unknown", "matched"))
    else:
        status = np.where(out["mmsi"].notna(), "matched", "unknown") if not out["mmsi"].isna().all() else np.full(len(out), "unknown")
        logger.warning(
            "%s: no explicit unmatched level in the schema mapping; "
            "match_status falls back to the MMSI definition",
            path.name,
        )
    out["match_status"] = pd.Series(status, dtype="string")
    out["matched_flag"] = out["match_status"].eq("matched")
    out["unmatched_flag"] = out["match_status"].eq("unmatched")
    out["unknown_flag"] = out["match_status"].eq("unknown")
    out["mmsi_matched_flag"] = out["mmsi"].notna()

    out["pipeline"] = pipeline
    out["source_file"] = path.name
    out["satellite_platform"] = out["scene_id"].map(parse_scene_platform)
    return out


# ---------------------------------------------------------------------------
# daily AOI coverage
# ---------------------------------------------------------------------------
def build_daily_aoi_coverage(
    scenes: pd.DataFrame,
    cfg: dict[str, Any],
    logger,
    area_km2: float,
) -> pd.DataFrame:
    """Compute, per calendar date, the AOI area actually imaged by SAR.

    Dates without any acquisition are *absent* from the output (never zero).

    ``area_km2`` must be the area of the AOI these scenes belong to. It is
    passed in explicitly so that a non-primary AOI can never be normalised by the
    primary AOI's area.
    """
    if scenes.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_scenes",
                "n_scenes_valid_geometry",
                "aoi_covered_km2",
                "aoi_covered_fraction",
                "max_scene_coverage_fraction",
                "platforms",
                "pipelines",
            ]
        )
    working = scenes.copy()
    working["date"] = pd.to_datetime(working["scene_date_parsed"]).dt.normalize()
    geoms = gpd.GeoSeries(
        shapely.from_wkt(working["intersection_wkt"].fillna("").to_numpy(), on_invalid="ignore"),
        crs="EPSG:4326",
    )
    working = gpd.GeoDataFrame(working, geometry=geoms)
    working = working[working.geometry.notna() & ~working.geometry.is_empty]
    if working.empty:
        return pd.DataFrame()

    equal_area_crs = cfg["aoi"]["equal_area_crs"]

    records: list[dict[str, Any]] = []
    for date_value, group in working.groupby("date", sort=True):
        unioned = group.geometry.union_all() if hasattr(group.geometry, "union_all") else group.geometry.unary_union
        unioned_area = float(
            gpd.GeoSeries([unioned], crs="EPSG:4326").to_crs(equal_area_crs).area.iloc[0] / 1e6
        )
        records.append(
            {
                "date": date_value,
                "n_scenes": int(len(group)),
                "n_scenes_valid_geometry": int(group["geometry_valid"].sum()),
                "aoi_covered_km2": unioned_area,
                "aoi_covered_fraction": unioned_area / area_km2 if area_km2 else np.nan,
                "max_scene_coverage_fraction": float(group["coverage_fraction"].max()),
                "platforms": ",".join(sorted(group["satellite_platform"].dropna().unique())),
                "pipelines": ",".join(sorted(group["pipeline"].dropna().unique())),
            }
        )
    daily = pd.DataFrame(records)
    daily = add_calendar_fields(daily, "date")
    logger.info(
        "daily AOI coverage: %d acquisition dates between %s and %s",
        len(daily),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )
    return daily


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("01_prepare_sar", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 1 - ingest GFW SAR footprints + detections for the AOI")

    mapping = load_schema_mapping(cfg, logger)
    aoi_key = args.aoi or primary_aoi_key(cfg)
    # Window isolation is a single config key: every aoi_scoped_path call picks it
    # up, so a second calendar window for the same AOI cannot overwrite the first.
    if args.window_tag:
        cfg["paths"]["window_tag"] = args.window_tag
    window_tag = cfg["paths"].get("window_tag")
    if args.aoi is not None and args.aoi not in list_aois(cfg):
        logger.error("unknown AOI '%s'; configured AOIs: %s", args.aoi, list_aois(cfg))
        return 2
    aoi_geom = aoi_geometry(cfg, aoi_key)
    area_km2 = aoi_area_km2(cfg, aoi_geom)
    logger.info(
        "AOI '%s' area = %.1f km2 (equal-area %s)",
        aoi_key,
        area_km2,
        cfg["aoi"]["equal_area_crs"],
    )

    load_start = pd.Timestamp(args.start or cfg["analysis"]["load_start"])
    load_end = pd.Timestamp(args.end or cfg["analysis"]["load_end"])
    # When an explicit window is supplied it also bounds the analysis cut-off,
    # otherwise a historical run would keep every later detection.
    analysis_end = pd.Timestamp(args.end) if args.end else pd.Timestamp(cfg["analysis"]["phase1_end"])
    if args.start or args.end:
        logger.info("window override: %s .. %s", load_start.date(), load_end.date())

    selected, discovery_ledger = discover_sar_files(cfg, logger)
    ledger = FilterLedger(name="prepare_sar")
    for step in discovery_ledger.steps:
        ledger.steps.append(step)

    footprint_files = files_in_window(selected, load_start, load_end, kind="footprints")
    detection_files = files_in_window(selected, load_start, load_end, kind="detections")
    if args.limit_months:
        footprint_files = footprint_files.head(args.limit_months)
        detection_files = detection_files.head(args.limit_months)
        logger.warning("debug mode: limited to the first %d months per kind", args.limit_months)

    # ---- footprints ------------------------------------------------------
    scene_frames: list[pd.DataFrame] = []
    for row in footprint_files.itertuples(index=False):
        frame = process_footprint_file(
            Path(row.path), label_or_na(row.pipeline), aoi_geom, cfg, ledger, logger
        )
        if not frame.empty:
            scene_frames.append(frame)
        logger.info("footprints %s -> %d AOI-intersecting scenes", row.file_name, len(frame))
    scenes = pd.concat(scene_frames, ignore_index=True) if scene_frames else pd.DataFrame()
    if scenes.empty:
        logger.error("no SAR scene footprint intersects the AOI - check config/hormuz_core.geojson")
        return 4

    before_dedup = len(scenes)
    scenes = scenes.sort_values(["scene_id", "pipeline"]).drop_duplicates("scene_id", keep="first")
    ledger.record(
        step="scene_dedup",
        before=before_dedup,
        after=len(scenes),
        reason="same scene_id encountered in more than one monthly file",
    )
    scenes = scenes.reset_index(drop=True)
    scenes["coverage_fraction"] = scenes["coverage_km2"] / area_km2
    scenes["aoi_area_km2"] = area_km2
    scenes = add_calendar_fields(scenes, "scene_date_parsed")

    # Coverage distribution over the AOI-intersecting scenes is the evidence used
    # to calibrate ``sar.min_scene_coverage_fraction``.
    quantiles = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    distribution = scenes["coverage_fraction"].quantile(quantiles).rename("coverage_fraction").reset_index()
    distribution.columns = ["quantile", "coverage_fraction"]
    distribution.to_csv(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["diagnostics_dir"]) / "sar_scene_coverage_distribution.csv",
            aoi_key,
            cfg,
            tag=window_tag,
        ),
        index=False,
    )
    logger.info(
        "scene coverage_fraction: n=%d median=%.3f p25=%.3f p75=%.3f max=%.3f",
        len(scenes),
        scenes["coverage_fraction"].median(),
        scenes["coverage_fraction"].quantile(0.25),
        scenes["coverage_fraction"].quantile(0.75),
        scenes["coverage_fraction"].max(),
    )

    # ---- detections ------------------------------------------------------
    detection_frames: list[pd.DataFrame] = []
    for row in detection_files.itertuples(index=False):
        frame = process_detection_file(
            Path(row.path), label_or_na(row.pipeline), aoi_geom, mapping, cfg, ledger, logger
        )
        if not frame.empty:
            detection_frames.append(frame)
        logger.info("detections %s -> %d detections inside AOI", row.file_name, len(frame))
    detections = pd.concat(detection_frames, ignore_index=True) if detection_frames else pd.DataFrame()
    if detections.empty:
        logger.error("no SAR detection falls inside the AOI - check the AOI and the detection files")
        return 4

    before_window = len(detections)
    detections = detections[detections["timestamp"] <= analysis_end].reset_index(drop=True)
    ledger.record(
        step="detection_analysis_window",
        before=before_window,
        after=len(detections),
        reason=f"timestamp after phase-1 analysis end ({analysis_end.date()})",
    )

    before_dup = len(detections)
    detection_key = ["scene_id", "timestamp", "lat", "lon"]
    detections = detections.drop_duplicates(subset=detection_key, keep="first").reset_index(drop=True)
    ledger.record(
        step="detection_dedup",
        before=before_dup,
        after=len(detections),
        reason="identical (scene_id, timestamp, lat, lon) appearing in more than one monthly file",
    )

    # ---- orphan diagnostics ---------------------------------------------
    known_scenes = set(scenes["scene_id"])
    detections["scene_in_footprints"] = detections["scene_id"].isin(known_scenes)
    orphans = detections[~detections["scene_in_footprints"]]
    orphan_summary = (
        orphans.groupby("source_file")
        .agg(n_orphan_detections=("scene_id", "size"), n_distinct_scenes=("scene_id", "nunique"))
        .reset_index()
        if not orphans.empty
        else pd.DataFrame(columns=["source_file", "n_orphan_detections", "n_distinct_scenes"])
    )
    logger.info(
        "detections without a matching footprint scene: %d rows / %d scenes (%.3f%% of AOI detections)",
        len(orphans),
        orphans["scene_id"].nunique(),
        100.0 * len(orphans) / len(detections),
    )

    # ---- daily coverage --------------------------------------------------
    daily = build_daily_aoi_coverage(scenes, cfg, logger, area_km2)

    # ---- write outputs ---------------------------------------------------
    interim = resolve_path(cfg["paths"]["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    scene_path = aoi_scoped_path(interim / "sar_scenes.parquet", aoi_key, cfg, tag=window_tag)
    detection_path = aoi_scoped_path(
        interim / "sar_detections_aoi.parquet", aoi_key, cfg, tag=window_tag
    )
    daily_path = aoi_scoped_path(
        interim / "sar_daily_aoi_coverage.parquet", aoi_key, cfg, tag=window_tag
    )

    scene_columns = [
        "scene_id",
        "scene_date",
        "scene_date_parsed",
        "acquisition_start",
        "acquisition_end",
        "acquisition_start_parsed",
        "acquisition_end_parsed",
        "year",
        "month",
        "iso_year",
        "iso_week",
        "week_key",
        "week_start",
        "pipeline",
        "satellite_platform",
        "geometry_valid",
        "footprint_area_km2",
        "intersection_km2",
        "coverage_km2",
        "coverage_fraction",
        "aoi_area_km2",
        "intersection_wkt",
        "source_file",
    ]
    scenes[scene_columns].to_parquet(scene_path, index=False)
    detections.to_parquet(detection_path, index=False)
    daily.to_parquet(daily_path, index=False)
    logger.info("wrote %s (%d rows)", scene_path, len(scenes))
    logger.info("wrote %s (%d rows)", detection_path, len(detections))
    logger.info("wrote %s (%d rows)", daily_path, len(daily))

    ledger_frame = ledger.to_frame()
    ledger_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "prepare_sar_ledger.csv",
        aoi_key,
        cfg,
        tag=window_tag,
    )
    ledger_frame.to_csv(ledger_path, index=False)
    ledger.write_json(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["diagnostics_dir"]) / "prepare_sar_ledger.json",
            aoi_key,
            cfg,
            tag=window_tag,
        )
    )
    orphan_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "orphan_detections_by_file.csv",
        aoi_key,
        cfg,
        tag=window_tag,
    )
    orphan_summary.to_csv(orphan_path, index=False)
    logger.info("wrote %s", ledger_path)

    ledger.log_summary(logger)
    summary = {
        "aoi_key": aoi_key,
        "window_tag": window_tag,
        "aoi_area_km2": area_km2,
        "n_scenes_intersecting_aoi": int(len(scenes)),
        "n_scenes_above_threshold": int(
            (scenes["coverage_fraction"] >= float(cfg["sar"]["min_scene_coverage_fraction"])).sum()
        ),
        "min_scene_coverage_fraction": float(cfg["sar"]["min_scene_coverage_fraction"]),
        "n_aoi_detections": int(len(detections)),
        "n_orphan_detections": int(len(orphans)),
        "n_acquisition_dates": int(len(daily)),
        "date_min": str(pd.to_datetime(scenes["scene_date_parsed"]).min().date()),
        "date_max": str(pd.to_datetime(scenes["scene_date_parsed"]).max().date()),
    }
    write_json(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["diagnostics_dir"]) / "prepare_sar_summary.json",
            aoi_key,
            cfg,
            tag=window_tag,
        ),
        summary,
    )
    logger.info("STEP 1 summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
