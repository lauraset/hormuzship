"""Shared infrastructure for the Hormuz AIS-vs-SAR observability feasibility test.

This module deliberately contains *no* analysis logic. It only provides:

* configuration loading (``config/config.yaml`` is the single source of truth),
* logging setup,
* AOI loading and equal-area geometry measurement helpers,
* GFW file discovery / de-duplication,
* filter ledgers so that every row-level filter is reported
  (before / after / removed / reason),
* schema-mapping loading produced by ``src/00_inspect_data.py``.

All timestamps are handled as UTC-naive ``datetime64[ns]`` because GFW ships
timestamp strings suffixed with ``UTC``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration.

    Parameters
    ----------
    path:
        Optional path to a config file. Defaults to ``config/config.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration. Environment variables prefixed with
        ``HORMUZ_`` may override ``paths.*`` entries via
        ``HORMUZ_<UPPER_SNAKE_PATH_KEY>`` (e.g. ``HORMUZ_SAR_ARCHIVE_ROOT``).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"config file is not a mapping: {config_path}")
    cfg["_config_path"] = str(config_path)
    cfg.setdefault("paths", {})
    for key in list(cfg["paths"]):
        env_key = f"HORMUZ_{key.upper()}"
        if env_key in os.environ:
            cfg["paths"][key] = os.environ[env_key]
    return cfg


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path against the project root (absolute paths pass through)."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def config_path(cfg: Mapping[str, Any], *keys: str) -> Path:
    """Fetch a nested path from the config and resolve it."""
    node: Any = cfg
    for key in keys:
        node = node[key]
    return resolve_path(node)


def cached_json(cfg: Mapping[str, Any], *keys: str) -> Any:
    """Fetch a nested scalar from the config (no path resolution)."""
    node: Any = cfg
    for key in keys:
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def get_logger(
    name: str,
    cfg: Mapping[str, Any] | None = None,
    level: str | None = None,
) -> logging.Logger:
    """Create (or fetch) a logger writing to stdout and to ``outputs/logs``.

    Repeated calls with the same ``name`` do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_hormuz_configured", False):
        return logger

    resolved_level = (level or (cfg or {}).get("logging", {}).get("level", "INFO")).upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    logger.propagate = False

    # GFW payloads contain non-ASCII (vessel names, flags). Force UTF-8 on the
    # console so a log line can never crash a run on a non-UTF-8 code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if cfg is not None and "logging" in cfg and "log_file" in cfg["logging"]:
        log_file = resolve_path(cfg["logging"]["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger._hormuz_configured = True  # type: ignore[attr-defined]
    return logger


def banner(logger: logging.Logger, title: str) -> None:
    """Log a visually separated section header."""
    logger.info("=" * 78)
    logger.info("%s", title)
    logger.info("=" * 78)


# ---------------------------------------------------------------------------
# filter accounting
# ---------------------------------------------------------------------------
@dataclass
class FilterLedger:
    """Records row counts for every filtering step so nothing vanishes silently.

    Each step stores ``before``, ``after``, ``removed`` and ``reason`` together
    with free-form ``details``. The ledger can be written to JSON or CSV.
    """

    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        step: str,
        before: int,
        after: int,
        reason: str,
        **details: Any,
    ) -> dict[str, Any]:
        """Append a filtering step and return the recorded entry."""
        removed = int(before) - int(after)
        entry = {
            "step": step,
            "before_rows": int(before),
            "after_rows": int(after),
            "removed_rows": removed,
            "removed_share": (removed / before) if before else 0.0,
            "reason": reason,
            "details": details,
        }
        for key, value in details.items():
            entry[key] = value
        self.steps.append(entry)
        return entry

    def to_frame(self) -> pd.DataFrame:
        """Return the ledger as a tidy DataFrame."""
        if not self.steps:
            return pd.DataFrame(
                columns=["ledger", "step", "before_rows", "after_rows", "removed_rows", "reason"]
            )
        records = []
        for step in self.steps:
            record = {k: v for k, v in step.items() if k != "details"}
            record["ledger"] = self.name
            records.append(record)
        return pd.DataFrame(records)

    def write_json(self, path: str | Path) -> Path:
        """Persist the ledger to JSON (UTF-8)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ledger": self.name, "steps": self.steps}
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def log_summary(self, logger: logging.Logger) -> None:
        """Log one line per recorded step."""
        for step in self.steps:
            logger.info(
                "[ledger:%s] %-42s before=%-9d after=%-9d removed=%-9d (%.2f%%) | %s",
                self.name,
                step["step"],
                step["before_rows"],
                step["after_rows"],
                step["removed_rows"],
                100.0 * step["removed_share"],
                step["reason"],
            )


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON payload with UTF-8 encoding, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# AOI helpers
# ---------------------------------------------------------------------------
def list_aois(cfg: Mapping[str, Any]) -> list[str]:
    """Return every configured AOI key, primary first."""
    aoi_cfg = cfg["aoi"]
    definitions = aoi_cfg.get("definitions")
    if not definitions:
        # legacy flat layout: a single AOI under aoi.*
        return [str(aoi_cfg.get("name", "primary_aoi"))]
    keys = list(definitions)
    primary = aoi_cfg.get("primary")
    if primary in keys:
        keys.remove(primary)
        keys.insert(0, str(primary))
    return keys


def primary_aoi_key(cfg: Mapping[str, Any]) -> str:
    """Key of the primary AOI specification."""
    aoi_cfg = cfg["aoi"]
    if aoi_cfg.get("definitions"):
        return str(aoi_cfg.get("primary") or list(aoi_cfg["definitions"])[0])
    return str(aoi_cfg.get("name", "primary_aoi"))


def resolve_aoi(cfg: Mapping[str, Any], aoi_key: str | None = None) -> tuple[str, dict[str, Any]]:
    """Resolve an AOI key to ``(key, definition)``.

    Supports the legacy flat layout (``aoi.geojson`` + ``aoi.primary_feature``)
    as well as the current ``aoi.definitions`` mapping. Raises a clear error for
    an unknown key instead of silently falling back to the primary AOI.
    """
    aoi_cfg = cfg["aoi"]
    if not aoi_cfg.get("definitions"):
        key = aoi_key or primary_aoi_key(cfg)
        return key, {
            "geojson": aoi_cfg["geojson"],
            "primary_feature": aoi_cfg.get("primary_feature"),
            "role": "primary",
        }
    key = aoi_key or primary_aoi_key(cfg)
    definitions = aoi_cfg["definitions"]
    if key not in definitions:
        raise KeyError(
            f"unknown AOI key '{key}'; configured AOIs are {list(definitions)}"
        )
    return key, dict(definitions[key])


def aoi_scoped_path(
    path: str | Path,
    aoi_key: str,
    cfg: Mapping[str, Any],
    tag: str | None = None,
) -> Path:
    """Return the AOI-scoped output path for a file or directory name.

    The primary AOI keeps the plain name (so the documented deliverable names
    such as ``outputs/tables/data_coverage.csv`` are stable); any other AOI gets
    the configured suffix inserted before the extension, e.g.
    ``data/interim/sar_scenes__hormuz_narrow_strait.parquet``.

    ``tag`` appends a further suffix (e.g. ``__w2024_2026``) so that the *same*
    AOI can be prepared for two different calendar windows without either run
    overwriting the other. This exists so the Phase 4A extension can screen a new
    window for an already-screened AOI while leaving the earlier screening
    artifacts untouched.

    When ``tag`` is omitted the value of ``paths.window_tag`` from the config is
    used, so a run can isolate a second calendar window for an AOI by setting that
    one key instead of threading a tag through every call site.
    """
    resolved = resolve_path(path)
    if tag is None:
        tag = cfg.get("paths", {}).get("window_tag")
    if aoi_key == primary_aoi_key(cfg):
        suffix = ""
    else:
        template = cfg["aoi"].get("non_primary_suffix_template", "__{aoi_key}")
        suffix = template.format(aoi_key=aoi_key)
    if tag:
        suffix = f"{suffix}__{tag}"
    if not suffix:
        return resolved
    if resolved.suffix:
        return resolved.with_name(f"{resolved.stem}{suffix}{resolved.suffix}")
    return resolved.with_name(f"{resolved.name}{suffix}")


def load_aoi(cfg: Mapping[str, Any], aoi_key: str | None = None) -> gpd.GeoDataFrame:
    """Load the AOI polygon(s) from the configured GeoJSON.

    Parameters
    ----------
    cfg:
        Parsed configuration.
    aoi_key:
        Which configured AOI to load. Defaults to the primary AOI.

    Raises
    ------
    FileNotFoundError
        If the GeoJSON is missing.
    ValueError
        If no feature carries the configured ``primary_feature`` name, if the
        geometry is empty/invalid, or if the CRS is not the expected one.
    """
    key, aoi_cfg = resolve_aoi(cfg, aoi_key)
    aoi_path = resolve_path(aoi_cfg["geojson"])
    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI GeoJSON for '{key}' not found: {aoi_path}. Coordinates must come from this "
            "file, never from hardcoded Python literals."
        )

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"AOI GeoJSON has no features: {aoi_path}")

    name_field = aoi_cfg.get("name_property", "name")
    primary = aoi_cfg.get("primary_feature")
    if primary is not None and name_field in gdf.columns:
        selected = gdf[gdf[name_field] == primary]
        if selected.empty:
            raise ValueError(
                f"AOI feature '{primary}' not found in {aoi_path} (property '{name_field}'). "
                f"Available: {sorted(gdf[name_field].astype(str).unique())}"
            )
        gdf = selected

    expected_epsg = int(aoi_cfg.get("expected_crs_epsg", 4326))
    if gdf.crs is None:
        raise ValueError(
            f"AOI GeoJSON has no CRS information: {aoi_path}. "
            f"Expected EPSG:{expected_epsg} (lon/lat)."
        )
    if gdf.crs.to_epsg() != expected_epsg:
        gdf = gdf.to_crs(epsg=expected_epsg)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        repaired = gdf.geometry[invalid].apply(shapely.make_valid)
        gdf.loc[invalid, "geometry"] = repaired
    if gdf.geometry.is_empty.any():
        raise ValueError(f"AOI contains an empty geometry: {aoi_path}")
    return gdf


def aoi_geometry(cfg: Mapping[str, Any], aoi_key: str | None = None) -> BaseGeometry:
    """Return the AOI as a single (possibly multi-part) geometry in EPSG:4326."""
    gdf = load_aoi(cfg, aoi_key)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    return geom


def aoi_area_km2(
    cfg: Mapping[str, Any],
    aoi_geom: BaseGeometry | None = None,
    aoi_key: str | None = None,
) -> float:
    """Area of the AOI in km2, computed in the configured equal-area CRS."""
    geom = aoi_geom if aoi_geom is not None else aoi_geometry(cfg, aoi_key)
    equal_area_crs = cfg["aoi"]["equal_area_crs"]
    series = gpd.GeoSeries([geom], crs=f"EPSG:{cfg['aoi']['expected_crs_epsg']}").to_crs(equal_area_crs)
    return float(series.area.iloc[0] / 1e6)


def measure_intersections(
    polygons_4326: gpd.GeoSeries,
    aoi_geom: BaseGeometry,
    equal_area_crs: str,
    source_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """Measure polygon-vs-AOI intersection areas in an equal-area CRS.

    Parameters
    ----------
    polygons_4326:
        Scene footprint geometries (any CRS given by ``source_crs``).
    aoi_geom:
        AOI geometry in the same CRS as ``polygons_4326``.
    equal_area_crs:
        Target CRS for area computation (e.g. ``EPSG:6933``).
    source_crs:
        CRS of the supplied geometries.

    Returns
    -------
    pandas.DataFrame
        Columns: ``geometry_valid``, ``footprint_area_km2``, ``intersection_km2``,
        ``coverage_fraction``, ``intersection_wkt`` (WKT in the source CRS).
        Rows are positionally aligned with ``polygons_4326``.
    """
    geoms = gpd.GeoSeries(polygons_4326, crs=source_crs)

    not_null = geoms.notna() & ~geoms.is_empty
    valid = np.zeros(len(geoms), dtype=bool)
    repaired = geoms.copy()
    for position, geom in enumerate(geoms):
        if not not_null.iloc[position]:
            continue
        candidate = geom
        if not shapely.is_valid(geom):
            candidate = shapely.make_valid(geom)
        # A "repaired" geometry may degenerate into lines/points; only polygonal
        # parts can carry an area and therefore a coverage fraction.
        if candidate.is_empty or candidate.area <= 0:
            continue
        repaired.iloc[position] = candidate
        valid[position] = True

    aoi_series = gpd.GeoSeries([aoi_geom], crs=source_crs)
    aoi_area = float(aoi_series.to_crs(equal_area_crs).area.iloc[0])

    usable = repaired.where(valid)
    intersections = usable.intersection(aoi_geom)
    intersections = intersections.where(valid)

    footprint_area = (
        gpd.GeoSeries(usable, crs=source_crs).to_crs(equal_area_crs).area.to_numpy() / 1e6
    )
    inter_area = (
        gpd.GeoSeries(intersections, crs=source_crs).to_crs(equal_area_crs).area.to_numpy() / 1e6
    )
    inter_area = np.where(valid, inter_area, np.nan)
    fraction = inter_area / (aoi_area / 1e6)

    wkt = np.array([g.wkt if (valid[i] and g is not None and not g.is_empty) else None
                    for i, g in enumerate(intersections)], dtype=object)

    return pd.DataFrame(
        {
            "geometry_valid": valid,
            "footprint_area_km2": footprint_area,
            "intersection_km2": inter_area,
            "coverage_fraction": fraction,
            "intersection_wkt": wkt,
        }
    )


def bbox_of(geom: BaseGeometry, buffer_deg: float = 0.0) -> tuple[float, float, float, float]:
    """Return ``(min_lon, min_lat, max_lon, max_lat)`` optionally buffered in degrees."""
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    return (
        min_lon - buffer_deg,
        min_lat - buffer_deg,
        max_lon + buffer_deg,
        max_lat + buffer_deg,
    )


def points_within(
    lon: np.ndarray | pd.Series,
    lat: np.ndarray | pd.Series,
    geom: BaseGeometry,
) -> np.ndarray:
    """Vectorised point-in-polygon test for lon/lat arrays."""
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    finite = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    out = np.zeros(len(lon_arr), dtype=bool)
    if finite.any():
        pts = shapely.points(lon_arr[finite], lat_arr[finite])
        out[finite] = shapely.intersects(pts, geom)
    return out


# ---------------------------------------------------------------------------
# GFW time handling
# ---------------------------------------------------------------------------
GFW_TIMESTAMP_FORMATS: tuple[str, ...] = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def parse_gfw_datetimes(values: pd.Series, logger: logging.Logger | None = None) -> pd.Series:
    """Parse GFW timestamp strings such as ``2025-01-01 04:59:28 UTC`` to UTC-naive.

    Missing values stay ``NaT`` (never silently dropped or filled).
    """
    text = values.astype("string").str.strip()
    text = text.str.replace(" UTC", "", regex=False).str.replace("Z", "", regex=False)
    parsed = pd.to_datetime(text, format=GFW_TIMESTAMP_FORMATS[0], errors="coerce")
    if parsed.isna().any():
        remaining = parsed.isna() & text.notna()
        if remaining.any():
            fallback = pd.to_datetime(text[remaining], format="mixed", errors="coerce")
            parsed.loc[remaining] = fallback
        if logger is not None and parsed.isna().any():
            logger.warning(
                "parse_gfw_datetimes: %d timestamps could not be parsed (kept as NaT)",
                int(parsed.isna().sum()),
            )
    return parsed


def parse_scene_platform(scene_id: str) -> str:
    """Extract the satellite platform token from a GFW scene id.

    Example
    -------
    >>> parse_scene_platform("S1A_IW_GRDH_1SDV_20250101T045915_..._0B4A")
    'S1A'
    """
    if not isinstance(scene_id, str) or not scene_id:
        return "unknown"
    token = scene_id.split("_", 1)[0]
    return token if token else "unknown"


def scene_acquisition_time(scene_id: str) -> pd.Timestamp | pd.NaT:
    """Extract the acquisition timestamp embedded in a GFW scene id."""
    match = re.search(r"_(\d{8})T(\d{6})_", scene_id or "")
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S", errors="coerce")


# ---------------------------------------------------------------------------
# Windows and weeks
# ---------------------------------------------------------------------------
def resolve_md_window(
    year: int,
    start_md: str,
    end_md: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a month-day window (``MM-DD``) to timestamps for a given year."""
    start = pd.Timestamp(f"{year}-{start_md}")
    end = pd.Timestamp(f"{year}-{end_md}")
    if end < start:
        raise ValueError(f"window end {end_md} precedes start {start_md} for year {year}")
    return start, end


def resolve_phases(cfg: Mapping[str, Any], year: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Resolve every configured phase window to concrete dates for ``year``."""
    return {
        name: resolve_md_window(year, spec["start_md"], spec["end_md"])
        for name, spec in cfg["analysis"]["phases"].items()
    }


def resolve_pre_post(cfg: Mapping[str, Any], year: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Resolve the pre/post calendar split (used for placebo years too)."""
    return {
        name: resolve_md_window(year, spec["start_md"], spec["end_md"])
        for name, spec in cfg["analysis"]["pre_post"].items()
    }


def in_window(dates: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    """Boolean mask for dates inside an inclusive ``(start, end)`` window."""
    start, end = window
    return (dates >= start) & (dates <= end)


def add_calendar_fields(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add year / month / iso_week / week_start columns derived from ``date_col``."""
    out = frame.copy()
    dates = pd.to_datetime(out[date_col])
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    iso = dates.dt.isocalendar()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    out["week_key"] = out["iso_year"].astype(str) + "-W" + out["iso_week"].astype(str).str.zfill(2)
    # ISO weeks start on Monday; to_period("W-SUN").start_time is that Monday.
    out["week_start"] = dates.dt.to_period("W-SUN").dt.start_time
    return out


# ---------------------------------------------------------------------------
# GFW file discovery
# ---------------------------------------------------------------------------
def discover_sar_files(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
    kind: str | None = None,
) -> tuple[pd.DataFrame, FilterLedger]:
    """Inventory GFW SAR detections / footprints and de-duplicate re-downloads.

    Search order (first hit wins for a given ``(kind, month)``):

    1. local staging directories (``data/raw/sar_detections``, ``.../footprints``)
    2. the external read-only archive root (e.g. ``D:/gfwdata``), including
       per-year subdirectories

    De-duplication rules (all recorded in the returned ledger):

    * a full-month file beats a partial ``..._YYYYMMDD.csv`` file;
    * if two pipeline versions ship the same month, the one earliest in
      ``sar.pipeline_preference`` wins.

    Returns
    -------
    (pandas.DataFrame, FilterLedger)
        The selected files (one row per kind+month) and the removal ledger.
    """
    sar_cfg = cfg["sar"]
    detection_re = re.compile(sar_cfg["detection_filename_pattern"])
    footprint_re = re.compile(sar_cfg["footprint_filename_pattern"])

    roots = {
        "detections": [resolve_path(cfg["paths"]["sar_detections_dir"]),
                       resolve_path(cfg["paths"]["sar_archive_root"])],
        "footprints": [resolve_path(cfg["paths"]["sar_footprints_dir"]),
                       resolve_path(cfg["paths"]["sar_archive_root"])],
    }
    patterns = {"detections": detection_re, "footprints": footprint_re}
    kinds = [kind] if kind else ["detections", "footprints"]

    candidates: list[dict[str, Any]] = []
    for current_kind in kinds:
        seen_paths: set[Path] = set()
        for root_rank, root in enumerate(roots[current_kind]):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.csv")):
                if path in seen_paths:
                    continue
                match = patterns[current_kind].match(path.name)
                if not match:
                    continue
                seen_paths.add(path)
                groups = match.groupdict()
                year = int(groups["year"])
                month = int(groups["month"])
                day = int(groups["day"]) if groups.get("day") else None
                stat = path.stat()
                candidates.append(
                    {
                        "kind": current_kind,
                        "path": str(path),
                        "file_name": path.name,
                        "source_rank": root_rank,
                        "year": year,
                        "month": month,
                        "day": day,
                        "yyyymm": f"{year}{month:02d}",
                        "pipeline": groups.get("pipeline"),
                        "is_partial_month": day is not None,
                        "size_bytes": stat.st_size,
                        "mtime": pd.Timestamp(stat.st_mtime, unit="s"),
                    }
                )

    inventory = pd.DataFrame(candidates)
    ledger = FilterLedger(name="sar_file_discovery")
    if inventory.empty:
        if logger:
            logger.error("no GFW SAR files matched the configured patterns")
        return inventory, ledger

    # local copy (source_rank 0) wins over archive copy for the identical file name
    inventory = (
        inventory.sort_values(["kind", "yyyymm", "source_rank", "is_partial_month"])
        .drop_duplicates(subset=["kind", "yyyymm", "day"], keep="first")
        .reset_index(drop=True)
    )

    selected_rows: list[pd.Series] = []
    preference = [str(p) for p in sar_cfg.get("pipeline_preference", [])]

    def pipeline_rank(value: Any) -> int:
        text = "" if value is None else str(value)
        return preference.index(text) if text in preference else len(preference)

    for (current_kind, yyyymm), group in inventory.groupby(["kind", "yyyymm"], sort=True):
        before = len(group)
        group = group.copy()
        group["pipeline_rank"] = group["pipeline"].apply(pipeline_rank)
        group = group.sort_values(["pipeline_rank", "is_partial_month", "mtime"])

        if sar_cfg.get("prefer_full_month_over_partial", True) and (~group["is_partial_month"]).any():
            full = group[~group["is_partial_month"]]
            dropped_partial = group[group["is_partial_month"]]
            for _, row in dropped_partial.iterrows():
                ledger.record(
                    step=f"drop_partial_month:{current_kind}:{yyyymm}",
                    before=before,
                    after=before - 1,
                    reason="full-month file already available for this month",
                    dropped_file=row["file_name"],
                )
            group = full

        chosen = group.iloc[0]
        duplicates = group.iloc[1:]
        for _, row in duplicates.iterrows():
            ledger.record(
                step=f"drop_duplicate_pipeline:{current_kind}:{yyyymm}",
                before=before,
                after=before - 1,
                reason="same month available from a preferred pipeline version",
                dropped_file=row["file_name"],
                kept_file=chosen["file_name"],
            )
            if logger:
                logger.warning(
                    "duplicate %s files for %s: keeping %s (pipeline %s), dropping %s (pipeline %s)",
                    current_kind, yyyymm, chosen["file_name"], chosen["pipeline"],
                    row["file_name"], row["pipeline"],
                )
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected = selected.sort_values(["kind", "year", "month"]).reset_index(drop=True)
    if logger:
        for current_kind in kinds:
            subset = selected[selected["kind"] == current_kind]
            logger.info(
                "discovered %d %s files (%s .. %s)",
                len(subset),
                current_kind,
                subset["yyyymm"].min() if len(subset) else "-",
                subset["yyyymm"].max() if len(subset) else "-",
            )
    return selected, ledger


def files_in_window(
    selected: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    kind: str | None = None,
) -> pd.DataFrame:
    """Subset a discovery table to files whose month overlaps ``[start, end]``."""
    frame = selected.copy()
    if kind is not None:
        frame = frame[frame["kind"] == kind]
    month_start = pd.to_datetime(frame["yyyymm"] + "01", format="%Y%m%d")
    month_end = month_start + pd.offsets.MonthEnd(1)
    keep = (month_start <= end) & (month_end >= start)
    return frame[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# schema mapping
# ---------------------------------------------------------------------------
def load_schema_mapping(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load the verified schema mapping produced by ``src/00_inspect_data.py``.

    Raises
    ------
    FileNotFoundError
        If the mapping has not been generated yet, or if an unresolved
        *required* role exists. Later scripts must never guess column names.
    """
    path = resolve_path(cfg["schema"]["mapping_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"schema mapping missing: {path}. Run `python src/00_inspect_data.py` first and "
            "confirm the detected mapping before running downstream analysis."
        )
    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"schema mapping is not a mapping: {path}")

    required = cfg["schema"].get("required_roles", [])
    missing = [role for role in required if not mapping.get(role)]
    if missing:
        raise ValueError(
            "schema mapping does not resolve required roles "
            f"{missing}. See {path} and {resolve_path(cfg['schema']['report_md'])}. "
            "Downstream analysis is intentionally blocked until the mapping is confirmed."
        )
    if logger is not None:
        logger.info("schema mapping loaded from %s", path)
        logger.info(
            "  scene_id=%s datetime=%s lat=%s lon=%s match_label=%s footprint_wkt=%s",
            mapping.get("scene_id"),
            mapping.get("datetime"),
            mapping.get("latitude"),
            mapping.get("longitude"),
            mapping.get("match_label"),
            mapping.get("footprint_wkt"),
        )
    return mapping


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def ensure_dirs(cfg: Mapping[str, Any]) -> list[Path]:
    """Create every configured output directory and return the created paths."""
    keys = [
        "interim_dir",
        "processed_dir",
        "figures_dir",
        "tables_dir",
        "diagnostics_dir",
        "logs_dir",
    ]
    created = []
    for key in keys:
        path = resolve_path(cfg["paths"][key])
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def nullable_int(value: Any) -> Any:
    """Cast a value to ``int`` unless it is NaN/None, in which case return ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return int(value)


def label_or_na(value: Any) -> str:
    """Return a clean string label, mapping missing values to ``"n/a"``.

    Used for provenance columns (pipeline version, platform) so that the literal
    string ``"None"`` can never leak into an output table.
    """
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "nan", "null"} else "n/a"


def read_env(env_path: str | Path | None = None, logger: logging.Logger | None = None) -> dict[str, str]:
    """Read ``.env`` into a mapping without mutating ``os.environ``.

    ``python-dotenv`` is used when available; otherwise a minimal parser handles
    ``KEY=VALUE`` lines so the pipeline still runs. Secrets are never logged.
    """
    path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        try:
            from dotenv import dotenv_values  # type: ignore

            values = {k: v for k, v in dotenv_values(path).items() if v is not None}
            if logger:
                logger.info("loaded %d keys from %s (python-dotenv)", len(values), path.name)
        except ImportError:
            if logger:
                logger.warning(
                    "python-dotenv not installed; using the built-in minimal .env parser "
                    "(install requirements.txt to switch back)"
                )
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    elif logger:
        logger.info("no .env file found at %s", path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> Any:
    """Element-wise division returning NaN where the denominator is 0 or missing."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((den > 0), num / np.where(den == 0, np.nan, den), np.nan)
    if np.ndim(out) == 0:
        return float(out)
    return out


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{context}: missing column(s) {missing}; available columns: {list(frame.columns)}"
        )


def read_csv_fast(
    path: str | Path,
    columns: Sequence[str] | None = None,
    block_size: int = 1 << 25,
    expected_columns: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Read a CSV (optionally a column subset) with pyarrow and return pandas.

    One reader is used for every GFW file so that missing-value semantics are
    identical across pipeline versions: pyarrow only treats ``""`` as null,
    whereas pandas would additionally coerce ``NA``/``nan`` strings. Callers
    that must distinguish "unknown" from a real category therefore normalise
    ``sar.null_tokens`` explicitly.

    Parameters
    ----------
    columns:
        Columns to keep (a subset is fine).
    expected_columns:
        The documented GFW column order for this file type. When provided and the
        file turns out to have **lost its header row** (observed once in the
        archive: ``sar_detections_footprints_201907.csv``), the file is read
        positionally with these names instead of failing. The fallback is always
        logged so the anomaly is visible rather than silent, and it is only used
        when the first line does not start with the expected first column name.
    """
    import pyarrow.csv as pa_csv

    path = Path(path)
    if expected_columns:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline()
        first_token = first_line.split(",", 1)[0].strip()
        if first_token != str(expected_columns[0]).strip():
            if logger is not None:
                logger.warning(
                    "%s has no header row (first token %r); reading %d columns "
                    "positionally with the documented GFW order",
                    path.name,
                    first_token[:40],
                    len(expected_columns),
                )
            convert_options = (
                pa_csv.ConvertOptions(include_columns=list(columns)) if columns else None
            )
            table = pa_csv.read_csv(
                str(path),
                read_options=pa_csv.ReadOptions(
                    column_names=list(expected_columns), block_size=block_size
                ),
                convert_options=convert_options,
            )
            frame = table.to_pandas()
            frame.attrs["header_missing"] = True
            return frame

    convert_options = pa_csv.ConvertOptions(include_columns=list(columns)) if columns else None
    table = pa_csv.read_csv(
        str(path),
        read_options=pa_csv.ReadOptions(block_size=block_size),
        convert_options=convert_options,
    )
    return table.to_pandas()


def missing_tokens_mask(labels: pd.Series, null_tokens: Sequence[str]) -> pd.Series:
    """Boolean mask of values that GFW uses to mean "no value"."""
    normalised = labels.astype("string").str.strip().str.lower()
    tokens = {str(token).strip().lower() for token in null_tokens}
    return labels.isna() | normalised.isin(tokens)


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "FilterLedger",
    "add_calendar_fields",
    "aoi_area_km2",
    "aoi_geometry",
    "aoi_scoped_path",
    "banner",
    "bbox_of",
    "cached_json",
    "config_path",
    "discover_sar_files",
    "ensure_dirs",
    "files_in_window",
    "get_logger",
    "in_window",
    "label_or_na",
    "list_aois",
    "load_aoi",
    "load_config",
    "load_schema_mapping",
    "measure_intersections",
    "missing_tokens_mask",
    "nullable_int",
    "parse_gfw_datetimes",
    "parse_scene_platform",
    "points_within",
    "primary_aoi_key",
    "read_csv_fast",
    "read_env",
    "require_columns",
    "resolve_aoi",
    "resolve_md_window",
    "resolve_path",
    "resolve_phases",
    "resolve_pre_post",
    "safe_divide",
    "scene_acquisition_time",
    "write_json",
]
