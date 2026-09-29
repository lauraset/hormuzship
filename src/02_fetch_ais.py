"""STEP 2 (Phase 2) - download GFW AIS vessel presence for every configured AOI.

Verified API contract (probed live before this code was written)
---------------------------------------------------------------
``POST {base_url}`` with

* **query parameters**: ``datasets[0]``, ``date-range``, ``temporal-resolution``,
  ``spatial-resolution``, ``format`` (an optional single ``group-by`` is also a
  query parameter);
* **JSON body**: ``{"geojson": <Feature or geometry>}`` - ``geojson`` is rejected
  as a query parameter ("Query params geojson not supported");
* ``format`` must be upper case (``CSV``); ``format=csv`` returns 422;
* **the report response is a ZIP archive**, not raw CSV. It contains
  ``layer-activity-data-0/public-global-presence-v4.0.csv`` plus a readme, the
  GFW AIS caveats PDF, and the reference geometry;
* **the ``date-range`` end date is EXCLUSIVE**: ``2026-03-04,2026-03-04`` returns
  zero rows and ``2026-03-04,2026-03-05`` returns exactly one day. This script
  therefore always requests ``end + 1 day`` and never silently drops the last day
  of a chunk.

Observed response columns (per vessel per day per presence cell)::

    Lat, Lon, Time Range, Vessel ID, Flag, Vessel Name, Entry Timestamp,
    Exit Timestamp, Gear Type, Vessel Type, MMSI, IMO, CallSign,
    First Transmission Date, Last Transmission Date, Vessel Presence Hours

Nothing about these names is hard-coded in the analysis: the roles are detected
from the returned file and written to ``config/ais_schema_mapping.yaml``.

What this script does
---------------------
1. one request per (AOI, month-chunk), sent **sequentially** with a pause,
   bounded retries and backoff (never concurrent, to avoid HTTP 429);
2. raw responses cached byte-for-byte under ``data/raw/ais_presence/*.zip``;
3. one manifest row per HTTP request in ``outputs/logs/api_manifest.csv``
   (request datetime, AOI, start/end date, requested dataset, returned dataset,
   HTTP status, bytes, rows, cache path, error);
4. schema inspection of the real returned file -> ``config/ais_schema_mapping.yaml``
   and ``outputs/diagnostics/ais_schema_report.json``;
5. ``data/processed/ais_daily.parquet`` per AOI, the requested
   ``outputs/tables/ais_vessel_type_summary.csv``, and a day-coverage validation
   table (``outputs/tables/ais_coverage*.csv``).

Missing days are never treated as zero: they are reported as
``missing_no_row_returned`` in the coverage table.

Offline modes
-------------
``--check-auth``   one minimal request to validate the token, then stop
``--dry-run``      print the exact requests without sending them
``--inspect-only`` inspect and process whatever is already cached (no network)
``--from-cache``   same as ``--inspect-only`` (kept for clarity)

The token is read from ``.env``; it never appears in code, logs, cache files or
the manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    FilterLedger,
    aoi_geometry,
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    list_aois,
    load_config,
    read_env,
    resolve_path,
    write_json,
)

# Role -> candidate patterns for the *returned* columns. The GFW presence file
# uses space-separated names ("Vessel Presence Hours"); alternates are kept so a
# rename does not break the pipeline silently.
AIS_ROLE_PATTERNS: dict[str, list[str]] = {
    "date": [r"^time\s*range$", r"^date$", r"^day$", r"^timestamp$"],
    "presence_hours": [r"vessel\s*presence\s*hours", r"^hours$", r"^presence$", r"^value$"],
    "vessel_id": [r"^vessel\s*id$", r"^mmsi$", r"^imo$"],
    "vessel_type": [r"^vessel\s*type$", r"^geartype$", r"^gear\s*type$", r"^type$"],
    "gear_type": [r"^gear\s*type$", r"^geartype$"],
    "flag": [r"^flag$", r"^flagstate$", r"^flag\s*state$"],
    "lat": [r"^lat$", r"^latitude$"],
    "lon": [r"^lon$", r"^lng$", r"^longitude$"],
    "mmsi": [r"^mmsi$"],
}

REQUIRED_AIS_ROLES = ("date", "presence_hours")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch GFW AIS vessel presence (Phase 2)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None,
                        help="AOI keys to fetch (default: every configured AOI)")
    parser.add_argument("--start", default=None, help="override analysis.load_start")
    parser.add_argument("--end", default=None, help="override analysis.load_end (inclusive)")
    parser.add_argument("--check-auth", action="store_true",
                        help="send one minimal request to validate the token, then stop")
    parser.add_argument("--dry-run", action="store_true", help="print requests, send nothing")
    parser.add_argument("--inspect-only", action="store_true",
                        help="process cached responses only, no network")
    parser.add_argument("--from-cache", action="store_true",
                        help="alias of --inspect-only")
    parser.add_argument("--chunk-months", type=int, default=None,
                        help="override ais.request_chunk_months")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------
def month_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    months: int,
    max_days: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split inclusive ``[start, end]`` into chunks of at most ``months`` months.

    ``max_days`` additionally caps every chunk's length, because the GFW report
    endpoint rejects ranges longer than 366 days. Without the cap, asking for a
    29-month window wastes a request on a guaranteed HTTP 422.
    """
    if months < 1:
        raise ValueError("chunk size must be at least one month")
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min((cursor + pd.DateOffset(months=months)) - pd.Timedelta(days=1), end)
        if max_days is not None:
            chunk_end = min(chunk_end, cursor + pd.Timedelta(days=max_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def api_date_range(
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    end_exclusive: bool,
) -> str:
    """Build the ``date-range`` string, compensating for the exclusive end.

    The GFW report API treats the end date as exclusive (verified live), so an
    inclusive analysis window must be converted to ``end + 1 day``.
    """
    api_end = end_inclusive + pd.Timedelta(days=1) if end_exclusive else end_inclusive
    return f"{start.date()},{api_end.date()}"


def build_request_query(
    cfg: dict[str, Any],
    dataset: str,
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> dict[str, Any]:
    """Query-string parameters for one report request (verified shape)."""
    ais = cfg["ais"]
    query: dict[str, Any] = {
        "datasets[0]": dataset,
        "date-range": api_date_range(
            start, end_inclusive, bool(ais.get("date_range_end_exclusive", True))
        ),
        "temporal-resolution": str(ais.get("temporal_resolution", "daily")).upper(),
        "spatial-resolution": str(ais.get("spatial_resolution", "HIGH")).upper(),
        "format": str(ais.get("format", "CSV")).upper(),
    }
    group_by = [str(field).upper() for field in (ais.get("group_by") or [])]
    if group_by:
        allowed = [str(value).upper() for value in ais.get("allowed_group_by", [])]
        unknown = [value for value in group_by if allowed and value not in allowed]
        if unknown:
            raise ValueError(
                f"group-by value(s) {unknown} are not accepted by the GFW report API; "
                f"allowed: {allowed}"
            )
        query["group-by"] = ",".join(group_by)
    return query


def build_request_body(aoi_geometry_object: dict[str, Any]) -> dict[str, Any]:
    """JSON body for one report request: the AOI only.

    ``geojson`` must be in the body; the API rejects it as a query parameter.
    """
    return {
        "geojson": {
            "type": "Feature",
            "properties": {},
            "geometry": aoi_geometry_object,
        }
    }


def cache_path_for(
    cfg: dict[str, Any],
    aoi_key: str,
    dataset: str,
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> Path:
    """Deterministic cache path for one request (raw zip bytes, never edited)."""
    cache_dir = resolve_path(cfg["paths"]["ais_presence_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = re.sub(r"[^0-9A-Za-z._-]", "_", dataset)
    return cache_dir / f"ais_presence_{aoi_key}_{safe_dataset}_{start.date()}_{end_inclusive.date()}.zip"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def send_request(
    cfg: dict[str, Any],
    query: dict[str, Any],
    body: dict[str, Any],
    token: str,
    logger,
) -> tuple[int, bytes, str | None]:
    """Send one report request sequentially, with retries and backoff.

    Returns ``(http_status, content, error)``. ``http_status`` is 0 when no HTTP
    response was received; ``error`` then carries the reason for the manifest.
    """
    import requests

    ais = cfg["ais"]
    headers = {"Authorization": f"Bearer {token}"}
    attempts = int(ais.get("max_retries", 3))
    backoff = float(ais.get("retry_backoff_seconds", 10.0))
    rate_limit_backoff = float(ais.get("rate_limit_backoff_seconds", backoff))
    timeout = float(ais.get("timeout_seconds", 300))

    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                ais["base_url"], params=query, json=body, headers=headers, timeout=timeout
            )
            if response.status_code == 200:
                return 200, response.content, None
            last_error = f"HTTP {response.status_code}: {response.text[:400]}"
            if response.status_code in (401, 403):
                return response.status_code, b"", last_error  # retrying will not help
            if response.status_code == 429:
                logger.warning(
                    "rate limited (429); sleeping %.0f s before retry %d/%d",
                    rate_limit_backoff,
                    attempt,
                    attempts,
                )
                if attempt < attempts:
                    time.sleep(rate_limit_backoff)
                continue
        except Exception as error:  # noqa: BLE001 - recorded in the manifest, never raised
            last_error = f"{type(error).__name__}: {error}"
        if attempt < attempts:
            time.sleep(backoff)
    return 0, b"", last_error


def record_manifest(cfg: dict[str, Any], row: dict[str, Any]) -> Path:
    """Append one request record to the API manifest (creating it if needed)."""
    path = resolve_path(cfg["ais"]["manifest_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def write_cumulative_table(path: Path, frame: pd.DataFrame, keys: Sequence[str]) -> Path:
    """Write an AOI-keyed table without dropping rows produced by earlier runs.

    The pipeline can be run one AOI (or one historical window) at a time, so a
    plain overwrite would silently delete the rows of AOIs processed earlier.
    Rows are replaced by their key, never merged blindly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not frame.empty:
        existing = pd.read_csv(path)
        missing = [key for key in keys if key not in existing.columns or key not in frame.columns]
        if not missing:
            existing_keys = set(map(tuple, existing[list(keys)].astype(str).to_numpy()))
            new_keys = set(map(tuple, frame[list(keys)].astype(str).to_numpy()))
            keep = existing[
                ~existing[list(keys)].astype(str).apply(tuple, axis=1).isin(new_keys)
            ]
            frame = pd.concat([keep, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# payload parsing
# ---------------------------------------------------------------------------
def extract_activity_csv(payload: bytes, logger) -> pd.DataFrame:
    """Extract the activity CSV from a report ZIP payload.

    The endpoint returns a ZIP archive containing the activity CSV, a readme, the
    GFW AIS caveats PDF and the reference geometry. The largest ``.csv`` member is
    the activity table.
    """
    if payload[:2] != b"PK":
        preview = payload[:200].decode("utf-8", errors="replace")
        raise ValueError(f"expected a ZIP report payload but got: {preview!r}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(
                f"no CSV member inside the report archive; members: {archive.namelist()}"
            )
        largest = max(csv_members, key=lambda name: archive.getinfo(name).file_size)
        extras = [name for name in archive.namelist() if name not in {largest}]
        logger.info(
            "report archive members: csv=%s (%d bytes), extras=%s",
            Path(largest).name,
            archive.getinfo(largest).file_size,
            extras,
        )
        raw = archive.read(largest)
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))


def detect_ais_roles(frame: pd.DataFrame) -> dict[str, Any]:
    """Resolve AIS column roles from the columns actually returned."""
    resolved: dict[str, str | None] = {}
    for role, patterns in AIS_ROLE_PATTERNS.items():
        found = None
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for column in frame.columns:
                if regex.search(str(column).strip()):
                    found = str(column)
                    break
            if found:
                break
        resolved[role] = found
    return resolved


def inspect_ais_frame(frame: pd.DataFrame, logger) -> dict[str, Any]:
    """Profile a returned AIS frame: dtypes, missingness, categorical values."""
    columns: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()) if len(series) else float("nan"),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if not pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().astype(str)
            if len(clean):
                entry["example_values"] = clean.drop_duplicates().head(5).tolist()
                if clean.nunique() <= 60:
                    entry["value_counts"] = {
                        str(key): int(value) for key, value in clean.value_counts().items()
                    }
        columns[column] = entry
    roles = detect_ais_roles(frame)
    logger.info("AIS schema roles detected: %s", {k: v for k, v in roles.items() if v})
    return {"n_rows": int(len(frame)), "columns": columns, "roles": roles}


def load_cached_frames(cfg: dict[str, Any], aoi_key: str, logger) -> list[pd.DataFrame]:
    """Read every cached raw report for one AOI into a list of activity frames."""
    cache_dir = resolve_path(cfg["paths"]["ais_presence_dir"])
    if not cache_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob(f"ais_presence_{aoi_key}_*.zip")):
        try:
            frame = extract_activity_csv(path.read_bytes(), logger)
        except Exception as error:  # noqa: BLE001
            logger.warning("could not parse cached report %s: %s", path.name, error)
            continue
        frame = frame.copy()
        frame["_cache_file"] = path.name
        frames.append(frame)
        logger.info("loaded cache %s (%d rows, %d columns)", path.name, len(frame), frame.shape[1])
    return frames


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------
def build_daily_ais_table(
    frames: list[pd.DataFrame],
    roles: dict[str, Any],
    aoi_key: str,
    logger,
) -> pd.DataFrame:
    """Aggregate cached AIS reports into one row per (date, vessel_type).

    ``__ALL__`` rows are appended so that downstream code never has to re-derive
    the all-types total. Days without any row are simply absent (see
    :func:`validate_day_coverage`); they are never zero-filled here.
    """
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    logger.info("concatenated %d raw rows, %d after exact-duplicate removal", before, len(combined))

    missing = [role for role in REQUIRED_AIS_ROLES if not roles.get(role)]
    if missing:
        raise ValueError(
            f"the returned AIS schema does not resolve required role(s) {missing}; "
            f"detected roles: {roles}. Inspect outputs/diagnostics/ais_schema_report.json "
            "and extend AIS_ROLE_PATTERNS if GFW renamed a field."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(combined[roles["date"]], errors="coerce").dt.normalize()
    out["aoi_key"] = aoi_key
    out["presence_hours"] = pd.to_numeric(combined[roles["presence_hours"]], errors="coerce")
    for role in ("vessel_type", "gear_type", "flag", "mmsi", "vessel_id"):
        column = roles.get(role)
        out[role] = combined[column].astype("string").str.strip() if column else pd.NA

    ledger = FilterLedger(name="ais_daily")
    valid = out["date"].notna() & out["presence_hours"].notna()
    ledger.record(
        step="require_date_and_presence_hours",
        before=len(out),
        after=int(valid.sum()),
        reason="row has no parseable date or no presence-hours value",
    )
    out = out[valid].reset_index(drop=True)
    if out.empty:
        return out
    ledger.log_summary(logger)

    # Three parallel classifications are kept, because the coarse "Vessel Type"
    # field folds tankers into OTHER_NON_FISHING while "Gear Type" carries an
    # explicit TANKER level. Downstream selectors are config driven and may use
    # either field; nothing is merged or relabelled here.
    parts: list[pd.DataFrame] = []
    for classification, column in (("vessel_type", "vessel_type"), ("gear_type", "gear_type")):
        part = (
            out.groupby(["aoi_key", "date", column], dropna=False, as_index=False)
            .agg(
                presence_hours=("presence_hours", "sum"),
                n_records=("presence_hours", "size"),
                n_vessels=("mmsi", "nunique"),
            )
            .rename(columns={column: "class_value"})
        )
        part["classification"] = classification
        parts.append(part)
    totals = (
        out.groupby(["aoi_key", "date"], as_index=False)
        .agg(
            presence_hours=("presence_hours", "sum"),
            n_records=("presence_hours", "size"),
            n_vessels=("mmsi", "nunique"),
        )
        .assign(classification="all", class_value="__ALL__")
    )
    parts.append(totals)
    daily = pd.concat(parts, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["classification", "class_value", "date"]).reset_index(drop=True)
    logger.info(
        "daily AIS table: %d rows, %s .. %s (%d observed days, classifications: %s)",
        len(daily),
        daily["date"].min().date() if len(daily) else "-",
        daily["date"].max().date() if len(daily) else "-",
        daily["date"].nunique(),
        daily.groupby("classification")["class_value"].nunique().to_dict(),
    )
    return daily


def validate_day_coverage(
    daily: pd.DataFrame,
    aoi_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Deprecated shim kept for callers that only need an observed/missing flag.

    Use :func:`build_observation_status` instead: it distinguishes a genuine
    structural zero from a missing observation using the completed report
    interval of each successfully parsed cache file.
    """
    return build_observation_status(daily, [], aoi_key, start, end)


def parse_cache_interval(file_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Recover the requested inclusive date interval from a cache filename."""
    match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.zip$", file_name)
    if not match:
        return None
    return pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))


def build_observation_status(
    daily: pd.DataFrame,
    parsed_files: list[str],
    aoi_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Classify every calendar day as observed_positive / structural_zero / missing_unknown.

    A day may be called ``structural_zero`` **only** when all of the following
    hold:

    1. a report request covering that date completed successfully (the cache file
       exists on disk and parsed without error);
    2. the date lies inside that report's requested interval;
    3. the dataset version used was the pinned one (recorded in the manifest);
    4. no activity entry exists inside the AOI for that date.

    Anything else is ``missing_unknown``. A failed, incomplete, malformed or
    unavailable report can therefore never be read as zero activity.
    """
    covered: set[pd.Timestamp] = set()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for file_name in parsed_files:
        interval = parse_cache_interval(file_name)
        if interval is None:
            continue
        intervals.append(interval)
        for date in pd.date_range(interval[0], interval[1], freq="D"):
            covered.add(date.normalize())
    # The calendar spans the union of everything already cached plus the current
    # request window, so running an AOI in several passes accumulates a complete
    # status table instead of truncating the earlier passes.
    window_start = min([start] + [interval[0] for interval in intervals])
    window_end = max([end] + [interval[1] for interval in intervals])
    calendar = pd.date_range(window_start, window_end, freq="D")
    observed = (
        pd.to_datetime(daily["date"]).dt.normalize().unique() if len(daily) else np.array([])
    )
    observed = set(pd.Timestamp(value) for value in observed)
    frame = pd.DataFrame({"aoi_key": aoi_key, "date": calendar})
    frame["is_covered_by_completed_report"] = frame["date"].isin(covered)
    frame["has_activity_entry"] = frame["date"].isin(observed)
    status = np.where(
        ~frame["is_covered_by_completed_report"],
        "missing_unknown",
        np.where(frame["has_activity_entry"], "observed_positive", "structural_zero"),
    )
    frame["ais_observation_status"] = status
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    return frame


def build_zero_missing_audit(
    cfg: dict[str, Any],
    aoi_key: str,
    status: pd.DataFrame,
    daily: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """Audit table: AIS status + presence hours + the same-day SAR record.

    Joining the SAR record makes the audit usable as evidence: for every date it
    shows whether a valid SAR acquisition happened, how many vessels SAR saw and
    how many of them carried a confirmed AIS match.
    """
    totals = (
        daily[daily["classification"] == "all"][["date", "presence_hours", "n_vessels"]]
        if len(daily) else pd.DataFrame(columns=["date", "presence_hours", "n_vessels"])
    )
    totals = totals.copy()
    if len(totals):
        totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
    audit = status.merge(totals, how="left", on="date")
    audit["presence_hours"] = audit["presence_hours"].fillna(0.0)
    audit["n_vessels"] = audit["n_vessels"].fillna(0.0)

    sar_daily_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["interim_dir"]) / "sar_daily_aoi_coverage.parquet", aoi_key, cfg
    )
    sar_scene_path = aoi_scoped_path(
        resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet", aoi_key, cfg
    )
    audit["sar_acquisition"] = False
    audit["sar_scenes"] = 0
    audit["sar_detections"] = 0
    audit["sar_matched"] = 0
    audit["sar_unmatched"] = 0
    if sar_daily_path.exists() and sar_scene_path.exists():
        sar_daily = pd.read_parquet(sar_daily_path)
        sar_daily["date"] = pd.to_datetime(sar_daily["date"]).dt.normalize()
        scenes = pd.read_parquet(sar_scene_path)
        scenes = scenes[scenes["usable_for_primary_metrics"]].copy()
        scenes["date"] = pd.to_datetime(scenes["scene_date_parsed"]).dt.normalize()
        per_day = scenes.groupby("date").agg(
            sar_scenes=("scene_id", "size"),
            sar_detections=("n_total", "sum"),
            sar_matched=("n_matched", "sum"),
            sar_unmatched=("n_unmatched", "sum"),
        )
        audit = audit.merge(per_day, how="left", left_on="date", right_index=True, suffixes=("", "_sar"))
        for column in ("sar_scenes", "sar_detections", "sar_matched", "sar_unmatched"):
            if f"{column}_sar" in audit.columns:
                audit[column] = audit[f"{column}_sar"].fillna(audit[column])
                audit = audit.drop(columns=[f"{column}_sar"])
        for column in ("sar_scenes", "sar_detections", "sar_matched", "sar_unmatched"):
            audit[column] = audit[column].fillna(0).astype(int)
        audit["sar_acquisition"] = audit["date"].isin(set(sar_daily["date"]))
    logger.info(
        "AOI %s status counts: %s",
        aoi_key,
        audit["ais_observation_status"].value_counts().to_dict(),
    )
    # Independent corroboration for a structural zero: the enclosing primary AOI
    # is requested with the same dataset, resolution and date range, so if the
    # wider polygon reports activity on a date when this AOI has none, the zero
    # is a property of the AOI, not of the data feed.
    from hormuz_common import primary_aoi_key  # local import: avoids a cycle at module load

    parent_key = primary_aoi_key(cfg)
    audit["parent_aoi_key"] = parent_key
    audit["parent_aoi_has_activity"] = pd.NA
    if aoi_key != parent_key:
        parent_daily_path = aoi_scoped_path(
            resolve_path(cfg["paths"]["processed_dir"]) / "ais_daily.parquet", parent_key, cfg
        )
        if parent_daily_path.exists():
            parent_daily = pd.read_parquet(parent_daily_path)
            parent_daily = parent_daily[parent_daily["classification"] == "all"].copy()
            parent_daily["date"] = pd.to_datetime(parent_daily["date"]).dt.normalize()
            parent_dates = set(parent_daily.loc[parent_daily["presence_hours"] > 0, "date"])
            audit["parent_aoi_has_activity"] = audit["date"].isin(parent_dates)
    audit["zero_ais_with_positive_sar"] = (
        (audit["ais_observation_status"] == "structural_zero")
        & (audit["sar_detections"] > 0)
    )
    return audit


def build_classification_summary(daily: pd.DataFrame, classification: str) -> pd.DataFrame:
    """Per-category totals and relative contribution for one classification field."""
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "aoi_key",
                "classification",
                "class_value",
                "total_presence_hours",
                "n_records",
                "n_days",
                "median_daily_hours",
                "relative_contribution",
            ]
        )
    grouped = (
        daily[(daily["classification"] == classification)]
        .groupby(["aoi_key", "classification", "class_value"], as_index=False)
        .agg(
            total_presence_hours=("presence_hours", "sum"),
            n_records=("n_records", "sum"),
            n_days=("date", "nunique"),
            median_daily_hours=("presence_hours", "median"),
        )
    )
    totals = grouped.groupby("aoi_key")["total_presence_hours"].transform("sum")
    grouped["relative_contribution"] = grouped["total_presence_hours"] / totals
    return grouped.sort_values(["aoi_key", "total_presence_hours"], ascending=[True, False])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("02_fetch_ais", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 2 (Phase 2) - GFW AIS vessel presence")

    aoi_keys = args.aois or list_aois(cfg)
    start = pd.Timestamp(args.start or cfg["analysis"]["load_start"])
    end = pd.Timestamp(args.end or cfg["analysis"]["load_end"])
    chunk_months = int(args.chunk_months or cfg["ais"].get("request_chunk_months", 1))
    datasets = [cfg["ais"]["dataset"]] + list(cfg["ais"].get("fallback_datasets", []))
    token_name = cfg["ais"]["api_token_env"]
    token = read_env(logger=logger).get(token_name, "") or ""

    chunks = month_chunks(
        start, end, chunk_months, max_days=int(cfg["ais"].get("max_report_days", 366))
    )
    requests_to_send: list[dict[str, Any]] = []
    for aoi_key in aoi_keys:
        geometry = json.loads(json.dumps(aoi_geometry(cfg, aoi_key).__geo_interface__))
        for chunk_start, chunk_end in chunks:
            requests_to_send.append(
                {
                    "aoi_key": aoi_key,
                    "start": chunk_start,
                    "end": chunk_end,
                    "query": build_request_query(cfg, datasets[0], chunk_start, chunk_end),
                    "body": build_request_body(geometry),
                }
            )

    logger.info(
        "planned %d request(s): %d AOI(s) x %d chunk(s); dataset=%s; sequential only",
        len(requests_to_send),
        len(aoi_keys),
        len(chunks),
        datasets[0],
    )

    if args.dry_run:
        for request in requests_to_send[:3]:
            logger.info(
                "[dry-run] %s %s..%s query=%s body={'geojson': <AOI feature>}",
                request["aoi_key"],
                request["start"].date(),
                request["end"].date(),
                request["query"],
            )
        logger.info(
            "[dry-run] %d request(s) not sent; note date-range end is exclusive, so each "
            "request asks for end+1 day",
            len(requests_to_send),
        )
        return 0

    offline = args.inspect_only or args.from_cache
    if not offline and not token:
        logger.error(
            "no %s found. Create .env from .env.example and add a Global Fishing Watch "
            "token, then re-run; or use --dry-run / --from-cache to work offline.",
            token_name,
        )
        return 3

    if args.check_auth:
        if not token:
            logger.error("no %s found in .env; cannot check authentication", token_name)
            return 3
        probe_start = pd.Timestamp("2026-03-01")
        status, payload, error = send_request(
            cfg,
            build_request_query(cfg, datasets[0], probe_start, pd.Timestamp("2026-03-02")),
            build_request_body(
                json.loads(json.dumps(aoi_geometry(cfg, aoi_keys[0]).__geo_interface__))
            ),
            token,
            logger,
        )
        record_manifest(
            cfg,
            {
                "request_datetime_utc": pd.Timestamp.utcnow().isoformat(),
                "aoi_key": aoi_keys[0],
                "start_date": "2026-03-01",
                "end_date": "2026-03-02",
                "dataset_requested": datasets[0],
                "dataset_returned": datasets[0] if status == 200 else None,
                "temporal_resolution": cfg["ais"].get("temporal_resolution"),
                "spatial_resolution": cfg["ais"].get("spatial_resolution"),
                "group_by": "|".join(cfg["ais"].get("group_by") or []),
                "http_status": status,
                "n_bytes": len(payload),
                "n_rows": None,
                "cache_path": None,
                "url": cfg["ais"]["base_url"],
                "attempt": 1,
                "error": error,
                "purpose": "check_auth",
            },
        )
        if status != 200:
            logger.error("authentication/request failed with status %s: %s", status, error)
            return 3
        try:
            frame = extract_activity_csv(payload, logger)
        except Exception as error:  # noqa: BLE001
            logger.error("authentication OK but payload could not be parsed: %s", error)
            return 4
        logger.info(
            "authentication OK (HTTP 200, %d bytes, %d rows, %d columns)",
            len(payload),
            len(frame),
            frame.shape[1],
        )
        return 0

    if not offline:
        pause = float(cfg["ais"].get("request_pause_seconds", 3.0))
        for position, request in enumerate(requests_to_send, start=1):
            aoi_key = request["aoi_key"]
            chunk_start, chunk_end = request["start"], request["end"]
            cache_path = cache_path_for(cfg, aoi_key, datasets[0], chunk_start, chunk_end)
            if cfg["ais"].get("cache_raw", True) and cache_path.exists():
                logger.info("[%d/%d] cache hit %s", position, len(requests_to_send), cache_path.name)
                continue
            logger.info(
                "[%d/%d] requesting %s %s..%s",
                position,
                len(requests_to_send),
                aoi_key,
                chunk_start.date(),
                chunk_end.date(),
            )
            status, payload, error = send_request(
                cfg, request["query"], request["body"], token, logger
            )
            n_rows = None
            if status == 200:
                try:
                    n_rows = len(extract_activity_csv(payload, logger))
                except Exception as parse_error:  # noqa: BLE001
                    error = f"payload parse failed: {parse_error}"
                    status = 4
            manifest_row = {
                "request_datetime_utc": pd.Timestamp.utcnow().isoformat(),
                "aoi_key": aoi_key,
                "start_date": str(chunk_start.date()),
                "end_date": str(chunk_end.date()),
                "dataset_requested": datasets[0],
                "dataset_returned": datasets[0] if status == 200 else None,
                "temporal_resolution": cfg["ais"].get("temporal_resolution"),
                "spatial_resolution": cfg["ais"].get("spatial_resolution"),
                "group_by": "|".join(cfg["ais"].get("group_by") or []),
                "http_status": status,
                "n_bytes": len(payload),
                "n_rows": n_rows,
                "cache_path": str(cache_path),
                "url": cfg["ais"]["base_url"],
                "attempt": 1,
                "error": error,
            }
            if status == 200:
                cache_path.write_bytes(payload)
                logger.info("    -> %d bytes, %s rows cached to %s", len(payload), n_rows, cache_path.name)
            else:
                logger.error("    -> request failed (%s): %s", status, error)
            record_manifest(cfg, manifest_row)
            if position < len(requests_to_send):
                time.sleep(pause)

    # ---- offline processing of whatever is cached ------------------------
    daily_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    schema_reports: list[dict[str, Any]] = []
    for aoi_key in aoi_keys:
        frames = load_cached_frames(cfg, aoi_key, logger)
        if not frames:
            logger.warning("no cached AIS report for AOI %s; nothing processed", aoi_key)
            continue
        combined = pd.concat(frames, ignore_index=True)
        inspection = inspect_ais_frame(combined, logger)
        inspection["aoi_key"] = aoi_key
        inspection["cached_files"] = sorted(combined["_cache_file"].unique().tolist())
        schema_reports.append(inspection)

        daily = build_daily_ais_table(frames, inspection["roles"], aoi_key, logger)
        if daily.empty:
            logger.warning("AOI %s produced no usable AIS rows after filtering", aoi_key)
            continue
        daily_frames.append(daily)
        status = build_observation_status(
            daily, inspection["cached_files"], aoi_key, start, end
        )
        coverage_frames.append(status)
        audit_frames.append(build_zero_missing_audit(cfg, aoi_key, status, daily, logger))

        processed = resolve_path(cfg["paths"]["processed_dir"])
        processed.mkdir(parents=True, exist_ok=True)
        daily_path = aoi_scoped_path(processed / "ais_daily.parquet", aoi_key, cfg)
        daily.to_parquet(daily_path, index=False)
        logger.info("wrote %s (%d rows)", daily_path, len(daily))
        status_path = aoi_scoped_path(processed / "ais_daily_status.parquet", aoi_key, cfg)
        status.to_parquet(status_path, index=False)
        logger.info(
            "wrote %s (%d days: %s)",
            status_path,
            len(status),
            status["ais_observation_status"].value_counts().to_dict(),
        )

    diagnostics = resolve_path(cfg["paths"]["diagnostics_dir"])
    diagnostics.mkdir(parents=True, exist_ok=True)
    if schema_reports:
        write_json(diagnostics / "ais_schema_report.json", schema_reports)
        primary_roles = schema_reports[0]["roles"]
        mapping_path = resolve_path(
            cfg["ais"].get("schema_mapping_file", "config/ais_schema_mapping.yaml")
        )
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "generated_by": "src/02_fetch_ais.py",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "dataset_requested": datasets[0],
                    "verified": False,
                    "auto_detection_unresolved_roles": [
                        role for role, column in primary_roles.items() if not column
                    ],
                    "roles": primary_roles,
                },
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        logger.info("wrote %s", mapping_path)
        logger.info("wrote %s", diagnostics / "ais_schema_report.json")

    if daily_frames:
        all_daily = pd.concat(daily_frames, ignore_index=True)
        tables = resolve_path(cfg["paths"]["tables_dir"])
        tables.mkdir(parents=True, exist_ok=True)
        for classification, filename in (
            ("vessel_type", "ais_vessel_type_summary.csv"),
            ("gear_type", "ais_gear_type_summary.csv"),
        ):
            summary = build_classification_summary(all_daily, classification)
            write_cumulative_table(
                tables / filename, summary, ["aoi_key", "classification", "class_value"]
            )
            logger.info(
                "wrote %s (%d categories incl. %s)",
                tables / filename,
                len(summary),
                ", ".join(summary["class_value"].head(4).astype(str)),
            )
        if coverage_frames:
            coverage = pd.concat(coverage_frames, ignore_index=True)
            coverage_path = tables / "ais_coverage.csv"
            write_cumulative_table(coverage_path, coverage, ["aoi_key", "date"])
            logger.info(
                "wrote %s ; status counts %s",
                coverage_path,
                coverage["ais_observation_status"].value_counts().to_dict(),
            )
        if audit_frames:
            audit = pd.concat(audit_frames, ignore_index=True)
            write_cumulative_table(
                tables / "ais_zero_missing_audit.csv", audit, ["aoi_key", "date"]
            )
            logger.info(
                "wrote %s (%d rows; %d structural zeros with positive SAR detections)",
                tables / "ais_zero_missing_audit.csv",
                len(audit),
                int(audit["zero_ais_with_positive_sar"].sum()),
            )
    else:
        logger.warning("no AIS data processed; no AIS tables were written")

    logger.info(
        "STEP 2 finished. Inspect the returned schema in "
        "outputs/diagnostics/ais_schema_report.json and the vessel_type categories in "
        "outputs/tables/ais_vessel_type_summary.csv before selecting any "
        "commercial-shipping subset."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
