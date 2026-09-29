"""STEP 0 - schema discovery for the GFW Sentinel-1 SAR archive.

Purpose
-------
Never assume GFW column names. This script

1. inventories every SAR detection / footprint file it can find,
2. profiles one deterministic sample per *schema signature*
   (``kind`` x ``pipeline version``),
3. exactly re-scans the files inside the configured loading window for row
   counts and temporal coverage,
4. detects the analysis roles (scene id, datetime, lat/lon, AIS-match label,
   footprint WKT, scores, length, platform, pipeline) and writes
   ``config/schema_mapping.yaml``,
5. writes ``outputs/diagnostics/schema_report.json`` and
   ``outputs/diagnostics/schema_report.md``.

If a *required* role cannot be resolved the script exits non-zero and later
scripts refuse to run (they re-check the mapping themselves).

Usage
-----
    python src/00_inspect_data.py                 # study window, exact counts
    python src/00_inspect_data.py --all-years     # exact counts for 2017-2026
    python src/00_inspect_data.py --confirm       # stamp mapping as manually confirmed
"""

from __future__ import annotations

import argparse
import importlib
import platform
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    banner,
    discover_sar_files,
    ensure_dirs,
    files_in_window,
    get_logger,
    load_config,
    parse_gfw_datetimes,
    resolve_path,
    write_json,
)

CATEGORICAL_MAX_UNIQUE = 60
PROFILE_ROWS_DEFAULT = 200_000

# Role -> candidate regular expressions, evaluated in order.
ROLE_PATTERNS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("datetime", [r"^timestamp$", r"^datetime$", r"^date_time$", r"time", r"date"]),
        ("latitude", [r"^lat$", r"^latitude$", r"^lat_dd$", r"^y$"]),
        ("longitude", [r"^lon$", r"^lng$", r"^long$", r"^longitude$", r"^x$"]),
        ("scene_id", [r"^scene_id$", r"^sceneid$", r"^scene$", r"^id$"]),
        ("footprint_wkt", [r"^footprint_wkt$", r"wkt", r"geometry", r"^geom$"]),
        ("matching_score", [r"^matching_score$", r"matching_score"]),
        ("presence_score", [r"^presence_score$", r"presence_score"]),
        ("fishing_score", [r"^fishing_score$", r"fishing_score"]),
        ("mmsi", [r"^mmsi$", r"vessel_id", r"shipid"]),
        ("length_m", [r"^length_m$", r"^length$", r"length"]),
        ("match_label", [r"matched_category", r"^match(ed)?_(category|label|status)$", r"^ais_match.*"]),
    ]
)


# ---------------------------------------------------------------------------
# low level CSV helpers (pyarrow keeps memory bounded on 400 MB footprint files)
# ---------------------------------------------------------------------------
def iter_csv_batches(
    path: Path,
    columns: Sequence[str] | None = None,
    block_size: int = 1 << 25,
) -> Iterator[pa.RecordBatch]:
    """Yield record batches from a CSV without loading the whole file."""
    convert_options = pa_csv.ConvertOptions(include_columns=list(columns)) if columns else None
    reader = pa_csv.open_csv(
        str(path),
        read_options=pa_csv.ReadOptions(block_size=block_size),
        parse_options=pa_csv.ParseOptions(),
        convert_options=convert_options,
    )
    yield from reader


def scan_csv(
    path: Path,
    columns: Sequence[str] | None = None,
    sample_rows: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Read up to ``sample_rows`` rows and count all rows in one pass.

    Returns
    -------
    (pandas.DataFrame, int)
        The sampled frame and the exact number of data rows in the file.
    """
    frames: list[pd.DataFrame] = []
    collected = 0
    total = 0
    for batch in iter_csv_batches(path, columns=columns):
        total += batch.num_rows
        if sample_rows is None or collected < sample_rows:
            take = batch if sample_rows is None else batch.slice(0, sample_rows - collected)
            frames.append(take.to_pandas())
            collected += take.num_rows
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, total


def count_rows_and_columns(path: Path, columns: Sequence[str]) -> tuple[int, pd.DataFrame]:
    """Exact row count plus min/max for the requested (cheap) columns."""
    total = 0
    mins: dict[str, Any] = {}
    maxs: dict[str, Any] = {}
    for batch in iter_csv_batches(path, columns=columns):
        total += batch.num_rows
        for column in columns:
            array = batch.column(column)
            if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
                as_pd = array.to_pandas().dropna()
                if len(as_pd):
                    if column in mins:
                        mins[column] = min(mins[column], as_pd.min())
                        maxs[column] = max(maxs[column], as_pd.max())
                    else:
                        mins[column] = as_pd.min()
                        maxs[column] = as_pd.max()
            elif pa.types.is_floating(array.type) or pa.types.is_integer(array.type):
                with np.errstate(invalid="ignore"):
                    current_min = np.nanmin(np.asarray(array.to_pandas(), dtype=float))
                    current_max = np.nanmax(np.asarray(array.to_pandas(), dtype=float))
                if column in mins:
                    mins[column] = float(np.nanmin([mins[column], current_min]))
                    maxs[column] = float(np.nanmax([maxs[column], current_max]))
                else:
                    mins[column] = float(current_min)
                    maxs[column] = float(current_max)
    summary = pd.DataFrame({"column": columns})
    summary["min"] = [mins.get(column) for column in columns]
    summary["max"] = [maxs.get(column) for column in columns]
    return total, summary


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------
def profile_frame(frame: pd.DataFrame, kind: str) -> dict[str, Any]:
    """Describe columns, dtypes, missingness and categorical cardinality."""
    columns: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for column in frame.columns:
        series = frame[column]
        missing_rate = float(series.isna().mean()) if len(series) else float("nan")
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_rate": missing_rate,
            "n_unique_sampled": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                entry.update(
                    {
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "mean": float(clean.mean()),
                        "median": float(clean.median()),
                    }
                )
                entry["negative_share"] = float((clean < 0).mean())
        else:
            clean = series.dropna().astype(str)
            if len(clean):
                entry["example_values"] = clean.drop_duplicates().head(5).tolist()
                entry["max_length"] = int(clean.str.len().max())
            if len(clean) and clean.nunique() <= CATEGORICAL_MAX_UNIQUE:
                counts = clean.value_counts()
                entry["categorical"] = True
                entry["value_counts"] = {str(k): int(v) for k, v in counts.items()}
                entry["value_shares"] = {
                    str(k): float(v / len(clean)) for k, v in counts.items()
                }
            else:
                entry["categorical"] = False
        columns[column] = entry
    return {"n_rows_as_profiled": int(len(frame)), "columns": columns}


def looks_like_datetime(series: pd.Series, column: str, logger) -> float:
    """Return the share of parseable datetime values for a textual column."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return 0.0
    sample = series.dropna().astype(str).head(2000)
    if sample.empty:
        return 0.0
    parsed = parse_gfw_datetimes(sample, logger=None)
    return float(parsed.notna().mean())


def detect_roles(
    profiles: dict[str, dict[str, Any]],
    logger,
) -> dict[str, Any]:
    """Resolve analysis roles from the profiled columns.

    Returns a mapping with one resolved column name per role (or ``None``),
    plus the evidence used for the decision.
    """
    # normalise to the inner profile dict so the rest of the function can treat
    # every entry as {"n_rows_as_profiled": int, "columns": {...}}
    detection_profiles = {
        key: value["profile"] for key, value in profiles.items() if value["kind"] == "detections"
    }
    footprint_profiles = {
        key: value["profile"] for key, value in profiles.items() if value["kind"] == "footprints"
    }
    if not detection_profiles:
        return {"resolved": {}, "evidence": {}, "unresolved": ["*"], "reason": "no detection files profiled"}

    detection_columns: list[str] = []
    for profile in detection_profiles.values():
        for column in profile["columns"]:
            if column not in detection_columns:
                detection_columns.append(column)
    footprint_columns: list[str] = []
    for profile in footprint_profiles.values():
        for column in profile["columns"]:
            if column not in footprint_columns:
                footprint_columns.append(column)

    def first_match(candidates: Sequence[str], target_columns: Sequence[str]) -> str | None:
        lowered = {column.lower(): column for column in target_columns}
        for pattern in candidates:
            regex = re.compile(pattern, re.IGNORECASE)
            for column in target_columns:
                if regex.search(column.lower()):
                    return lowered[column.lower()]
        return None

    resolved: dict[str, Any] = {}
    evidence: dict[str, Any] = {}

    # --- datetime ---------------------------------------------------------
    for pattern in ROLE_PATTERNS["datetime"]:
        regex = re.compile(pattern, re.IGNORECASE)
        best: tuple[float, str] | None = None
        for column in detection_columns:
            if not regex.search(column.lower()):
                continue
            rates = []
            for profile in detection_profiles.values():
                if column in profile["columns"]:
                    rates.append(profile["columns"][column].get("datetime_parse_rate", 0.0))
            rate = float(np.mean(rates)) if rates else 0.0
            if rate >= 0.99 and (best is None or rate > best[0]):
                best = (rate, column)
        if best:
            resolved["datetime"] = best[1]
            evidence["datetime"] = {"parse_rate": best[0], "matched_pattern": pattern}
            break

    resolved["latitude"] = first_match(ROLE_PATTERNS["latitude"], detection_columns)
    resolved["longitude"] = first_match(ROLE_PATTERNS["longitude"], detection_columns)

    # --- scene id (detections and footprints) -----------------------------
    scene_column = first_match(ROLE_PATTERNS["scene_id"], detection_columns)
    resolved["scene_id"] = scene_column
    evidence["scene_id"] = {
        "column": scene_column,
        "unique_sampled": (
            detection_profiles[next(iter(detection_profiles))]["columns"]
            .get(scene_column, {})
            .get("n_unique_sampled")
            if scene_column
            else None
        ),
    }

    footprint_scene_column = first_match(ROLE_PATTERNS["scene_id"], footprint_columns)
    resolved["footprint_scene_id"] = footprint_scene_column
    resolved["footprint_wkt"] = first_match(ROLE_PATTERNS["footprint_wkt"], footprint_columns)
    resolved["footprint_date"] = first_match([r"^date$", r"date"], footprint_columns)
    resolved["footprint_start_time"] = first_match([r"start_time", r"^start$"], footprint_columns)
    resolved["footprint_end_time"] = first_match([r"end_time", r"^end$"], footprint_columns)

    # --- AIS matching -----------------------------------------------------
    unmatched_sentinel = None
    match_label_column = None
    for profile in detection_profiles.values():
        candidate = first_match(ROLE_PATTERNS["match_label"], list(profile["columns"]))
        if candidate:
            counts = profile["columns"][candidate].get("value_counts", {})
            if counts:
                match_label_column = candidate
                for value in counts:
                    if str(value).strip().lower() == "unmatched":
                        unmatched_sentinel = value
                break
    resolved["match_label"] = match_label_column
    resolved["match_value_unmatched"] = unmatched_sentinel
    evidence["match_label"] = {
        "column": match_label_column,
        "unmatched_sentinel": unmatched_sentinel,
        "note": (
            "role resolved from a categorical column that contains an explicit "
            "'unmatched' level"
            if unmatched_sentinel
            else "categorical match column found but no 'unmatched' level; "
            "downstream will fall back to the MMSI-based definition"
        ),
    }

    for role in ("matching_score", "presence_score", "fishing_score", "mmsi", "length_m"):
        resolved[role] = first_match(ROLE_PATTERNS[role], detection_columns)
    resolved["matched_category"] = match_label_column

    required = ["scene_id", "datetime", "latitude", "longitude"]
    unresolved = [role for role in required if not resolved.get(role)]
    if not resolved.get("match_label") and not resolved.get("mmsi"):
        unresolved.append("match_label_or_mmsi")

    return {
        "resolved": resolved,
        "evidence": evidence,
        "unresolved": unresolved,
        "detection_columns": detection_columns,
        "footprint_columns": footprint_columns,
    }


def infer_crs(frame: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    """Infer/verify EPSG:4326 for lon/lat columns from their numeric ranges."""
    lat_col, lon_col = mapping.get("latitude"), mapping.get("longitude")
    if not lat_col or not lon_col or lat_col not in frame.columns or lon_col not in frame.columns:
        return {"crs": None, "reason": "latitude/longitude columns not available"}
    lat = pd.to_numeric(frame[lat_col], errors="coerce").dropna()
    lon = pd.to_numeric(frame[lon_col], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return {"crs": None, "reason": "empty coordinate columns"}
    within = bool(lat.between(-90, 90).all() and lon.between(-180, 180).all())
    swapped = bool(lat.between(-180, 180).all() and lon.between(-90, 90).all())
    return {
        "crs": "EPSG:4326" if within else None,
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "coordinates_within_wgs84_envelope": within,
        "possible_lat_lon_swap": swapped and not within,
    }


def infer_footprint_crs(frame: pd.DataFrame, wkt_column: str | None) -> dict[str, Any]:
    """Infer the footprint CRS from WKT coordinate ranges."""
    if not wkt_column or wkt_column not in frame.columns:
        return {"crs": None, "reason": "footprint WKT column not available"}
    sample = frame[wkt_column].dropna().astype(str).head(25)
    if sample.empty:
        return {"crs": None, "reason": "empty WKT column"}
    max_abs = 0.0
    prefix_counts: dict[str, int] = {}
    for text in sample:
        prefix = text.split("(")[0].strip().upper()
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            max_abs = max(max_abs, max(abs(float(value)) for value in numbers))
    return {
        "crs": "EPSG:4326" if max_abs <= 180.0 else None,
        "max_abs_coordinate_sampled": max_abs,
        "geometry_types": prefix_counts,
        "coordinates_within_wgs84_envelope": max_abs <= 180.0,
    }


# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame, float_format: str = "{:.6g}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return "_(empty)_"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else float_format.format(value)
            )
        else:
            formatted[column] = formatted[column].astype(str)
    header = "| " + " | ".join(str(column) for column in formatted.columns) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _collect_label_values(
    signatures: dict[str, Any],
    resolved: dict[str, Any],
    exclude: set[str] | None = None,
    include: set[str] | None = None,
) -> set[str]:
    """Collect distinct sampled levels of the match-label column across signatures.

    ``exclude`` removes normalised values (e.g. the ``unmatched`` sentinel);
    ``include`` restricts the result to specific normalised values
    (e.g. unknown placeholders such as ``NA``).
    """
    column = resolved.get("match_label")
    if not column:
        return set()
    collected: set[str] = set()
    for key, signature in signatures.items():
        if not key.startswith("detections"):
            continue
        counts = signature["profile"]["columns"].get(column, {}).get("value_counts", {})
        for value in counts:
            normalised = str(value).strip().lower()
            if exclude and normalised in exclude:
                continue
            if include is not None and normalised not in include:
                continue
            collected.add(str(value))
    return collected


def build_markdown_report(report: dict[str, Any]) -> str:
    """Assemble the human-readable schema report."""
    lines: list[str] = []
    lines.append("# GFW SAR schema discovery report")
    lines.append("")
    lines.append(f"Generated by `src/00_inspect_data.py` on {report['generated_at']}.")
    lines.append("")
    lines.append("## 1. Environment")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(report["environment"]["packages"], index=[0]).T.reset_index().rename(columns={"index": "package", 0: "version"})))
    lines.append("")
    lines.append(f"- Python: `{report['environment']['python']}`")
    lines.append(f"- Platform: `{report['environment']['platform']}`")
    lines.append(f"- Config: `{report['config_path']}`")
    lines.append(f"- Files discovered: {report['inventory']['n_files']} "
                 f"(detections: {report['inventory']['n_detection_files']}, "
                 f"footprints: {report['inventory']['n_footprint_files']})")
    lines.append(f"- Global row count (files with exact counts only): "
                 f"{report['inventory']['total_rows_scanned']}")

    lines.append("")
    lines.append("## 2. Resolved analysis roles (schema mapping)")
    lines.append("")
    mapping_frame = pd.DataFrame(
        [{"role": role, "column": column} for role, column in report["role_mapping"]["resolved"].items()]
    )
    lines.append(markdown_table(mapping_frame))
    lines.append("")
    if report["role_mapping"]["unresolved"]:
        lines.append(f"> **UNRESOLVED REQUIRED ROLES: {report['role_mapping']['unresolved']}**")
    else:
        lines.append("> All required roles were resolved.")
    lines.append("")
    lines.append("Evidence:")
    lines.append("")
    evidence_rows = []
    for role, evidence in report["role_mapping"]["evidence"].items():
        evidence_rows.append({"role": role, "evidence": str(evidence)})
    lines.append(markdown_table(pd.DataFrame(evidence_rows)))

    lines.append("")
    lines.append("## 3. Schema signatures")
    lines.append("")
    lines.append(
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "signature": key,
                        "kind": value["kind"],
                        "pipeline": value["pipeline"],
                        "n_files": value["n_files"],
                        "sample_rows": value["profile"]["n_rows_as_profiled"],
                        "columns": ", ".join(value["profile"]["columns"].keys()),
                    }
                    for key, value in report["signatures"].items()
                ]
            )
        )
    )

    lines.append("")
    lines.append("## 4. Column profiles per signature")
    for key, value in report["signatures"].items():
        lines.append("")
        lines.append(f"### {key}")
        lines.append("")
        rows = []
        for column, meta in value["profile"]["columns"].items():
            rows.append(
                {
                    "column": column,
                    "dtype": meta["dtype"],
                    "missing_rate": meta["missing_rate"],
                    "n_unique_sampled": meta["n_unique_sampled"],
                    "min": meta.get("min"),
                    "max": meta.get("max"),
                    "median": meta.get("median"),
                    "categorical": meta.get("categorical"),
                    "example_values": "; ".join(map(str, meta.get("example_values", [])))[:110],
                }
            )
        lines.append(markdown_table(pd.DataFrame(rows)))
        lines.append("")
        lines.append(f"CRS evidence: `{value['crs_evidence']}`")
        for column, meta in value["profile"]["columns"].items():
            if meta.get("value_counts"):
                lines.append("")
                lines.append(f"Categorical values for `{column}`:")
                lines.append("")
                counts = pd.DataFrame(
                    {
                        "value": list(meta["value_counts"].keys()),
                        "count_sampled": list(meta["value_counts"].values()),
                        "share_sampled": [
                            meta["value_shares"][key_] for key_ in meta["value_counts"]
                        ],
                    }
                )
                lines.append(markdown_table(counts))

    lines.append("")
    lines.append("## 5. Per-file inventory")
    lines.append("")
    inventory = pd.DataFrame(report["inventory"]["files"])
    if not inventory.empty:
        lines.append(
            markdown_table(
                inventory[
                    [
                        "kind",
                        "file_name",
                        "pipeline",
                        "yyyymm",
                        "row_count",
                        "temporal_min",
                        "temporal_max",
                        "size_mb",
                        "counts_scanned_exactly",
                    ]
                ]
            )
        )
    lines.append("")
    lines.append("## 6. Pipeline version timeline")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(report["pipeline_timeline"])))
    lines.append("")
    lines.append("## 7. Coverage gaps (months with no file)")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame({"missing_month": report["missing_months"]})))
    lines.append("")
    lines.append("## 8. Filter ledger (de-duplication)")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(report["discovery_ledger"])))
    lines.append("")
    lines.append("## 9. Warnings")
    lines.append("")
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GFW SAR schema discovery (step 0)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--profile-rows", type=int, default=PROFILE_ROWS_DEFAULT,
                        help="rows sampled per schema signature for column profiling")
    parser.add_argument("--all-years", action="store_true",
                        help="also compute exact row counts outside the loading window")
    parser.add_argument("--confirm", action="store_true",
                        help="stamp the generated schema mapping as manually confirmed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("00_inspect_data", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 0 - GFW SAR schema discovery")

    load_start = pd.Timestamp(cfg["analysis"]["load_start"])
    load_end = pd.Timestamp(cfg["analysis"]["load_end"])

    selected, discovery_ledger = discover_sar_files(cfg, logger)
    if selected.empty:
        logger.error(
            "No GFW SAR files found. Expected annual subfolders under %s containing files "
            "matching '%s' / '%s'.",
            cfg["paths"]["sar_archive_root"],
            cfg["sar"]["detection_filename_pattern"],
            cfg["sar"]["footprint_filename_pattern"],
        )
        return 2
    discovery_ledger.log_summary(logger)

    in_window = pd.concat(
        [
            files_in_window(selected, load_start, load_end, kind="detections"),
            files_in_window(selected, load_start, load_end, kind="footprints"),
        ],
        ignore_index=True,
    )
    logger.info(
        "loading window %s .. %s contains %d detection and %d footprint files",
        load_start.date(),
        load_end.date(),
        int((in_window["kind"] == "detections").sum()),
        int((in_window["kind"] == "footprints").sum()),
    )

    # ---- 1. signature profiling -----------------------------------------
    signatures: dict[str, dict[str, Any]] = {}
    for kind in ("detections", "footprints"):
        subset = selected[selected["kind"] == kind]
        if subset.empty:
            continue
        # profile from files inside the study window when possible
        window_subset = files_in_window(selected, load_start, load_end, kind=kind)
        source = window_subset if not window_subset.empty else subset
        for pipeline, group in source.groupby("pipeline", dropna=False):
            pipeline_label = "n/a" if pipeline is None or pd.isna(pipeline) else str(pipeline)
            key = f"{kind}|pipeline={pipeline_label}"
            if key in signatures:
                continue
            path = Path(group.iloc[0]["path"])
            frame, total = scan_csv(path, sample_rows=args.profile_rows)
            profile = profile_frame(frame, kind)
            profile["sampled_from"] = path.name
            profile["sampled_rows_of_total"] = min(args.profile_rows, total)
            # add datetime parseability evidence for text columns
            for column in frame.columns:
                if not pd.api.types.is_numeric_dtype(frame[column]):
                    rate = looks_like_datetime(frame[column], column, logger)
                    profile["columns"][column]["datetime_parse_rate"] = rate
            signatures[key] = {
                "kind": kind,
                "pipeline": pipeline_label,
                "n_files": int(len(group)),
                "files": group["file_name"].tolist(),
                "profile": profile,
                "crs_evidence": (
                    infer_footprint_crs(frame, None)
                    if kind == "footprints"
                    else {}
                ),
            }
            logger.info(
                "profiled signature %s from %s (%d/%d rows, %d columns)",
                key,
                path.name,
                profile["n_rows_as_profiled"],
                total,
                len(frame.columns),
            )

    # ---- 2. role detection ----------------------------------------------
    role_info = detect_roles(signatures, logger)
    resolved = role_info.get("resolved", {})
    logger.info("resolved roles: %s", {k: v for k, v in resolved.items() if v})

    # attach real CRS evidence now that roles are known
    for key, value in signatures.items():
        path = Path(selected[selected["file_name"] == value["files"][0]].iloc[0]["path"])
        frame, _ = scan_csv(path, sample_rows=args.profile_rows)
        if value["kind"] == "detections":
            value["crs_evidence"] = infer_crs(frame, resolved)
        else:
            value["crs_evidence"] = infer_footprint_crs(frame, resolved.get("footprint_wkt"))

    # ---- 3. exact per-file scan inside the window ------------------------
    detection_datetime = resolved.get("datetime")
    scene_column = resolved.get("scene_id")
    footprint_scene = resolved.get("footprint_scene_id")

    file_records: list[dict[str, Any]] = []
    total_rows_scanned = 0
    for row in selected.itertuples(index=False):
        path = Path(row.path)
        scan_exactly = args.all_years or bool(
            ((in_window["path"] == row.path).any())
        )
        record: dict[str, Any] = {
            "kind": row.kind,
            "file_name": row.file_name,
            "path": str(path),
            "year": row.year,
            "month": row.month,
            "day": row.day,
            "yyyymm": row.yyyymm,
            "pipeline": row.pipeline,
            "is_partial_month": bool(row.is_partial_month),
            "size_mb": round(row.size_bytes / 1e6, 2),
            "counts_scanned_exactly": bool(scan_exactly),
        }
        if scan_exactly:
            if row.kind == "detections":
                desired = [
                    detection_datetime,
                    scene_column,
                    resolved.get("latitude"),
                    resolved.get("longitude"),
                ]
                detection_columns = set(role_info.get("detection_columns", []))
                columns = [
                    column
                    for column in OrderedDict.fromkeys(c for c in desired if c)
                    if column in detection_columns
                ]
                total, summary = count_rows_and_columns(path, columns) if columns else (0, pd.DataFrame())
            else:
                footprint_columns = set(role_info.get("footprint_columns", []))
                columns = [
                    column
                    for column in OrderedDict.fromkeys(
                        [c for c in [footprint_scene, resolved.get("footprint_date")] if c]
                    )
                    if column in footprint_columns
                ]
                total, summary = count_rows_and_columns(path, columns) if columns else (0, pd.DataFrame())
            record["row_count"] = int(total)
            total_rows_scanned += int(total)
            if not summary.empty:
                lookup = summary.set_index("column")
                if detection_datetime and detection_datetime in lookup.index:
                    record["temporal_min"] = str(lookup.loc[detection_datetime, "min"])
                    record["temporal_max"] = str(lookup.loc[detection_datetime, "max"])
                footprint_date = resolved.get("footprint_date")
                if row.kind == "footprints" and footprint_date and footprint_date in lookup.index:
                    record["temporal_min"] = str(lookup.loc[footprint_date, "min"])
                    record["temporal_max"] = str(lookup.loc[footprint_date, "max"])
        else:
            record["row_count"] = None
            record["note"] = "outside the configured loading window"
        file_records.append(record)
        if scan_exactly:
            logger.info(
                "scanned %s: %s rows", row.file_name, record.get("row_count")
            )

    # ---- 4. pipeline timeline + coverage gaps ---------------------------
    detection_files = selected[selected["kind"] == "detections"].copy()
    timeline_rows: list[dict[str, Any]] = []
    for pipeline, group in detection_files.groupby("pipeline", dropna=False):
        timeline_rows.append(
            {
                "kind": "detections",
                "pipeline": "n/a" if pipeline is None or pd.isna(pipeline) else pipeline,
                "first_month": group["yyyymm"].min(),
                "last_month": group["yyyymm"].max(),
                "n_files": int(len(group)),
            }
        )
    footprint_files = selected[selected["kind"] == "footprints"]
    if not footprint_files.empty:
        timeline_rows.append(
            {
                "kind": "footprints",
                "pipeline": "n/a",
                "first_month": footprint_files["yyyymm"].min(),
                "last_month": footprint_files["yyyymm"].max(),
                "n_files": int(len(footprint_files)),
            }
        )
    pipeline_timeline = sorted(timeline_rows, key=lambda item: item["first_month"])

    expected_months = pd.period_range(load_start, load_end, freq="M").strftime("%Y%m").tolist()
    present = set(in_window["yyyymm"])
    missing_months = [month for month in expected_months if month not in present]

    warnings: list[str] = []
    if role_info.get("unresolved"):
        warnings.append(
            f"required roles unresolved: {role_info['unresolved']} - downstream analysis is blocked"
        )
    if missing_months:
        warnings.append(
            "month(s) inside the loading window without a discovered file: "
            + ", ".join(missing_months)
        )
    pipelines = sorted({str(p) for p in detection_files["pipeline"].dropna().unique()})
    if len(pipelines) > 1:
        warnings.append(
            "multiple detection pipeline versions are present in the archive ("
            + ", ".join(pipelines)
            + "); a pipeline change is a technical break and must be treated as such"
        )
    partial_present = selected[selected["is_partial_month"]]
    if not partial_present.empty:
        warnings.append(
            "partial-month files present in the archive: "
            + ", ".join(partial_present["file_name"].tolist())
            + " (de-duplicated, see ledger)"
        )

    packages = {}
    for name in ("pandas", "numpy", "geopandas", "shapely", "pyarrow", "pyproj", "scipy", "statsmodels", "matplotlib", "yaml", "dotenv"):
        try:
            packages[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional dependency reporting only
            packages[name] = "NOT INSTALLED"

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_path": cfg["_config_path"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
        },
        "inventory": {
            "n_files": int(len(selected)),
            "n_detection_files": int((selected["kind"] == "detections").sum()),
            "n_footprint_files": int((selected["kind"] == "footprints").sum()),
            "total_rows_scanned": int(total_rows_scanned),
            "files": file_records,
        },
        "signatures": signatures,
        "role_mapping": {
            "resolved": resolved,
            "evidence": role_info.get("evidence", {}),
            "unresolved": role_info.get("unresolved", []),
            "detection_columns": role_info.get("detection_columns", []),
            "footprint_columns": role_info.get("footprint_columns", []),
        },
        "pipeline_timeline": pipeline_timeline,
        "missing_months": missing_months,
        "discovery_ledger": discovery_ledger.to_frame().to_dict(orient="records"),
        "warnings": warnings,
    }

    json_path = write_json(resolve_path(cfg["schema"]["report_json"]), report)
    md_path = resolve_path(cfg["schema"]["report_md"])
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown_report(report), encoding="utf-8")
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)

    # ---- 5. schema mapping ----------------------------------------------
    mapping: dict[str, Any] = {
        "generated_by": "src/00_inspect_data.py",
        "generated_at": report["generated_at"],
        "verified": bool(args.confirm) and not role_info.get("unresolved"),
        "verified_by": "manual_confirmation" if args.confirm else "auto_detection",
        "auto_detection_unresolved_roles": role_info.get("unresolved", []),
        "source_schema_report": str(resolve_path(cfg["schema"]["report_json"])),
        "signatures": sorted(signatures.keys()),
    }
    mapping.update({role: column for role, column in resolved.items()})
    mapping["match_definition"] = {
        "primary": (
            f"{resolved.get('match_label')} != {resolved.get('match_value_unmatched')!r}"
            if resolved.get("match_label") and resolved.get("match_value_unmatched") is not None
            else None
        ),
        "primary_description": "GFW match label column, 'unmatched' means no AIS identity was assigned",
        "alternative": "mmsi is not null",
        "alternative_column": resolved.get("mmsi"),
        "matched_label_values": sorted(_collect_label_values(signatures, resolved, exclude={"unmatched"})),
        "unknown_label_values": sorted(
            _collect_label_values(signatures, resolved, include={"na", "nan", "", "none"})
        ),
    }
    mapping["platform_rule"] = "scene_id.split('_')[0] -> S1A/S1B/S1C/S1D"
    mapping["pipeline_source"] = "GFW file name token 'pipev<version>'"
    mapping_path = resolve_path(cfg["schema"]["mapping_file"])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(mapping, handle, sort_keys=False, allow_unicode=True)
    logger.info("wrote %s", mapping_path)

    if warnings:
        logger.warning("schema discovery produced %d warning(s):", len(warnings))
        for warning in warnings:
            logger.warning("  - %s", warning)

    if role_info.get("unresolved"):
        logger.error(
            "STOP: required schema roles unresolved (%s). Do not run later steps until the "
            "mapping in %s is fixed. See %s.",
            role_info["unresolved"],
            mapping_path,
            md_path,
        )
        return 3

    logger.info("STEP 0 complete - schema mapping written to %s", mapping_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
