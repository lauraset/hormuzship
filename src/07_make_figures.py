"""STEP 7 (Phase 2) - figures.

Rules taken from the project brief
---------------------------------
* matplotlib only (no seaborn);
* one figure per file, separate files (no multi-panel layouts unless the brief
  explicitly needs two series on one timeline - figure 7 does, and that is
  documented in its docstring);
* no fixed colour palette: colours come from the active matplotlib cycle;
* every figure marks the three technical/event dates from the config:
  2026-02-28 (event onset), 2026-03-01 (GFW pipeline v3 -> v4) and
  2026-03-28 (ocean-land-mask update);
* no post-June-2026 data (the analysis window ends 2026-05-31).

Figures
-------
1. weekly ``AIS_all_days`` anomaly
2. weekly ``AIS_SAR_days`` anomaly
3. weekly SAR density anomaly
4. ``AIS_all_days`` anomaly vs SAR anomaly
5. ``AIS_SAR_days`` anomaly vs SAR anomaly
6. observability assessment gap through time
7. matching rate and the observability gap on the same event timeline

Usage
-----
    python src/07_make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    list_aois,
    load_config,
    primary_aoi_key,
    resolve_aoi,
    resolve_path,
    resolve_phases,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2 figures")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None, help="AOI keys (default: all)")
    return parser.parse_args(argv)


def break_markers(cfg: dict[str, Any]) -> list[tuple[pd.Timestamp, str]]:
    """Configured technical/event dates to annotate, oldest first."""
    markers = [
        (pd.Timestamp(cfg["analysis"]["event_date"]), "event onset"),
    ]
    for entry in cfg["analysis"].get("technical_breaks", []):
        if entry.get("inside_window", False):
            name = str(entry["name"]).replace("gfw_sar_", "").replace("_", " ")
            markers.append((pd.Timestamp(entry["date"]), name))
    markers.sort(key=lambda item: item[0])
    return markers


def annotate_breaks(ax, cfg: dict[str, Any]) -> None:
    """Draw the event/technical break lines on an axis.

    Labels are stacked downwards in axes coordinates with an opaque background so
    that three breaks only days apart stay readable.
    """
    if not cfg.get("figures", {}).get("annotate_breaks", True):
        return
    markers = break_markers(cfg)
    for index, (date, label) in enumerate(markers):
        ax.axvline(date, linestyle="--", linewidth=1.0, alpha=0.7)
        y_fraction = 0.97 - 0.085 * index
        ax.annotate(
            f"{label}\n{date.date()}",
            xy=(date, y_fraction),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
        )


def finish(fig, ax, path: Path, cfg: dict[str, Any], ylabel: str, title: str, zero_line: bool = True) -> None:
    """Apply shared styling and save one figure."""
    if zero_line:
        ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("week starting (UTC)")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def series_for(weekly: pd.DataFrame, aoi_key: str, series: str) -> pd.DataFrame:
    """Rows of one series for one AOI, sorted by week."""
    subset = weekly[(weekly["aoi_key"] == aoi_key) & (weekly["series"] == series)].copy()
    if subset.empty:
        return subset
    subset["week_start"] = pd.to_datetime(subset["week_start"])
    return subset.sort_values("week_start")


def plot_single_anomaly(
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_keys: list[str],
    series: str,
    filename: str,
    title: str,
    figures_dir: Path,
    logger,
) -> bool:
    """One-line-per-AOI anomaly figure for a single series."""
    plotted = False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for aoi_key in aoi_keys:
        subset = series_for(weekly, aoi_key, series)
        if subset.empty:
            continue
        label = resolve_aoi(cfg, aoi_key)[1].get("label", aoi_key)
        ax.plot(subset["week_start"], subset["log_anomaly"], marker="o", markersize=3, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        logger.warning("figure %s skipped: no data for series %s", filename, series)
        return False
    annotate_breaks(ax, cfg)
    ax.legend(fontsize=8)
    finish(fig, ax, figures_dir / filename, cfg, "log anomaly vs same ISO week (2024/2025)", title)
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_pair(
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_keys: list[str],
    ais_series: str,
    sar_series: str,
    filename: str,
    title: str,
    figures_dir: Path,
    logger,
) -> bool:
    """AIS anomaly and SAR anomaly against the same timeline, one file per AOI."""
    plotted = False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for aoi_key in aoi_keys:
        ais = series_for(weekly, aoi_key, ais_series)
        sar = series_for(weekly, aoi_key, sar_series)
        if not ais.empty:
            ax.plot(ais["week_start"], ais["log_anomaly"], marker="o", markersize=3,
                    label=f"{aoi_key}: AIS anomaly")
            plotted = True
        if not sar.empty:
            ax.plot(sar["week_start"], sar["log_anomaly"], marker="s", markersize=3,
                    linestyle="--", label=f"{aoi_key}: SAR anomaly")
            plotted = True
    if not plotted:
        plt.close(fig)
        logger.warning("figure %s skipped: no data", filename)
        return False
    annotate_breaks(ax, cfg)
    ax.legend(fontsize=8)
    finish(fig, ax, figures_dir / filename, cfg, "log anomaly vs same ISO week (2024/2025)", title)
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_gap(
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_keys: list[str],
    filename: str,
    figures_dir: Path,
    logger,
) -> bool:
    """Observability assessment gap (AIS anomaly - SAR anomaly) through time."""
    plotted = False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for aoi_key in aoi_keys:
        for series_name in ("AIS_all_days|ALL", "AIS_SAR_days|ALL"):
            subset = series_for(weekly, aoi_key, series_name)
            if subset.empty:
                continue
            ax.plot(subset["week_start"], subset["observability_assessment_gap"],
                    marker="o", markersize=3, label=f"{aoi_key}: {series_name.split('|')[0]}")
            plotted = True
    if not plotted:
        plt.close(fig)
        logger.warning("figure %s skipped: no AIS series available", filename)
        return False
    annotate_breaks(ax, cfg)
    ax.legend(fontsize=8)
    finish(
        fig, ax, figures_dir / filename, cfg,
        "observability assessment gap\n(AIS anomaly - SAR anomaly)",
        "Observability assessment gap (AIS anomaly - SAR anomaly)",
    )
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_matching_rate_and_gap(
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
    aoi_keys: list[str],
    figures_dir: Path,
    logger,
) -> bool:
    """Matching rate and the gap on one timeline.

    Two stacked panels in one file are used deliberately: the matching rate and
    the anomaly gap are in different units, and the brief requires them to be
    read against the same event timeline. Overlaying them on twin axes would
    invite a false reading, so they share the x-axis instead.
    """
    processed = resolve_path(cfg["paths"]["processed_dir"])
    collected: dict[str, pd.DataFrame] = {}
    for aoi_key in aoi_keys:
        path = aoi_scoped_path(processed / "sar_weekly.parquet", aoi_key, cfg)
        if path.exists():
            frame = pd.read_parquet(path)
            frame["week_start"] = pd.to_datetime(frame["week_start"])
            collected[aoi_key] = frame
    if not collected and weekly.empty:
        logger.warning("figure 7 skipped: no data")
        return False

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top, bottom = axes
    for aoi_key, frame in collected.items():
        top.plot(frame["week_start"], frame["matching_rate_scene_median"],
                 marker="o", markersize=3, label=f"{aoi_key}: median scene matching rate")
    for aoi_key in aoi_keys:
        subset = series_for(weekly, aoi_key, "AIS_SAR_days|ALL")
        if not subset.empty:
            bottom.plot(subset["week_start"], subset["observability_assessment_gap"],
                        marker="o", markersize=3, label=f"{aoi_key}: gap (SAR-day AIS)")
    top.set_ylabel("matching rate")
    bottom.set_ylabel("AIS anomaly - SAR anomaly")
    bottom.set_xlabel("week starting (UTC)")
    for ax, title in ((top, "SAR-AIS matching rate"), (bottom, "Observability assessment gap")):
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.6)
        annotate_breaks(ax, cfg)
        ax.legend(fontsize=8)
    bottom.axhline(0.0, linewidth=0.8, alpha=0.6)
    fig.suptitle("Matching rate and observability gap on the same event timeline")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / "fig_phase2_07_matching_rate_and_gap.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def load_aoi_detections(cfg: dict[str, Any], aoi_key: str) -> pd.DataFrame:
    """Detections inside the primary analysis sample for one AOI."""
    processed = resolve_path(cfg["paths"]["processed_dir"])
    interim = resolve_path(cfg["paths"]["interim_dir"])
    metrics = pd.read_parquet(aoi_scoped_path(processed / "sar_scene_metrics.parquet", aoi_key, cfg))
    keep = set(metrics.loc[metrics["usable_for_primary_metrics"], "scene_id"])
    frame = pd.read_parquet(
        aoi_scoped_path(interim / "sar_detections_aoi.parquet", aoi_key, cfg)
    )
    frame = frame[frame["scene_id"].isin(keep)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def event_window(cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Acute + persistent event window for the event year."""
    phases = resolve_phases(cfg, int(cfg["analysis"]["event_year"]))
    return phases["acute_shock"][0], phases["persistent_disruption"][1]


def plot_length_distribution(
    cfg: dict[str, Any],
    aoi_key: str,
    figures_dir: Path,
    logger,
) -> bool:
    """Pre/post vessel-length distributions for one AOI (composition diagnostic)."""
    frame = load_aoi_detections(cfg, aoi_key)
    if frame.empty:
        logger.warning("length figure skipped for %s: no detections", aoi_key)
        return False
    start, end = event_window(cfg)
    frame["period"] = np.where(
        (frame["timestamp"] >= start) & (frame["timestamp"] <= end),
        "post_event_2026",
        "pre_event_2026",
    )
    bins = np.arange(0, 320, 20)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for period, group in frame.groupby("period"):
        ax.hist(
            pd.to_numeric(group["length_m"], errors="coerce").dropna(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"{period} (n={len(group)})",
        )
    ax.set_xlabel("estimated vessel length (m)")
    ax.set_ylabel("density within period")
    ax.set_title(f"{aoi_key}: vessel-length distribution before vs during the event")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / f"fig_phase3_length_distribution_{aoi_key}.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def plot_spatial_distribution(
    cfg: dict[str, Any],
    aoi_key: str,
    figures_dir: Path,
    logger,
) -> bool:
    """Pre/post spatial distribution of detections for one AOI."""
    frame = load_aoi_detections(cfg, aoi_key)
    if frame.empty:
        logger.warning("spatial figure skipped for %s: no detections", aoi_key)
        return False
    start, end = event_window(cfg)
    post = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    pre = frame[~((frame["timestamp"] >= start) & (frame["timestamp"] <= end))]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pre["lon"], pre["lat"], s=6, alpha=0.35, label=f"pre-event 2026 (n={len(pre)})")
    ax.scatter(post["lon"], post["lat"], s=18, alpha=0.8, marker="x",
               label=f"post-event 2026 (n={len(post)})")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{aoi_key}: detection locations before vs during the event")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    ax.set_aspect(1.0 / np.cos(np.radians(float(frame["lat"].mean()))))
    fig.tight_layout()
    path = figures_dir / f"fig_phase3_spatial_distribution_{aoi_key}.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def plot_standardized_effects(
    cfg: dict[str, Any],
    aoi_key: str,
    standardized: pd.DataFrame,
    figures_dir: Path,
    logger,
) -> bool:
    """Observed vs composition-standardized matching probability, per AOI."""
    subset = standardized[standardized["aoi_key"] == aoi_key].copy()
    if subset.empty:
        logger.warning("standardized figure skipped for %s: no rows", aoi_key)
        return False
    order = ["2024_reference", "2025_reference", "2026_baseline", "2026_acute", "2026_persistent"]
    subset["period"] = pd.Categorical(subset["period"], categories=order, ordered=True)
    subset = subset.sort_values("period")
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, subset["observed_matching_rate"], marker="o", label="observed matching rate")
    ax.plot(x, subset["standardized_matching_probability"], marker="s", linestyle="--",
            label="standardized (pre-event 2026 composition)")
    ax.errorbar(
        x,
        subset["standardized_matching_probability"],
        yerr=[
            subset["standardized_matching_probability"] - subset["standardized_ci_low"],
            subset["standardized_ci_high"] - subset["standardized_matching_probability"],
        ],
        fmt="none",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in subset["period"]], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("matching probability")
    ax.set_title(f"{aoi_key}: observed vs composition-standardized matching")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / f"fig_phase3_standardized_matching_{aoi_key}.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def plot_divergence_alignment(
    cfg: dict[str, Any],
    weekly: pd.DataFrame,
    alignment: pd.DataFrame,
    figures_dir: Path,
    logger,
) -> bool:
    """Anomaly onsets on one timeline, with each signal's own baseline threshold."""
    if weekly.empty or alignment.empty:
        logger.warning("alignment figure skipped: missing inputs")
        return False
    event_year = int(cfg["analysis"]["event_year"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for aoi_key in sorted(weekly["aoi_key"].unique()):
        for series_name in ("AIS_SAR_days|ALL", "SAR_ALL_detections"):
            subset = weekly[
                (weekly["aoi_key"] == aoi_key)
                & (weekly["series"] == series_name)
                & (weekly["iso_year"] == event_year)
            ].copy()
            if subset.empty:
                continue
            subset["week_start"] = pd.to_datetime(subset["week_start"])
            subset = subset[subset["week_start"] >= pd.Timestamp(f"{event_year}-01-01")]
            ax.plot(subset["week_start"], subset["log_anomaly"], marker="o", markersize=3,
                    label=f"{aoi_key}: {series_name.split('|')[0]}")
    annotate_breaks(ax, cfg)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_ylabel("log anomaly vs same ISO week (2024/2025)")
    ax.set_xlabel("week starting (UTC)")
    ax.set_title("AIS (SAR days) and SAR activity anomalies, event year 2026")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / "fig_phase3_divergence_alignment.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("07_make_figures", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 7 (Phase 2) - figures")

    aoi_keys = args.aois or list_aois(cfg)
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    tables = resolve_path(cfg["paths"]["tables_dir"])
    weekly_path = tables / "activity_anomaly_weekly.csv"
    weekly = pd.read_csv(weekly_path) if weekly_path.exists() else pd.DataFrame()
    if not weekly.empty:
        weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    else:
        logger.warning(
            "%s not found; run src/05_activity_anomaly.py first (SAR-only figures may still "
            "be produced if the SAR panels exist)",
            weekly_path.name,
        )

    produced: list[str] = []
    if plot_single_anomaly(weekly, cfg, aoi_keys, "AIS_all_days|ALL",
                           "fig_phase2_01_ais_all_days_anomaly.png",
                           "Weekly AIS presence anomaly (all days)", figures_dir, logger):
        produced.append("fig_phase2_01_ais_all_days_anomaly.png")
    if plot_single_anomaly(weekly, cfg, aoi_keys, "AIS_SAR_days|ALL",
                           "fig_phase2_02_ais_sar_days_anomaly.png",
                           "Weekly AIS presence anomaly (SAR acquisition days only)",
                           figures_dir, logger):
        produced.append("fig_phase2_02_ais_sar_days_anomaly.png")
    if plot_single_anomaly(weekly, cfg, aoi_keys, "SAR_ALL_detections",
                           "fig_phase2_03_sar_density_anomaly.png",
                           "Weekly SAR vessel-density anomaly", figures_dir, logger):
        produced.append("fig_phase2_03_sar_density_anomaly.png")
    if plot_pair(weekly, cfg, aoi_keys, "AIS_all_days|ALL", "SAR_ALL_detections",
                 "fig_phase2_04_ais_all_days_vs_sar.png",
                 "AIS (all days) vs SAR activity anomaly", figures_dir, logger):
        produced.append("fig_phase2_04_ais_all_days_vs_sar.png")
    if plot_pair(weekly, cfg, aoi_keys, "AIS_SAR_days|ALL", "SAR_ALL_detections",
                 "fig_phase2_05_ais_sar_days_vs_sar.png",
                 "AIS (SAR days only) vs SAR activity anomaly", figures_dir, logger):
        produced.append("fig_phase2_05_ais_sar_days_vs_sar.png")
    if plot_gap(weekly, cfg, aoi_keys, "fig_phase2_06_observability_gap.png", figures_dir, logger):
        produced.append("fig_phase2_06_observability_gap.png")
    if plot_matching_rate_and_gap(weekly, cfg, aoi_keys, figures_dir, logger):
        produced.append("fig_phase2_07_matching_rate_and_gap.png")

    # ---- Phase 3 identification figures ----------------------------------
    hormuz_aois = [
        key for key in aoi_keys
        if resolve_aoi(cfg, key)[1].get("role") != "technical_reference_corridor"
    ]
    for aoi_key in hormuz_aois:
        if plot_length_distribution(cfg, aoi_key, figures_dir, logger):
            produced.append(f"fig_phase3_length_distribution_{aoi_key}.png")
        if plot_spatial_distribution(cfg, aoi_key, figures_dir, logger):
            produced.append(f"fig_phase3_spatial_distribution_{aoi_key}.png")
    standardized_path = tables / "standardized_matching_effects.csv"
    if standardized_path.exists():
        standardized = pd.read_csv(standardized_path)
        for aoi_key in hormuz_aois:
            if plot_standardized_effects(cfg, aoi_key, standardized, figures_dir, logger):
                produced.append(f"fig_phase3_standardized_matching_{aoi_key}.png")
    alignment_path = tables / "divergence_alignment.csv"
    alignment = pd.read_csv(alignment_path) if alignment_path.exists() else pd.DataFrame()
    if plot_divergence_alignment(cfg, weekly, alignment, figures_dir, logger):
        produced.append("fig_phase3_divergence_alignment.png")

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "figure_manifest.json",
        {
            "figures_produced": produced,
            "aois": aoi_keys,
            "primary_aoi": primary_aoi_key(cfg),
            "annotated_breaks": [
                {"date": str(date.date()), "label": label} for date, label in break_markers(cfg)
            ],
        },
    )
    logger.info("produced %d figure(s): %s", len(produced), produced)
    return 0 if produced else 3


if __name__ == "__main__":
    raise SystemExit(main())
