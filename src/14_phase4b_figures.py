"""STEP 14 (Phase 4B) - publication-oriented figures.

matplotlib only, no seaborn, one figure per file, no fixed colour palette unless a
series must stay consistent with an earlier figure in the same figure.

Figures
-------
1  Red Sea SAR activity anomaly
2  Red Sea AIS activity metrics
3  Red Sea standardized matching probability
4  Panama SAR activity anomaly
5  Panama AIS presence hours vs distinct vessels
6  Panama presence-hours-per-vessel proxy
7  Panama standardized matching probability
8  Bab-2026 vs Hormuz same-calendar matching change
9  Bab-2026 vs Hormuz same-calendar SAR activity
10 cross-event: physical disruption vs observability disruption
11 event-role conceptual figure

Usage
-----
    python src/14_phase4b_figures.py
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
    add_calendar_fields,
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_path,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 4B figures")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def finish(fig, ax, path: Path, cfg: dict[str, Any], ylabel: str, title: str) -> None:
    """Apply shared styling and save one figure."""
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def weekly_matching_rate(scenes: pd.DataFrame) -> pd.DataFrame:
    """Pooled weekly matching rate from a scene table (comparable across AOIs)."""
    frame = scenes.copy()
    frame["date"] = pd.to_datetime(frame["scene_date_parsed"]).dt.normalize()
    frame = add_calendar_fields(frame, "date")
    weekly = (
        frame.groupby(["iso_year", "iso_week", "week_start"], as_index=False)
        .agg(n_matched=("n_matched", "sum"), n_unmatched=("n_unmatched", "sum"),
             n_scenes=("scene_id", "size"))
    )
    denominator = weekly["n_matched"] + weekly["n_unmatched"]
    weekly["matching_rate"] = weekly["n_matched"] / denominator.replace(0, np.nan)
    return weekly


def plot_weekly_series(
    weekly: pd.DataFrame,
    value_column: str,
    cfg: dict[str, Any],
    figures_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    phase_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    logger,
) -> bool:
    """One weekly series with phase boundaries marked."""
    if weekly.empty or value_column not in weekly:
        logger.warning("figure %s skipped: no %s column", filename, value_column)
        return False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weekly["week_start"], weekly[value_column], marker="o", markersize=3,
            label=value_column)
    for label, (start, end) in phase_windows.items():
        for boundary in (start, end):
            ax.axvline(boundary, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(label, xy=(start, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7, rotation=90, va="bottom", ha="right", alpha=0.8)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel("week starting (UTC)")
    finish(fig, ax, figures_dir / filename, cfg, ylabel, title)
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_phase_bars(
    phase_summary: pd.DataFrame,
    value_columns: Sequence[str],
    cfg: dict[str, Any],
    figures_dir: Path,
    filename: str,
    title: str,
    logger,
    orientation: str = "x",
) -> bool:
    """Phase-level comparison for one or more metrics."""
    if phase_summary.empty:
        logger.warning("figure %s skipped: empty phase summary", filename)
        return False
    phases = list(phase_summary["phase"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(phases))
    width = 0.8 / max(len(value_columns), 1)
    for index, column in enumerate(value_columns):
        if column not in phase_summary:
            continue
        ax.bar(positions + index * width, phase_summary[column].to_numpy(dtype=float),
               width=width, label=column)
    ax.set_xticks(positions + width * (len(value_columns) - 1) / 2)
    ax.set_xticklabels(phases, rotation=15, ha="right", fontsize=8)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    finish(fig, ax, figures_dir / filename, cfg, "value", title)
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_standardized(
    standardized: pd.DataFrame,
    cfg: dict[str, Any],
    figures_dir: Path,
    filename: str,
    title: str,
    logger,
) -> bool:
    """Observed vs standardized matching probability per period."""
    if standardized.empty:
        logger.warning("figure %s skipped: no standardized table", filename)
        return False
    frame = standardized.copy()
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(x, frame["observed_matching_rate"], marker="o", label="observed matching rate")
    ax.plot(x, frame["standardized_matching_probability"], marker="s", linestyle="--",
            label="standardized (pre-event composition)")
    ax.errorbar(
        x,
        frame["standardized_matching_probability"],
        yerr=[
            frame["standardized_matching_probability"] - frame["standardized_ci_low"],
            frame["standardized_ci_high"] - frame["standardized_matching_probability"],
        ],
        fmt="none",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.period}\n(n={int(row.n_detections)})" for row in frame.itertuples()],
                       fontsize=8)
    finish(fig, ax, figures_dir / filename, cfg, "matching probability", title)
    logger.info("wrote %s", figures_dir / filename)
    return True


def plot_cross_event(
    cross: pd.DataFrame,
    cfg: dict[str, Any],
    figures_dir: Path,
    logger,
) -> bool:
    """Two-dimensional plot: physical disruption vs observability disruption.

    x = historical-normalized SAR activity anomaly (Dimension A)
    y = composition-standardized matching change in percentage points (Dimension B)
    Cases are annotated, never ranked, and no composite score is drawn.
    """
    if cross.empty:
        logger.warning("cross-event figure skipped: empty table")
        return False
    frame = cross.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in frame.itertuples():
        x_value = row.SAR_activity_anomaly
        y_value = row.standardized_matching_change_pp
        if not np.isfinite(x_value):
            continue
        if np.isfinite(y_value):
            low = row.standardized_matching_CI_low
            high = row.standardized_matching_CI_high
            if np.isfinite(low) and np.isfinite(high):
                ax.errorbar(x_value, y_value, yerr=[[y_value - low], [high - y_value]],
                            fmt="o", capsize=4, markersize=7)
            else:
                ax.plot(x_value, y_value, "o", markersize=7)
        else:
            # descriptive cases have no standardized model: plot on the axis edge
            ax.plot(x_value, 0.0, "x", markersize=9)
            y_value = 0.0
        ax.annotate(row.event, xy=(x_value, y_value), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel(
        "physical activity anomaly\n"
        "(SAR weekly density, log anomaly vs same weeks in placebo years;\n"
        "negative = fewer detections per pass, positive = more, e.g. queueing)"
    )
    ax.set_ylabel("AIS observability disruption\n(composition-standardized matching change, pp)")
    ax.set_title("Physical disruption and AIS observability disruption are separate dimensions")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path = figures_dir / "fig_phase4b_10_cross_event_dimensions.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s (descriptive cases marked with x on the zero line)", path)
    return True


def plot_event_roles(
    roles: pd.DataFrame,
    cfg: dict[str, Any],
    figures_dir: Path,
    logger,
) -> bool:
    """Conceptual figure: the cases are not statistically equivalent replicas."""
    if roles.empty:
        logger.warning("event-role figure skipped: empty table")
        return False
    frame = roles[["event", "role", "disruption_type", "statistical_status", "inference_strength"]]
    fig, ax = plt.subplots(figsize=(12, 0.7 + 0.6 * len(frame)))
    ax.axis("off")
    table = ax.table(
        cellText=frame.to_numpy(),
        colLabels=["event", "case role", "disruption type", "statistical status", "inference strength"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title("Case roles in the Phase 4B matrix (not interchangeable replicas)")
    fig.tight_layout()
    path = figures_dir / "fig_phase4b_11_event_roles.png"
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)
    logger.info("wrote %s", path)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("14_phase4b_figures", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 14 (Phase 4B) - figures")

    tables = resolve_path(cfg["paths"]["tables_dir"])
    figures = resolve_path(cfg["paths"]["figures_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    interim = resolve_path(cfg["paths"]["interim_dir"])
    events = cfg["phase4b"]["events"]

    produced: list[str] = []
    for key, spec in events.items():
        sar_activity = tables / "phase4b_sar_activity.csv"
        sar_weekly_all = tables / "phase4b_sar_weekly.csv"
        standardized_path = tables / f"{key}_standardized_effects.csv"
        ais_path = tables / "phase4b_ais_phase_summary.csv"

        weekly = pd.read_csv(sar_weekly_all) if sar_weekly_all.exists() else pd.DataFrame()
        event_weekly = weekly[weekly["event"] == key].copy() if len(weekly) else weekly
        if len(event_weekly):
            event_weekly["week_start"] = pd.to_datetime(event_weekly["week_start"])
            event_weekly = event_weekly.sort_values("week_start")
        windows = {
            phase: (
                pd.Timestamp(spec["event_onset"]).normalize(),
                pd.Timestamp(spec["event_onset"]).normalize() + pd.Timedelta(days=1),
            )
            for phase in spec["phases"]
        }

        # --- per-event figures -------------------------------------------------
        if key == "redsea_2023":
            if plot_weekly_series(
                event_weekly, "sar_activity_anomaly", cfg, figures, 
                "fig_phase4b_01_redsea_sar_anomaly.png",
                "Red Sea / Bab el-Mandeb 2023-24: weekly SAR activity anomaly",
                "SAR log anomaly vs same ISO weeks (2021/2022)", windows, logger,
            ):
                produced.append("fig_phase4b_01_redsea_sar_anomaly.png")
            if not event_weekly.empty:
                event_weekly["sar_density_plot"] = event_weekly["sar_density_median"]
                if plot_weekly_series(
                    event_weekly, "sar_density_plot", cfg, figures,
                    "fig_phase4b_01b_redsea_sar_density.png",
                    "Red Sea: weekly SAR density in the event year",
                    "detections per 1000 km2 per acquisition", windows, logger,
                ):
                    produced.append("fig_phase4b_01b_redsea_sar_density.png")
            if ais_path.exists():
                ais = pd.read_csv(ais_path)
                redsea = ais[(ais["event"] == key)]
                if plot_phase_bars(
                    redsea, ["ais_presence_hours_daily_mean", "ais_distinct_vessels_daily_mean"],
                    cfg, figures, "fig_phase4b_02_redsea_ais_metrics.png",
                    "Red Sea: AIS activity metrics by phase and year role", logger,
                ):
                    produced.append("fig_phase4b_02_redsea_ais_metrics.png")
            if standardized_path.exists() and plot_standardized(
                pd.read_csv(standardized_path), cfg, figures,
                "fig_phase4b_03_redsea_standardized_matching.png",
                "Red Sea: observed vs composition-standardized matching probability", logger,
            ):
                produced.append("fig_phase4b_03_redsea_standardized_matching.png")

        if key == "panama_2023":
            if plot_weekly_series(
                event_weekly, "sar_activity_anomaly", cfg, figures,
                "fig_phase4b_04_panama_sar_anomaly.png",
                "Panama Canal 2023-24: weekly SAR activity anomaly",
                "SAR log anomaly vs same ISO weeks (2021/2022)", windows, logger,
            ):
                produced.append("fig_phase4b_04_panama_sar_anomaly.png")
            if ais_path.exists():
                ais = pd.read_csv(ais_path)
                panama = ais[ais["event"] == key]
                if plot_phase_bars(
                    panama, ["ais_presence_hours_daily_mean", "ais_distinct_vessels_daily_mean"],
                    cfg, figures, "fig_phase4b_05_panama_ais_hours_vs_vessels.png",
                    "Panama: AIS presence hours vs distinct vessels by phase", logger,
                ):
                    produced.append("fig_phase4b_05_panama_ais_hours_vs_vessels.png")
                if plot_phase_bars(
                    panama, ["ais_presence_hours_per_vessel"], cfg, figures,
                    "fig_phase4b_06_panama_dwell_proxy.png",
                    "Panama: presence hours per distinct vessel (dwell proxy)", logger,
                ):
                    produced.append("fig_phase4b_06_panama_dwell_proxy.png")
            if standardized_path.exists() and plot_standardized(
                pd.read_csv(standardized_path), cfg, figures,
                "fig_phase4b_07_panama_standardized_matching.png",
                "Panama: observed vs composition-standardized matching probability", logger,
            ):
                produced.append("fig_phase4b_07_panama_standardized_matching.png")

    # --- Bab-2026 vs Hormuz on the same calendar -----------------------------
    bab_scenes = pd.read_parquet(
        aoi_scoped_path(processed / "sar_scene_metrics.parquet", "bab_el_mandeb_core", cfg,
                        tag="w2024_2026")
    )
    hormuz_scenes = pd.read_parquet(processed / "sar_scene_metrics.parquet")
    bab_weekly = weekly_matching_rate(bab_scenes)
    hormuz_weekly = weekly_matching_rate(hormuz_scenes)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for frame, label in ((hormuz_weekly, "hormuz_core"), (bab_weekly, "bab_el_mandeb_core")):
        subset = frame[(frame["week_start"] >= "2026-01-01") & (frame["week_start"] <= "2026-06-01")]
        ax.plot(subset["week_start"], subset["matching_rate"], marker="o", markersize=3, label=label)
    ax.axvline(pd.Timestamp("2026-03-01"), linestyle="--", linewidth=1.0, alpha=0.7)
    ax.annotate("2026-03-01\nv3 -> v4 + onset", xy=(pd.Timestamp("2026-03-01"), 0.02),
                xycoords=("data", "axes fraction"), fontsize=7, va="bottom")
    finish(fig, ax, figures / "fig_phase4b_08_bab_vs_hormuz_matching.png", cfg,
           "weekly pooled matching rate",
           "Bab el-Mandeb vs Hormuz 2026: same calendar, same pipeline boundary")
    produced.append("fig_phase4b_08_bab_vs_hormuz_matching.png")

    weekly_all = pd.read_csv(tables / "phase4b_sar_weekly.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for event_key in ("bab_mandeb_2026", "redsea_2023"):
        subset = weekly_all[weekly_all["event"] == event_key]
        if subset.empty:
            continue
        subset = subset[(subset["week_start"] >= "2026-01-01") & (subset["week_start"] <= "2026-06-01")]
        ax.plot(pd.to_datetime(subset["week_start"]), subset["sar_density_median"],
                marker="o", markersize=3, label=event_key)
    hormuz_weekly_frozen = pd.read_parquet(processed / "sar_weekly.parquet")
    hormuz_weekly_frozen = hormuz_weekly_frozen[
        (pd.to_datetime(hormuz_weekly_frozen["week_start"]) >= "2026-01-01")
        & (pd.to_datetime(hormuz_weekly_frozen["week_start"]) <= "2026-06-01")
    ]
    ax.plot(pd.to_datetime(hormuz_weekly_frozen["week_start"]),
            hormuz_weekly_frozen["sar_density_median"], marker="s", markersize=3,
            linestyle="--", label="hormuz_core (frozen)")
    ax.axvline(pd.Timestamp("2026-03-01"), linestyle="--", linewidth=1.0, alpha=0.7)
    finish(fig, ax, figures / "fig_phase4b_09_bab_vs_hormuz_sar_activity.png", cfg,
           "SAR density (detections per 1000 km2 per acquisition)",
           "Bab el-Mandeb vs Hormuz 2026: same calendar SAR activity")
    produced.append("fig_phase4b_09_bab_vs_hormuz_sar_activity.png")

    # --- cross-event and role figures ----------------------------------------
    cross_path = tables / "cross_event_final.csv"
    roles_path = tables / "event_role_summary.csv"
    if cross_path.exists() and plot_cross_event(pd.read_csv(cross_path), cfg, figures, logger):
        produced.append("fig_phase4b_10_cross_event_dimensions.png")
    if roles_path.exists() and plot_event_roles(pd.read_csv(roles_path), cfg, figures, logger):
        produced.append("fig_phase4b_11_event_roles.png")

    logger.info("produced %d figure(s): %s", len(produced), produced)
    return 0 if produced else 3


if __name__ == "__main__":
    raise SystemExit(main())
