"""STEP 6 (Phase 3) - formal matching models and composition standardization.

Why this step exists
--------------------
The raw matching-rate decline mixes two things: a change in *who* is in the AOI
(vessel size and spatial composition) and a change in *how likely* a given kind
of vessel is to carry a confirmed AIS match. Phase 3 has to separate them, so the
primary quantity reported here is not a p-value but a **composition-standardized
matching probability**: predicted probabilities evaluated under the pre-event
2026 covariate distribution.

Model
-----
Detection-level logistic regression with standard errors **clustered by SAR
scene**, because detections inside one acquisition are not independent.

    matched_flag ~ C(period) + C(length_bin) + C(cell)          (primary)
    matched_flag ~ C(period) + C(length_bin)                    (length only)
    matched ~ C(period) + C(length_bin)      (scene-level grouped robustness)

``period`` identifies year x event phase with the five levels
``2024_reference``, ``2025_reference``, ``2026_baseline``, ``2026_acute`` and
``2026_persistent``. A literal ``+ C(year) + C(phase)`` parameterisation is rank
deficient (the 2026 year dummy is the sum of the three 2026 phase dummies), so the
combined factor is used and the fact is recorded in the coefficient table.

``matching_score`` and MMSI presence are **deliberately excluded** as predictors:
both are produced by the matching process itself and would be endogenous to the
outcome definition.

Standardization
---------------
For every period the script predicts each observation's matching probability,
then reweights the covariate combinations (length bin x spatial cell) to the
pre-event 2026 baseline distribution. Confidence intervals come from coefficient
draws out of the cluster-robust covariance matrix.

Outputs
-------
``outputs/tables/formal_matching_models.csv``
``outputs/tables/standardized_matching_effects.csv``

Usage
-----
    python src/06_statistical_tests.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hormuz_common import (  # noqa: E402
    aoi_scoped_path,
    banner,
    ensure_dirs,
    get_logger,
    in_window,
    list_aois,
    load_config,
    primary_aoi_key,
    resolve_aoi,
    resolve_path,
    resolve_phases,
    write_json,
)

PERIOD_LABELS = (
    "2024_reference",
    "2025_reference",
    "2026_baseline",
    "2026_acute",
    "2026_persistent",
)
PHASE_TO_PERIOD = {
    "baseline": "2026_baseline",
    "acute_shock": "2026_acute",
    "persistent_disruption": "2026_persistent",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Formal matching models (Phase 3)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--aois", nargs="*", default=None, help="AOI keys (default: Hormuz AOIs)")
    return parser.parse_args(argv)


def default_aois(cfg: dict[str, Any]) -> list[str]:
    """Hormuz AOIs only; the reference corridors are SAR-only fixtures."""
    return [
        key for key in list_aois(cfg)
        if resolve_aoi(cfg, key)[1].get("role") != "technical_reference_corridor"
    ]


def cell_labels(lat: pd.Series, lon: pd.Series, degrees: float) -> pd.Series:
    """Fixed geographic cell label from a pre-specified grid size."""
    lat_idx = np.floor(pd.to_numeric(lat, errors="coerce") / degrees).astype("Int64")
    lon_idx = np.floor(pd.to_numeric(lon, errors="coerce") / degrees).astype("Int64")
    return lat_idx.astype("string") + "_" + lon_idx.astype("string")


def prepare_model_frame(
    cfg: dict[str, Any],
    aoi_key: str,
    logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the detection-level modelling frame and its diagnostics."""
    metrics = pd.read_parquet(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["processed_dir"]) / "sar_scene_metrics.parquet",
            aoi_key,
            cfg,
        )
    )
    keep = set(metrics.loc[metrics["usable_for_primary_metrics"], "scene_id"])
    detections = pd.read_parquet(
        aoi_scoped_path(
            resolve_path(cfg["paths"]["interim_dir"]) / "sar_detections_aoi.parquet",
            aoi_key,
            cfg,
        )
    )
    frame = detections[detections["scene_id"].isin(keep)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    # Outcome: only detections with a known match label can be modelled.
    n_before = len(frame)
    frame = frame[frame["matching_rate"].notna() if "matching_rate" in frame.columns else slice(None)]
    frame = frame[frame["matched_flag"].notna() | frame["unmatched_flag"].notna()].copy()
    n_dropped_unknown = n_before - len(frame)
    frame["matched_flag"] = frame["matched_flag"].astype(bool).astype(int)

    # Vessel length: fixed bins from the config.
    bins = [float(value) for value in cfg["identification"]["length_bins_m"]]
    length = pd.to_numeric(frame["length_m"], errors="coerce")
    frame["length_bin"] = pd.cut(length, bins=bins, right=False).astype(str)
    n_missing_length = int(length.isna().sum())
    frame = frame[length.notna()].copy()

    # Space: fixed cells, rare cells pooled so the model stays estimable.
    degrees = float(cfg["identification"]["spatial_cell_degrees"])
    frame["cell_raw"] = cell_labels(frame["lat"], frame["lon"], degrees)
    min_cell = int(cfg["identification"]["min_detections_per_cell"])
    counts = frame["cell_raw"].value_counts()
    keep_cells = set(counts[counts >= min_cell].index)
    frame["cell"] = np.where(frame["cell_raw"].isin(keep_cells), frame["cell_raw"], "cell_other")

    # Period: year x phase, using the same calendar windows for every year.
    period = pd.Series("outside_window", index=frame.index, dtype=object)
    for year in cfg["analysis"]["years"]:
        for phase, window in resolve_phases(cfg, year).items():
            mask = in_window(frame["timestamp"], window)
            if year == int(cfg["analysis"]["event_year"]):
                period.loc[mask] = PHASE_TO_PERIOD[phase]
            elif int(year) in [int(value) for value in cfg["analysis"]["reference_years"]]:
                period.loc[mask] = f"{int(year)}_reference"
    dropped_outside = int((period == "outside_window").sum())
    frame["period"] = period
    frame = frame[frame["period"] != "outside_window"].copy()
    frame["period"] = pd.Categorical(frame["period"], categories=list(PERIOD_LABELS))
    frame["year"] = frame["timestamp"].dt.year

    diagnostics = {
        "aoi_key": aoi_key,
        "n_detections_usable_scenes": int(n_before),
        "n_dropped_unknown_match_label": int(n_dropped_unknown),
        "n_dropped_missing_length": n_missing_length,
        "n_dropped_outside_calendar_windows": dropped_outside,
        "n_model_rows": int(len(frame)),
        "n_scenes": int(frame["scene_id"].nunique()),
        "n_cells_kept": int(len(keep_cells)),
        "n_cells_pooled_to_cell_other": int(len(counts) - len(keep_cells)),
        "min_detections_per_cell": min_cell,
        "spatial_cell_degrees": degrees,
        "period_counts": frame["period"].value_counts().to_dict(),
        "period_scene_counts": frame.groupby("period", observed=True)["scene_id"].nunique().to_dict(),
    }
    logger.info(
        "AOI %s model frame: %d detections, %d scenes, %d cells (+%d pooled), periods=%s",
        aoi_key,
        len(frame),
        diagnostics["n_scenes"],
        diagnostics["n_cells_kept"],
        diagnostics["n_cells_pooled_to_cell_other"],
        diagnostics["period_counts"],
    )
    return frame, diagnostics


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def fit_detection_logit(
    frame: pd.DataFrame,
    formula: str,
    cluster_variable: str,
    max_iterations: int,
    logger,
) -> tuple[Any, Any]:
    """Fit a logistic GLM with cluster-robust covariance.

    Returns ``(result, design_info)``; ``design_info`` is needed later to build
    prediction matrices for the standardization step.
    """
    import patsy
    import statsmodels.api as sm

    # dmatrices handles the formula's left-hand side; the outcome column comes
    # from the formula so the two can never drift apart.
    y_matrix, design = patsy.dmatrices(formula, frame, return_type="dataframe")
    y = np.asarray(y_matrix, dtype=float).ravel()
    groups = frame[cluster_variable].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, design, family=sm.families.Binomial())
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
            maxiter=max_iterations,
        )
    logger.info(
        "fitted %s: n=%d, clusters=%d, converged=%s",
        formula, int(result.nobs), int(groups.nunique()), bool(result.converged),
    )
    return result, design.design_info


def fit_scene_level_logit(
    frame: pd.DataFrame,
    formula: str,
    max_iterations: int,
    logger,
) -> Any:
    """Scene-level grouped binomial robustness model.

    Each scene contributes one observation with ``n_total`` trials, so within-scene
    correlation is absorbed by construction rather than by a sandwich estimator.
    """
    import patsy
    import statsmodels.api as sm

    scene_summary = (
        frame.groupby("scene_id", as_index=False)
        .agg(
            n_matched=("matched_flag", "sum"),
            n_total=("matched_flag", "size"),
            period=("period", lambda values: values.mode().iloc[0]),
            length_bin=("length_bin", lambda values: values.mode().iloc[0]),
        )
    )
    scene_summary["matched_share"] = scene_summary["n_matched"] / scene_summary["n_total"]
    y_matrix, design = patsy.dmatrices(formula, scene_summary, return_type="dataframe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            np.asarray(y_matrix, dtype=float).ravel(),
            design,
            family=sm.families.Binomial(),
            freq_weights=scene_summary["n_total"].to_numpy(dtype=float),
        )
        result = model.fit(maxiter=max_iterations)
    logger.info(
        "fitted scene-level %s: scenes=%d, converged=%s",
        formula, len(scene_summary), bool(result.converged),
    )
    return result


def coefficient_table(
    result: Any,
    aoi_key: str,
    model_name: str,
    formula: str,
    n_clusters: int,
    note: str,
) -> pd.DataFrame:
    """Coefficient table with log-odds, odds ratios and confidence intervals."""
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    conf = np.asarray(result.conf_int(), dtype=float)
    return pd.DataFrame(
        {
            "aoi_key": aoi_key,
            "model": model_name,
            "formula": formula,
            "term": list(result.params.index),
            "log_odds": params,
            "std_error": bse,
            "z_value": params / np.where(bse > 0, bse, np.nan),
            "p_value": pvalues,
            "log_odds_ci_low": conf[:, 0],
            "log_odds_ci_high": conf[:, 1],
            "odds_ratio": np.exp(params),
            "odds_ratio_ci_low": np.exp(conf[:, 0]),
            "odds_ratio_ci_high": np.exp(conf[:, 1]),
            "n_observations": float(result.nobs),
            "n_clusters": int(n_clusters),
            "converged": bool(result.converged),
            "note": note,
        }
    )


def standardized_probabilities(
    result: Any,
    design_info: Any,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    logger,
) -> pd.DataFrame:
    """Composition-standardized matching probabilities by period.

    Standardization holds the covariate mix fixed at the pre-event 2026 baseline:
    every (length bin x spatial cell) combination from the baseline period
    receives its baseline weight, while the period factor is set to each level in
    turn. The interval comes from draws out of the cluster-robust covariance.
    """
    import patsy

    baseline_period = PHASE_TO_PERIOD[str(cfg["identification"]["standardize_to_phase"])]
    baseline = frame[frame["period"] == baseline_period]
    if baseline.empty:
        logger.warning("no baseline rows for standardization; skipping")
        return pd.DataFrame()
    combinations = (
        baseline.groupby(["length_bin", "cell"], observed=True)
        .size()
        .rename("weight")
        .reset_index()
    )
    combinations["weight"] = combinations["weight"] / combinations["weight"].sum()

    draws = int(cfg["identification"].get("simulation_draws", 1000))
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
    covariance = np.asarray(result.cov_params(), dtype=float)
    params = np.asarray(result.params, dtype=float)
    beta_draws = rng.multivariate_normal(params, covariance, size=draws)

    records: list[dict[str, Any]] = []
    for period in PERIOD_LABELS:
        new_data = combinations[["length_bin", "cell"]].copy()
        new_data["period"] = period
        design = patsy.build_design_matrices([design_info], new_data, return_type="dataframe")[0]
        design_values = np.asarray(design, dtype=float)
        weights = combinations["weight"].to_numpy(dtype=float)
        point = 1.0 / (1.0 + np.exp(-(design_values @ params)))
        standardized = float(np.sum(weights * point))
        linear = design_values @ beta_draws.T
        predicted = 1.0 / (1.0 + np.exp(-linear))
        simulated = predicted.T @ weights
        observed_subset = frame[frame["period"] == period]
        records.append(
            {
                "period": period,
                "standardized_matching_probability": standardized,
                "standardized_ci_low": float(np.percentile(simulated, 2.5)),
                "standardized_ci_high": float(np.percentile(simulated, 97.5)),
                "observed_matching_rate": float(observed_subset["matched_flag"].mean())
                if len(observed_subset) else np.nan,
                "n_detections": int(len(observed_subset)),
                "n_scenes": int(observed_subset["scene_id"].nunique()),
                "median_length_m": float(observed_subset["length_m"].median())
                if len(observed_subset) else np.nan,
                "n_standardization_combinations": int(len(combinations)),
                "standardization_baseline": f"{baseline_period}",
                "coefficient_draws": draws,
            }
        )
    out = pd.DataFrame(records)
    reference = out.loc[out["period"] == baseline_period]
    if len(reference) == 1:
        out["standardized_difference_vs_baseline_pp"] = (
            100.0 * (out["standardized_matching_probability"] - float(reference["standardized_matching_probability"].iloc[0]))
        )
        out["observed_difference_vs_baseline_pp"] = (
            100.0 * (out["observed_matching_rate"] - float(reference["observed_matching_rate"].iloc[0]))
        )
    # interval for the difference: rebuild from the same coefficient draws
    baseline_design = patsy.build_design_matrices(
        [design_info],
        combinations[["length_bin", "cell"]].assign(period=baseline_period),
        return_type="dataframe",
    )[0]
    baseline_linear = np.asarray(baseline_design, dtype=float) @ beta_draws.T
    baseline_simulated = (1.0 / (1.0 + np.exp(-baseline_linear))).T @ combinations["weight"].to_numpy()
    difference_lows: list[float] = []
    difference_highs: list[float] = []
    for period in PERIOD_LABELS:
        new_data = combinations[["length_bin", "cell"]].copy()
        new_data["period"] = period
        design = patsy.build_design_matrices([design_info], new_data, return_type="dataframe")[0]
        simulated = (1.0 / (1.0 + np.exp(-(np.asarray(design, dtype=float) @ beta_draws.T)))).T @ combinations["weight"].to_numpy()
        diff = 100.0 * (simulated - baseline_simulated)
        difference_lows.append(float(np.percentile(diff, 2.5)))
        difference_highs.append(float(np.percentile(diff, 97.5)))
    out["standardized_difference_ci_low_pp"] = difference_lows
    out["standardized_difference_ci_high_pp"] = difference_highs
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = get_logger("06_statistical_tests", cfg)
    ensure_dirs(cfg)
    banner(logger, "STEP 6 (Phase 3) - matching models and composition standardization")

    aoi_keys = args.aois or default_aois(cfg)
    tables = resolve_path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    models_cfg = cfg["models"]
    primary_formula = str(models_cfg["primary_formula"])
    length_only_formula = str(models_cfg["length_only_formula"])
    grouped_formula = str(models_cfg["grouped_formula"])
    cluster_variable = str(models_cfg["cluster_variable"])
    max_iterations = int(models_cfg["max_iterations"])

    coefficient_frames: list[pd.DataFrame] = []
    standardized_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}

    for aoi_key in aoi_keys:
        frame, frame_diagnostics = prepare_model_frame(cfg, aoi_key, logger)
        diagnostics[aoi_key] = frame_diagnostics
        if frame.empty:
            logger.warning("AOI %s has no model rows; skipping", aoi_key)
            continue
        n_clusters = int(frame[cluster_variable].nunique())

        result, design_info = fit_detection_logit(
            frame, primary_formula, cluster_variable, max_iterations, logger
        )
        coefficient_frames.append(
            coefficient_table(
                result, aoi_key, "detection_logit_cluster_scene",
                primary_formula, n_clusters,
                note=(
                    "period identifies year x phase; a literal + C(year) is rank deficient "
                    "because the 2026 year dummy equals the sum of the 2026 phase dummies. "
                    "matching_score and MMSI presence are excluded as endogenous."
                ),
            )
        )
        standardized = standardized_probabilities(result, design_info, frame, cfg, logger)
        if not standardized.empty:
            standardized["aoi_key"] = aoi_key
            standardized["model"] = "detection_logit_cluster_scene"
            standardized_frames.append(standardized)

        length_only, _ = fit_detection_logit(
            frame, length_only_formula, cluster_variable, max_iterations, logger
        )
        coefficient_frames.append(
            coefficient_table(
                length_only, aoi_key, "detection_logit_length_only",
                length_only_formula, n_clusters,
                note="no spatial fixed effects; length composition only",
            )
        )

        grouped = fit_scene_level_logit(frame, grouped_formula, max_iterations, logger)
        coefficient_frames.append(
            coefficient_table(
                grouped, aoi_key, "scene_level_grouped_binomial",
                grouped_formula, n_clusters,
                note="one observation per scene (n_total trials); within-scene correlation "
                     "absorbed by construction; length bin fixed at the scene mode",
            )
        )

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(tables / "formal_matching_models.csv", index=False)
    logger.info("wrote %s (%d rows)", tables / "formal_matching_models.csv", len(coefficients))

    standardized_all = pd.concat(standardized_frames, ignore_index=True)
    standardized_all.to_csv(tables / "standardized_matching_effects.csv", index=False)
    logger.info(
        "wrote %s (%d rows)", tables / "standardized_matching_effects.csv", len(standardized_all)
    )

    headline = standardized_all[
        standardized_all["period"].isin(["2026_baseline", "2026_acute", "2026_persistent"])
    ][["aoi_key", "period", "n_detections", "observed_matching_rate",
       "standardized_matching_probability", "standardized_ci_low", "standardized_ci_high",
       "observed_difference_vs_baseline_pp", "standardized_difference_vs_baseline_pp",
       "standardized_difference_ci_low_pp", "standardized_difference_ci_high_pp"]]
    logger.info("standardized 2026 matching probabilities:\n%s", headline.round(3).to_string(index=False))

    write_json(
        resolve_path(cfg["paths"]["diagnostics_dir"]) / "matching_models_summary.json",
        {
            "primary_formula": primary_formula,
            "length_only_formula": length_only_formula,
            "grouped_formula": grouped_formula,
            "cluster_variable": cluster_variable,
            "standardize_to_phase": cfg["identification"]["standardize_to_phase"],
            "spatial_cell_degrees": float(cfg["identification"]["spatial_cell_degrees"]),
            "min_detections_per_cell": int(cfg["identification"]["min_detections_per_cell"]),
            "excluded_predictors": ["matching_score", "mmsi_present"],
            "excluded_predictor_reason": (
                "both are produced by the matching process and are endogenous to the outcome"
            ),
            "frame_diagnostics": diagnostics,
            "standardized_2026": headline.to_dict(orient="records"),
            "aois": aoi_keys,
            "primary_aoi": primary_aoi_key(cfg),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
