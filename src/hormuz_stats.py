"""Bootstrap and ratio estimators shared by the robustness tables and Phase 3.

Design rules
------------
* The resampling unit is the **scene**, never the detection. Restricting AIS or
  SAR to days/weeks is a separate concern; when a rate is estimated from
  scenes, the uncertainty must reflect which scenes happened to be acquired.
* Every function is deterministic given a seed, and takes plain arrays so it can
  be unit-tested without any data files.
* Small samples are reported as such rather than silently bootstrapped into a
  spuriously narrow interval (see ``min_scenes_for_ci``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    """A point estimate with an optional uncertainty interval."""

    estimate: float
    low: float
    high: float
    n_units: int
    n_observations: int
    ci_computed: bool
    method: str


def _percentile_ci(draws: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval from bootstrap draws."""
    alpha = (1.0 - level) / 2.0
    return float(np.nanpercentile(draws, 100 * alpha)), float(np.nanpercentile(draws, 100 * (1 - alpha)))


def pooled_rate(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Pooled ratio (sum of numerators / sum of denominators), NaN if undefined."""
    num = float(np.nansum(numerator))
    den = float(np.nansum(denominator))
    return num / den if den > 0 else float("nan")


def bootstrap_pooled_rate(
    numerator: Sequence[float] | np.ndarray,
    denominator: Sequence[float] | np.ndarray,
    iterations: int,
    level: float,
    seed: int,
    min_units_for_ci: int = 3,
    cluster: Sequence | None = None,
) -> Interval:
    """Scene-level bootstrap CI for a pooled rate.

    Parameters
    ----------
    numerator, denominator:
        Per-scene counts (e.g. matched and matched+unmatched detections).
    iterations, level, seed:
        Bootstrap size, CI level, deterministic seed.
    min_units_for_ci:
        Below this many scenes the interval is not computed and
        ``ci_computed`` is False (the caller must flag the cell).
    cluster:
        Optional cluster label per scene (kept for future use); when supplied the
        resampling is over unique clusters rather than rows.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    estimate = pooled_rate(num, den)
    n_units = len(num)
    n_observations = int(np.nansum(den))
    if n_units < min_units_for_ci or n_observations == 0 or (den <= 0).all():
        return Interval(estimate, float("nan"), float("nan"), n_units, n_observations, False, "not_computed")

    rng = np.random.default_rng(seed)
    if cluster is not None:
        labels = np.asarray(cluster)
        unique = pd.unique(labels)
        index = {label: np.flatnonzero(labels == label) for label in unique}
        draws = np.empty(iterations, dtype=float)
        for draw in range(iterations):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([index[label] for label in chosen])
            draws[draw] = pooled_rate(num[rows], den[rows])
    else:
        positions = rng.integers(0, n_units, size=(iterations, n_units))
        draws = num[positions].sum(axis=1) / np.maximum(den[positions].sum(axis=1), 1e-12)
    low, high = _percentile_ci(draws, level)
    return Interval(estimate, low, high, n_units, n_observations, True, "scene_bootstrap_percentile")


def bootstrap_rate_difference(
    num_pre: Sequence[float],
    den_pre: Sequence[float],
    num_post: Sequence[float],
    den_post: Sequence[float],
    iterations: int,
    level: float,
    seed: int,
    min_units_for_ci: int = 3,
) -> Interval:
    """Bootstrap CI for ``rate_post - rate_pre`` (independent resampling).

    The two periods share no scenes, so they are resampled independently; the
    interval therefore includes both sampling and scene-realisation uncertainty.
    """
    a_num = np.asarray(num_pre, dtype=float)
    a_den = np.asarray(den_pre, dtype=float)
    b_num = np.asarray(num_post, dtype=float)
    b_den = np.asarray(den_post, dtype=float)
    estimate = pooled_rate(b_num, b_den) - pooled_rate(a_num, a_den)
    n_units = min(len(a_num), len(b_num))
    n_observations = int(np.nansum(a_den) + np.nansum(b_den))
    if len(a_num) < min_units_for_ci or len(b_num) < min_units_for_ci:
        return Interval(estimate, float("nan"), float("nan"), n_units, n_observations, False, "not_computed")

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a_num), size=(iterations, len(a_num)))
    idx_b = rng.integers(0, len(b_num), size=(iterations, len(b_num)))
    rate_a = a_num[idx_a].sum(axis=1) / np.maximum(a_den[idx_a].sum(axis=1), 1e-12)
    rate_b = b_num[idx_b].sum(axis=1) / np.maximum(b_den[idx_b].sum(axis=1), 1e-12)
    low, high = _percentile_ci(rate_b - rate_a, level)
    return Interval(estimate, low, high, n_units, n_observations, True, "scene_bootstrap_percentile")


def bootstrap_log_ratio(
    current: Sequence[float],
    reference: Sequence[float],
    epsilon: float,
    iterations: int,
    level: float,
    seed: int,
    min_units_for_ci: int = 3,
) -> Interval:
    """Bootstrap CI for ``log((current + eps) / (reference + eps))``.

    Used for activity anomalies, where both series are weekly values in unknown
    but internally consistent units. ``current`` and ``reference`` are resampled
    independently (different calendar years).
    """
    a = np.asarray(current, dtype=float)
    b = np.asarray(reference, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), min(len(a), len(b)), 0, False, "not_computed")
    estimate = float(np.log((np.median(a) + epsilon) / (np.median(b) + epsilon)))
    if len(a) < min_units_for_ci or len(b) < min_units_for_ci:
        return Interval(estimate, float("nan"), float("nan"), min(len(a), len(b)), len(a) + len(b), False, "not_computed")
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(iterations, len(a)))
    idx_b = rng.integers(0, len(b), size=(iterations, len(b)))
    draws = np.log(
        (np.median(a[idx_a], axis=1) + epsilon) / (np.median(b[idx_b], axis=1) + epsilon)
    )
    low, high = _percentile_ci(draws, level)
    return Interval(estimate, low, high, min(len(a), len(b)), len(a) + len(b), True, "percentile_bootstrap")


def add_interval_columns(
    frame: pd.DataFrame,
    interval: Interval | None = None,
    prefix: str = "",
    *,
    estimate_column: str = "estimate",
    low_column: str = "ci_low",
    high_column: str = "ci_high",
) -> pd.DataFrame:
    """Copy an :class:`Interval` into explicit columns of a one-row frame.

    ``ci_computed`` is stored explicitly so that "no interval available" is never
    confused with "interval is exactly zero".
    """
    out = frame.copy()
    if interval is None:
        out[f"{prefix}{estimate_column}"] = np.nan
        out[f"{prefix}{low_column}"] = np.nan
        out[f"{prefix}{high_column}"] = np.nan
        out[f"{prefix}ci_computed"] = False
        return out
    out[f"{prefix}{estimate_column}"] = interval.estimate
    out[f"{prefix}{low_column}"] = interval.low
    out[f"{prefix}{high_column}"] = interval.high
    out[f"{prefix}ci_computed"] = interval.ci_computed
    return out


__all__ = [
    "Interval",
    "add_interval_columns",
    "bootstrap_log_ratio",
    "bootstrap_pooled_rate",
    "bootstrap_rate_difference",
    "pooled_rate",
]
