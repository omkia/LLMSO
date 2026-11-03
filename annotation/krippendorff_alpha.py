"""Krippendorff's alpha for the annotation reliability file.

The manuscript reports alpha = 0.84 for the multidimensional ECI and
alpha = 0.79 for the ICI. It does not state the difference function.
``level`` is therefore an argument: nominal, ordinal, or interval.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def krippendorff_alpha(reliability_data: np.ndarray, level: str = "interval") -> float:
    """Alpha for a matrix shaped (items, raters), with NaN for missing."""
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be items by raters")
    coincidence, values = _coincidence(data)
    if coincidence is None or len(values) < 2:
        return float("nan")
    totals = coincidence.sum(axis=1)
    n = float(coincidence.sum())
    if n < 2:
        return float("nan")
    delta = _delta(values, totals, level)
    observed = float(np.sum(coincidence * delta))
    expected = float(np.sum(np.outer(totals, totals) * delta))
    if expected == 0:
        return 1.0
    do = observed / n
    de = expected / (n * (n - 1))
    if de == 0:
        return float("nan")
    return 1.0 - (do / de)


def _coincidence(data: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    pairable = []
    for row in data:
        vals = row[~np.isnan(row)]
        if len(vals) >= 2:
            pairable.append(vals)
    if not pairable:
        return None, np.array([])
    values = np.unique(np.concatenate(pairable))
    index = {value: i for i, value in enumerate(values)}
    matrix = np.zeros((len(values), len(values)), dtype=float)
    for vals in pairable:
        m = len(vals)
        counts: dict[float, int] = defaultdict(int)
        for value in vals:
            counts[float(value)] += 1
        for left, n_left in counts.items():
            for right, n_right in counts.items():
                increment = n_left * (n_left - 1) if left == right else n_left * n_right
                matrix[index[left], index[right]] += increment / (m - 1)
    return matrix, values


def _delta(values: np.ndarray, marginals: np.ndarray, level: str) -> np.ndarray:
    level = level.lower()
    size = len(values)
    delta = np.zeros((size, size), dtype=float)
    if level == "nominal":
        delta[:, :] = 1.0
        np.fill_diagonal(delta, 0.0)
        return delta
    if level == "interval":
        diff = values.reshape(-1, 1) - values.reshape(1, -1)
        return diff ** 2
    if level == "ordinal":
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_marginals = marginals[order]
        cumulative = np.cumsum(sorted_marginals)
        for i, left in enumerate(sorted_values):
            for j, right in enumerate(sorted_values):
                if i == j:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                span = cumulative[hi] - (cumulative[lo - 1] if lo else 0.0)
                span -= (sorted_marginals[lo] + sorted_marginals[hi]) / 2.0
                delta[order[i], order[j]] = span ** 2
        return delta
    raise ValueError("level must be nominal, ordinal, or interval")


def alpha_from_long(frame: pd.DataFrame, item_col: str, rater_col: str,
                     value_cols: list[str], level: str) -> dict[str, float]:
    results = {}
    for column in value_cols:
        wide = frame.pivot_table(index=item_col, columns=rater_col, values=column, aggfunc="mean")
        results[column] = float(krippendorff_alpha(wide.to_numpy(), level=level))
    item_cols = [c for c in value_cols if c.startswith("eci_")]
    if item_cols:
        scored = frame.copy()
        scored["eci_composite"] = scored[item_cols].mean(axis=1)
        wide = scored.pivot_table(index=item_col, columns=rater_col, values="eci_composite", aggfunc="mean")
        results["eci_composite"] = float(krippendorff_alpha(wide.to_numpy(), level=level))
    item_cols = [c for c in value_cols if c.startswith("ici_")]
    if item_cols:
        scored = frame.copy()
        scored["ici_composite"] = scored[item_cols].mean(axis=1)
        wide = scored.pivot_table(index=item_col, columns=rater_col, values="ici_composite", aggfunc="mean")
        results["ici_composite"] = float(krippendorff_alpha(wide.to_numpy(), level=level))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Krippendorff alpha from a long annotation file.")
    parser.add_argument("ratings")
    parser.add_argument("--level", default="interval", choices=("nominal", "ordinal", "interval"))
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--rater-col", default="rater_id")
    args = parser.parse_args()
    frame = pd.read_csv(args.ratings)
    value_cols = [c for c in frame.columns if c.startswith("eci_") or c.startswith("ici_") or c.startswith("quality_")]
    results = alpha_from_long(frame, args.item_col, args.rater_col, value_cols, args.level)
    for name, value in results.items():
        print(f"{name}\t{value:.4f}")


if __name__ == "__main__":
    main()
