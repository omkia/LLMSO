"""High-dimensional fixed-effect OLS with Liang–Zeger cluster-robust covariance.

Month, tag, and contributor fixed effects are removed by alternating
projections. The cluster factor is applied to the within estimator.
This is the estimator the manuscript states for the net-vote models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    n: int
    g: int
    rank: int
    absorbed: list[str]
    resid_sd: float

    def as_frame(self) -> pd.DataFrame:
        from analysis.inference import normal_interval, two_sided_normal_p

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.divide(self.beta, self.se, out=np.full_like(self.beta, np.nan), where=self.se > 0)
        p_value = []
        ci_low = []
        ci_high = []
        for estimate, std_error, z_value in zip(self.beta, self.se, z):
            if std_error > 0 and z_value == z_value:
                p_value.append(two_sided_normal_p(z_value))
                low, high = normal_interval(estimate, std_error)
                ci_low.append(low)
                ci_high.append(high)
            else:
                p_value.append(np.nan)
                ci_low.append(np.nan)
                ci_high.append(np.nan)
        return pd.DataFrame({
            "term": self.names,
            "estimate": self.beta,
            "std_error": self.se,
            "z": z,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "absorbed": [name in self.absorbed for name in self.names],
        })


def fit_hdfe(
    frame: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str],
    cluster_col: str,
    weight_col: str | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> OLSResult:
    needed = [y_col, *x_cols, *fe_cols, cluster_col]
    if weight_col:
        needed.append(weight_col)
    data = frame.dropna(subset=needed).copy()
    if data.empty:
        raise ValueError("no complete rows for " + y_col)
    y = data[y_col].to_numpy(dtype=float)
    x = data[x_cols].to_numpy(dtype=float)
    weights = data[weight_col].to_numpy(dtype=float) if weight_col else np.ones(len(data))
    if np.any(weights <= 0):
        raise ValueError("sampling weights must be positive")
    groups = [data[col].to_numpy() for col in fe_cols]
    y_d, x_d = _demean_together(y, x, groups, weights, max_iter, tol)
    scales = np.sqrt(np.average(x_d ** 2, axis=0, weights=weights))
    keep = scales > 1e-8
    absorbed = [name for name, flag in zip(x_cols, ~keep) if flag]
    if not np.any(keep):
        raise ValueError("every regressor was absorbed by the fixed effects")
    x_keep = x_d[:, keep]
    names = [name for name, flag in zip(x_cols, keep) if flag]
    sw = np.sqrt(weights)
    beta_keep, _, _, _ = np.linalg.lstsq(x_keep * sw[:, None], y_d * sw, rcond=1e-10)
    fitted = x_keep @ beta_keep
    resid = y_d - fitted
    xtx_inv = np.linalg.pinv((x_keep * sw[:, None]).T @ (x_keep * sw[:, None]), rcond=1e-10)
    cov, g = _cluster_cov(x_keep, resid, weights, data[cluster_col].to_numpy(), xtx_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    beta = np.full(len(x_cols), np.nan)
    se_full = np.full(len(x_cols), np.nan)
    cov_full = np.full((len(x_cols), len(x_cols)), np.nan)
    idx = np.flatnonzero(keep)
    beta[idx] = beta_keep
    se_full[idx] = se
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            cov_full[i, j] = cov[a, b]
    return OLSResult(x_cols, beta, se_full, cov_full, len(data), g, len(names), absorbed, float(np.std(resid)))


def wald_joint(result: OLSResult, terms: list[str]) -> dict[str, float]:
    """Joint Wald test that the named coefficients are zero."""
    present = [term for term in terms if term in result.names and term not in result.absorbed]
    if not present:
        return {"df": 0, "wald": float("nan"), "f": float("nan"), "g": result.g}
    idx = [result.names.index(term) for term in present]
    b = result.beta[idx]
    cov = result.cov[np.ix_(idx, idx)]
    try:
        stat = float(b.T @ np.linalg.pinv(cov, rcond=1e-12) @ b)
    except np.linalg.LinAlgError:
        stat = float("nan")
    q = len(present)
    f_stat = stat / q if q else float("nan")
    return {"df": q, "wald": stat, "f": f_stat, "g": result.g, "terms": len(present)}


def _demean_together(y, x, groups, weights, max_iter, tol):
    y_d = y.reshape(-1, 1).copy()
    x_d = x.copy()
    coded = []
    for group in groups:
        codes, _ = pd.factorize(group, sort=False)
        coded.append((codes, int(codes.max()) + 1 if len(codes) else 0))
    for _ in range(max_iter):
        prev_y = y_d.copy()
        prev_x = x_d.copy()
        for codes, n_groups in coded:
            y_d = _demean(y_d, codes, n_groups, weights)
            x_d = _demean(x_d, codes, n_groups, weights)
        if max(np.max(np.abs(y_d - prev_y)), np.max(np.abs(x_d - prev_x))) < tol:
            break
    return y_d.ravel(), x_d


def _demean(values, codes, n_groups, weights):
    w = weights.reshape(-1, 1)
    sums = np.zeros((n_groups, values.shape[1]))
    wsum = np.zeros((n_groups, 1))
    np.add.at(sums, codes, values * w)
    np.add.at(wsum, codes, w)
    means = sums / np.maximum(wsum, 1e-15)
    return values - means[codes]


def _cluster_cov(x, resid, weights, clusters, xtx_inv):
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    k = x.shape[1]
    n = x.shape[0]
    meat = np.zeros((k, k))
    for cluster_id in range(g):
        idx = np.flatnonzero(codes == cluster_id)
        score = x[idx].T @ (weights[idx] * resid[idx])
        meat += np.outer(score, score)
    if g > 1 and n > k:
        meat *= (g / (g - 1)) * ((n - 1) / (n - k))
    cov = xtx_inv @ meat @ xtx_inv
    return cov, g
