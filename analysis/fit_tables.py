"""Estimate the manuscript's answer-level tables from a prepared sample.

Net-vote models are ordinary least squares with contributor, primary-tag,
and month fixed effects, and Liang–Zeger standard errors clustered by
primary tag. Reported p-values are the two-sided normal probabilities for
estimate / SE. Nothing here overwrites a coefficient or a covariance with
the printed table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.design import (
    CORE_FE,
    LOWER_ORDER,
    TEXT_CONTROLS,
    VALUATION_FOCAL,
    add_event_indicators,
    add_prewindow_indicators,
    add_supply,
    add_top_tag_indicators,
    available,
    prepare,
    pseudo_post,
    tag_volume_slope,
)
from analysis.hdfe import OLSResult, fit_hdfe, wald_joint
from manuscript_spec import LLM_PUBLIC


def _fit(frame: pd.DataFrame, y: str, cols: list[str]) -> OLSResult:
    use = [col for col in cols if col in frame.columns]
    return fit_hdfe(frame, y, use, list(CORE_FE), "primary_tag", "sampling_weight")


def valuation(frame: pd.DataFrame, extra: tuple[str, ...] = ()) -> OLSResult:
    data = prepare(frame)
    controls = add_top_tag_indicators(data)
    if "duplicate_link" in data.columns:
        controls.append("duplicate_link")
    if "n_tags" in data.columns:
        controls.append("n_tags")
    controls.extend(col for col in extra if col in data.columns)
    return _fit(data, "net_vote", list(LOWER_ORDER) + controls)


def baseline(frame: pd.DataFrame, outcome: str) -> OLSResult:
    data = prepare(frame)
    cols = ["asi", "post_x_asi"]
    if "n_tags" in data.columns:
        cols.append("n_tags")
    if "duplicate_link" in data.columns:
        cols.append("duplicate_link")
    return _fit(data, outcome, cols)


def event_study(frame: pd.DataFrame) -> tuple[OLSResult, dict]:
    data = prepare(frame)
    names = add_event_indicators(data)
    result = _fit(data, "net_vote", ["asi", *names])
    # Five leads, June-October 2022. The post-arrival contrast is k=0
    # through k=+12 (13 coefficients). k=+13 is estimated but not in this test.
    leads = [f"k_{k}_x_asi" for k in (-6, -5, -4, -3, -2)]
    post = [f"k_{k}_x_asi" for k in range(0, 13)]
    return result, {
        "leads_k_m6_to_m2": wald_joint(result, leads),
        "post_k_0_to_12": wald_joint(result, post),
    }


def prewindow(frame: pd.DataFrame) -> tuple[OLSResult, dict]:
    data = prepare(frame)
    names = add_prewindow_indicators(data)
    result = _fit(data, "net_vote", ["asi", *names])
    eight = [name for name in names if name.startswith("bin_20") and "dec" not in name and name != "bin_post_x_asi"]
    return result, {"eight_bins": wald_joint(result, eight)}


def time_placebos(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-arrival sample only, at the three printed pseudo-event dates."""
    data = prepare(frame)
    pre = data.loc[data["answer_creation_date"] < LLM_PUBLIC].copy()
    rows = []
    for event, label in (
        ("2019-12-01", "30 November 2019"),
        ("2020-12-01", "30 November 2020"),
        ("2021-07-01", "30 June 2021"),
    ):
        pre["post"] = pseudo_post(pre, event).to_numpy()
        pre["post_x_ici"] = pre["post"] * pre["ici"]
        pre["post_x_eci"] = pre["post"] * pre["eci"]
        result = _fit(pre, "net_vote", ["ici", "eci", "post_x_ici", "post_x_eci"])
        table = result.as_frame().set_index("term")
        for term in ("post_x_ici", "post_x_eci"):
            rows.append({
                "pseudo_event": label,
                "term": term,
                "estimate": table.loc[term, "estimate"] if term in table.index else np.nan,
                "std_error": table.loc[term, "std_error"] if term in table.index else np.nan,
                "n": result.n,
                "g": result.g,
            })
    return pd.DataFrame(rows)


def sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    masks = {"baseline": pd.Series(True, index=data.index)}
    if "community_wiki" in data.columns:
        masks["exclude_community_wiki"] = data["community_wiki"] != 1
    if "self_answer" in data.columns:
        masks["exclude_self_answers"] = data["self_answer"] != 1
    if "ai_flagged" in data.columns:
        masks["exclude_ai_flagged"] = data["ai_flagged"] != 1
    if all(col in data.columns for col in ("community_wiki", "self_answer", "ai_flagged")):
        masks["exclude_all_three"] = (
            (data["community_wiki"] != 1) & (data["self_answer"] != 1) & (data["ai_flagged"] != 1)
        )
    rows = []
    for name, mask in masks.items():
        result = valuation(data.loc[mask].drop(columns=["post", "post_x_ici", "post_x_eci",
                                                        "post_x_asi", "asi_x_ici", "asi_x_eci",
                                                        "post_x_asi_x_ici", "post_x_asi_x_eci"],
                                               errors="ignore"))
        table = result.as_frame().set_index("term")
        for term in VALUATION_FOCAL:
            rows.append({
                "restriction": name,
                "term": term,
                "estimate": table.loc[term, "estimate"] if term in table.index else np.nan,
                "std_error": table.loc[term, "std_error"] if term in table.index else np.nan,
                "n": result.n,
            })
    return pd.DataFrame(rows)


def with_tag_slope(frame: pd.DataFrame) -> OLSResult:
    data = prepare(frame)
    data["tag_slope"] = tag_volume_slope(data)
    data["post_x_tag_slope"] = data["post"] * data["tag_slope"]
    controls = add_top_tag_indicators(data)
    extra = ["post_x_tag_slope"]
    if "n_tags" in data.columns:
        extra.append("n_tags")
    if "duplicate_link" in data.columns:
        extra.append("duplicate_link")
    return _fit(data, "net_vote", list(LOWER_ORDER) + controls + extra)


def with_supply(frame: pd.DataFrame) -> OLSResult:
    data = add_supply(prepare(frame))
    return _fit(data, "net_vote", list(LOWER_ORDER) + ["supply", "supply_x_eci"])


def outside_copilot_languages(frame: pd.DataFrame, languages: set[str]) -> OLSResult:
    data = prepare(frame)
    subset = data.loc[~data["primary_tag"].astype(str).str.lower().isin({item.lower() for item in languages})]
    return valuation(subset.drop(columns=["post", "post_x_ici", "post_x_eci", "post_x_asi",
                                          "asi_x_ici", "asi_x_eci", "post_x_asi_x_ici",
                                          "post_x_asi_x_eci"], errors="ignore"))


def descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    rows = []
    answer_vars = [col for col in (
        "net_vote", "accepted", "comment_count", "ici", "eci", "answer_length", "code_ratio"
    ) if col in data.columns]
    for col in answer_vars:
        rows.append(_dist(col, data[col], "answer"))
    if "asi" in data.columns:
        questions = data.drop_duplicates("question_id")
        rows.append(_dist("asi", questions["asi"], "question"))
    if "complexity" in data.columns:
        questions = data.drop_duplicates("question_id")
        rows.append(_dist("complexity", questions["complexity"], "question"))
    if "reputation_pre" in data.columns:
        users = data.drop_duplicates("user_id")
        rows.append(_dist("reputation_pre", users["reputation_pre"], "contributor"))
    if "volatility" in data.columns:
        tags = data.drop_duplicates("primary_tag")
        rows.append(_dist("volatility", tags["volatility"], "tag"))
    return pd.DataFrame(rows)


def generalized_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF on mean-centered valuation terms, without absorbing the fixed effects.

    The printed VIF table includes Post as its own column, so this diagnostic
    does too. Post is absorbed in the estimating equation.
    """
    data = prepare(frame)
    cols = ["post", *LOWER_ORDER]
    cols = available(data, cols)
    centered = data[cols].astype(float)
    centered = centered - centered.mean()
    rows = []
    for col in cols:
        others = [other for other in cols if other != col]
        y = centered[col].to_numpy()
        x = np.column_stack([np.ones(len(centered)), centered[others].to_numpy()])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=1e-10)
        fitted = x @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float(((y - fitted) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        vif = 1 / (1 - r2) if r2 == r2 and r2 < 1 else np.inf
        rows.append({"term": col, "df": 1, "gvif": vif, "gvif_adjusted": float(np.sqrt(vif)) if np.isfinite(vif) else np.inf})
    return pd.DataFrame(rows)


def two_event(frame: pd.DataFrame) -> tuple[OLSResult, dict]:
    data = prepare(frame)
    names = []
    for k in range(-6, 6):
        name = f"copilot_{k}_x_asi"
        data[name] = ((data["copilot_k"] == k).astype(float)) * data["asi"]
        names.append(name)
    # Same clock as the event study: omit k=-1 (November 2022).
    # k=0 is December 2022. The set stops at k=+12, so it has 18 terms.
    for k in list(range(-6, 13)):
        if k == -1:
            continue
        name = f"llm_{k}_x_asi"
        data[name] = ((data["event_k"] == k).astype(float)) * data["asi"]
        names.append(name)
    result = _fit(data, "net_vote", ["asi", *names])
    tests = {
        "copilot_leads": wald_joint(result, [f"copilot_{k}_x_asi" for k in range(-6, 0)]),
        "copilot_lags": wald_joint(result, [f"copilot_{k}_x_asi" for k in range(0, 6)]),
        "llm_after_arrival": wald_joint(result, [f"llm_{k}_x_asi" for k in range(0, 13)]),
    }
    return result, tests


def _dist(name: str, values: pd.Series, level: str) -> dict:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return {"variable": name, "level": level, "n": 0}
    return {
        "variable": name,
        "level": level,
        "n": int(series.shape[0]),
        "mean": float(series.mean()),
        "sd": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }
