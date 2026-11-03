"""Design matrices for the printed equations.

Regressors are on the original [0, 1] index scale. The manuscript's VIF
table is a separate mean-centered diagnostic and is not used to redefine
the estimating equations.
"""

from __future__ import annotations

import math

import pandas as pd

from manuscript_spec import (
    COPILOT_GA,
    EVENT_K0_MONTH,
    EVENT_K0_YEAR,
    LLM_PUBLIC,
    POST_START,
    SUPPLY_ECI_MAX,
    SUPPLY_ICI_MIN,
)


CORE_FE = ("user_id", "primary_tag", "answer_month")
LOWER_ORDER = (
    "ici", "eci", "post_x_ici", "post_x_eci",
    "asi", "post_x_asi", "asi_x_ici", "asi_x_eci",
    "post_x_asi_x_ici", "post_x_asi_x_eci",
)
VALUATION_FOCAL = ("post_x_ici", "post_x_eci", "post_x_asi_x_ici", "post_x_asi_x_eci")
TEXT_CONTROLS = ("answer_length", "flesch_kincaid", "code_ratio", "formatting_score")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["answer_creation_date"] = pd.to_datetime(out["answer_creation_date"], utc=True, errors="coerce")
    out["answer_month"] = out["answer_creation_date"].dt.strftime("%Y-%m")
    out["post"] = (out["answer_creation_date"] >= POST_START).astype(float)
    out["month_index"] = out["answer_creation_date"].dt.year * 12 + out["answer_creation_date"].dt.month
    origin = EVENT_K0_YEAR * 12 + EVENT_K0_MONTH
    out["event_k"] = out["month_index"] - origin
    copilot_origin = int(COPILOT_GA[:4]) * 12 + int(COPILOT_GA[5:7])
    out["copilot_k"] = out["month_index"] - copilot_origin
    if "question_creation_date" in out.columns:
        q = pd.to_datetime(out["question_creation_date"], utc=True, errors="coerce")
        out["question_year"] = q.dt.year
        out["stratum"] = out["primary_tag"].astype(str) + ":" + out["question_year"].astype("Int64").astype(str)
    for col in ("ici", "eci", "asi"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["post_x_ici"] = out["post"] * out["ici"] if "ici" in out.columns else pd.NA
    out["post_x_eci"] = out["post"] * out["eci"] if "eci" in out.columns else pd.NA
    if "asi" in out.columns:
        out["post_x_asi"] = out["post"] * out["asi"]
        if "ici" in out.columns:
            out["asi_x_ici"] = out["asi"] * out["ici"]
            out["post_x_asi_x_ici"] = out["post"] * out["asi"] * out["ici"]
        if "eci" in out.columns:
            out["asi_x_eci"] = out["asi"] * out["eci"]
            out["post_x_asi_x_eci"] = out["post"] * out["asi"] * out["eci"]
    if "sampling_weight" not in out.columns:
        out["sampling_weight"] = 1.0
    return out


def available(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    return [col for col in columns if col in frame.columns and frame[col].notna().any()]


def add_top_tag_indicators(frame: pd.DataFrame) -> list[str]:
    """Indicators for the five most common tags appearing anywhere in ``tags``.

    Primary-tag indicators would be absorbed by the primary-tag fixed effect.
    The manuscript describes these as additional tag indicators.
    """
    if "tags" not in frame.columns:
        return []
    from preprocessing.textutil import tag_set
    counts: dict[str, int] = {}
    parsed = []
    for value in frame["tags"]:
        tags = tag_set(value)
        parsed.append(tags)
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    top = [tag for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
    names = []
    for rank, tag in enumerate(top, start=1):
        name = f"has_top_tag_{rank}"
        frame[name] = [float(tag in tags) for tags in parsed]
        names.append(name)
    return names


def add_event_indicators(frame: pd.DataFrame) -> list[str]:
    names = []
    k = frame["event_k"]
    for value in list(range(-6, 14)):
        if value == -1:
            continue
        name = f"k_{value}_x_asi"
        frame[name] = ((k == value).astype(float)) * frame["asi"]
        names.append(name)
    # k = 0 (December 2022) is included. k = -1 (November 2022) is omitted.
    # Months after k = +13 are collapsed into one bin. Baseline Post still
    # starts in January 2023.
    frame["k_prebin_x_asi"] = ((k < -6).astype(float)) * frame["asi"]
    frame["k_postbin_x_asi"] = ((k > 13).astype(float)) * frame["asi"]
    names.extend(["k_prebin_x_asi", "k_postbin_x_asi"])
    return names


def add_prewindow_indicators(frame: pd.DataFrame) -> list[str]:
    """Eight pre-arrival bins, with November 2022 as the omitted month."""
    start = pd.to_datetime(frame["answer_creation_date"], utc=True)
    bins = {
        "bin_2019h1_x_asi": ("2019-01-01", "2019-07-01"),
        "bin_2019h2_x_asi": ("2019-07-01", "2020-01-01"),
        "bin_2020h1_x_asi": ("2020-01-01", "2020-07-01"),
        "bin_2020h2_x_asi": ("2020-07-01", "2021-01-01"),
        "bin_2021h1_x_asi": ("2021-01-01", "2021-07-01"),
        "bin_2021h2_x_asi": ("2021-07-01", "2022-01-01"),
        "bin_2022_jan_may_x_asi": ("2022-01-01", "2022-06-01"),
        "bin_2022_jun_oct_x_asi": ("2022-06-01", "2022-11-01"),
    }
    names = []
    for name, (lo, hi) in bins.items():
        frame[name] = (start.between(lo, hi, inclusive="left").astype(float)) * frame["asi"]
        names.append(name)
    # Keep December 2022 and the post period out of the November reference.
    frame["bin_2022_dec_x_asi"] = (
        start.between("2022-12-01", "2023-01-01", inclusive="left").astype(float) * frame["asi"]
    )
    frame["bin_post_x_asi"] = (start.ge(POST_START).astype(float)) * frame["asi"]
    names.extend(["bin_2022_dec_x_asi", "bin_post_x_asi"])
    return names


def add_supply(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag-month log count of high-ICI, low-ECI answers.

    Thresholds 0.70 and 0.30 are printed. The manuscript treats this
    control as endogenous.
    """
    out = frame.copy()
    high = (out["ici"] > SUPPLY_ICI_MIN) & (out["eci"] < SUPPLY_ECI_MAX)
    key = out["primary_tag"].astype(str) + "|" + out["answer_month"].astype(str)
    counts = high.groupby(key).sum()
    out["supply"] = (1.0 + key.map(counts).astype(float)).map(math.log)
    out["supply_x_eci"] = out["supply"] * out["eci"]
    return out


def tag_volume_slope(frame: pd.DataFrame) -> pd.Series:
    """Linear monthly change in answer volume from January 2019 through November 2022."""
    pre = frame.loc[frame["answer_creation_date"] < LLM_PUBLIC]
    slopes = {}
    for tag, group in pre.groupby("primary_tag"):
        counts = group.groupby("month_index").size()
        if len(counts) < 2:
            slopes[tag] = 0.0
            continue
        x = counts.index.to_numpy(dtype=float)
        y = counts.to_numpy(dtype=float)
        x = x - x.mean()
        denom = float((x ** 2).sum())
        slopes[tag] = float((x * (y - y.mean())).sum() / denom) if denom else 0.0
    return frame["primary_tag"].map(slopes).astype(float)


def pseudo_post(frame: pd.DataFrame, event: str, end: str = LLM_PUBLIC) -> pd.Series:
    dates = frame["answer_creation_date"]
    return ((dates >= event) & (dates < end)).astype(float)
