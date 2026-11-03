"""Hypothetical deleted-answer bounds from the manuscript.

The public dump omits deleted posts. The printed cells are hypothetical
bounds under assumed deletion rates and vote penalties. They are not
estimated regression coefficients and are not recovered from a covariance
matrix. scenario_table() writes that checklist. fit_deleted_scenarios()
is a separate imputation refit on whatever file it is given. It does not
overwrite the hypothetical bounds with the refit.
"""

from __future__ import annotations

import pandas as pd

from analysis.design import VALUATION_FOCAL, prepare
from analysis.fit_tables import valuation
from manuscript_spec import POST_START, PRE_PERIOD_ANSWERS, POST_PERIOD_ANSWERS


SCENARIOS = (
    {"scenario": "1_baseline", "post_rate": 0.08, "pre_rate": 0.03, "penalty_ici": -1.8, "penalty_eci": -0.4, "printed_post_x_ici": -0.119, "printed_post_x_eci": 0.211},
    {"scenario": "2_conservative", "post_rate": 0.05, "pre_rate": 0.02, "penalty_ici": -1.8, "penalty_eci": -0.4, "printed_post_x_ici": -0.131, "printed_post_x_eci": 0.218},
    {"scenario": "3_aggressive", "post_rate": 0.10, "pre_rate": 0.05, "penalty_ici": -1.8, "penalty_eci": -0.4, "printed_post_x_ici": -0.105, "printed_post_x_eci": 0.202},
    {"scenario": "4_high_penalty", "post_rate": 0.08, "pre_rate": 0.03, "penalty_ici": -2.5, "penalty_eci": -0.6, "printed_post_x_ici": -0.108, "printed_post_x_eci": 0.195},
    {"scenario": "5_worst_case", "post_rate": 0.10, "pre_rate": 0.05, "penalty_ici": -2.5, "penalty_eci": -0.6, "printed_post_x_ici": -0.092, "printed_post_x_eci": 0.184},
    {"scenario": "6_minimal_impact", "post_rate": 0.02, "pre_rate": 0.01, "penalty_ici": -1.0, "penalty_eci": -0.2, "printed_post_x_ici": -0.142, "printed_post_x_eci": 0.225},
)


def deleted_counts(post_rate: float, pre_rate: float, post_n: int = POST_PERIOD_ANSWERS, pre_n: int = PRE_PERIOD_ANSWERS) -> tuple[int, int]:
    return round(post_rate * post_n), round(pre_rate * pre_n)


def scenario_table() -> pd.DataFrame:
    rows = []
    for item in SCENARIOS:
        n_post, n_pre = deleted_counts(item["post_rate"], item["pre_rate"])
        rows.append({
            **item,
            "n_deleted_post": n_post,
            "n_deleted_pre": n_pre,
            "hypothetical_post_x_ici_bound": item["printed_post_x_ici"],
            "hypothetical_post_x_eci_bound": item["printed_post_x_eci"],
            "kind": "hypothetical_bound",
        })
    return pd.DataFrame(rows)


def impute_deleted(frame: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    data = prepare(frame)
    post = data["answer_creation_date"] >= POST_START
    pre = ~post
    n_post, n_pre = deleted_counts(scenario["post_rate"], scenario["pre_rate"], int(post.sum()), int(pre.sum()))
    pieces = [data]
    if n_post and post.any():
        top_cut = data.loc[post, "asi"].quantile(0.75)
        top = data.loc[post & (data["asi"] >= top_cut)]
        if top.empty:
            top = data.loc[post]
        high_ici = top.loc[top["ici"] >= top["ici"].median()]
        high_eci = top.loc[top["eci"] >= top["eci"].median()]
        if high_ici.empty:
            high_ici = top
        if high_eci.empty:
            high_eci = top
        n_ici = n_post // 2
        n_eci = n_post - n_ici
        ici_rows = high_ici.sample(n=n_ici, replace=True, random_state=1).copy()
        eci_rows = high_eci.sample(n=n_eci, replace=True, random_state=2).copy()
        ici_rows["net_vote"] = float(high_ici["net_vote"].mean()) + scenario["penalty_ici"]
        eci_rows["net_vote"] = float(high_eci["net_vote"].mean()) + scenario["penalty_eci"]
        pieces.extend([ici_rows, eci_rows])
    if n_pre and pre.any():
        pieces.append(data.loc[pre].sample(n=n_pre, replace=True, random_state=3).copy())
    return pd.concat(pieces, ignore_index=True)


def fit_deleted_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in SCENARIOS:
        result = valuation(impute_deleted(frame, item))
        table = result.as_frame().set_index("term")
        n_post, n_pre = deleted_counts(item["post_rate"], item["pre_rate"])
        for term, printed_key in (("post_x_ici", "printed_post_x_ici"), ("post_x_eci", "printed_post_x_eci")):
            if term not in VALUATION_FOCAL:
                continue
            rows.append({
                "scenario": item["scenario"],
                "term": term,
                "imputation_refit": float(table.loc[term, "estimate"]) if term in table.index else float("nan"),
                "imputation_refit_se": float(table.loc[term, "std_error"]) if term in table.index else float("nan"),
                "hypothetical_bound": item[printed_key],
                "n_deleted_post_full_sample": n_post,
                "n_deleted_pre_full_sample": n_pre,
                "n_fit": result.n,
            })
    return pd.DataFrame(rows)
