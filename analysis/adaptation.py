"""Contributor-month adaptation models (H5, H7) and the printed exiter scenarios.

The scenario rows are weighted averages of the stayer coefficient and an
assumption about exiters. They are not separate regressions.
"""

from __future__ import annotations

import pandas as pd

from analysis.hdfe import OLSResult, fit_hdfe


def fit_adaptation(panel: pd.DataFrame, outcome: str) -> OLSResult:
    data = panel.copy()
    if "post" not in data.columns:
        data["month"] = pd.to_datetime(data["month"], utc=True, errors="coerce")
        data["post"] = (data["month"] >= "2023-01-01").astype(float)
        data["answer_month"] = data["month"].dt.strftime("%Y-%m")
    data["post_x_status"] = data["post"] * data["status_pre"]
    if "sampling_weight" not in data.columns:
        data["sampling_weight"] = 1.0
    return fit_hdfe(
        data,
        outcome,
        ["post_x_status"],
        ["user_id", "answer_month"],
        "user_id",
        "sampling_weight",
    )


def attrition_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare pre-period means for contributors who do and do not appear after 2022."""
    data = panel.copy()
    data["month"] = pd.to_datetime(data["month"], utc=True, errors="coerce")
    pre = data.loc[data["month"] < "2022-12-01"]
    post_users = set(data.loc[data["month"] >= "2023-01-01", "user_id"])
    rows = []
    for user_id, group in pre.groupby("user_id"):
        rows.append({
            "user_id": user_id,
            "retained": user_id in post_users,
            "reputation_pre": group["reputation_pre"].iloc[0] if "reputation_pre" in group.columns else pd.NA,
            "status_pre": group["status_pre"].iloc[0] if "status_pre" in group.columns else pd.NA,
            "mean_answers": group["answer_count"].mean() if "answer_count" in group.columns else pd.NA,
            "mean_eci_share": group["eci_share"].mean() if "eci_share" in group.columns else pd.NA,
            "mean_asi": group["selected_asi"].mean() if "selected_asi" in group.columns else pd.NA,
        })
    users = pd.DataFrame(rows)
    summary = []
    for flag, label in ((True, "retained"), (False, "exited")):
        part = users.loc[users["retained"] == flag]
        summary.append({
            "group": label,
            "n_contributors": int(len(part)),
            "median_reputation_pre": part["reputation_pre"].median() if "reputation_pre" in part else pd.NA,
            "mean_answers": part["mean_answers"].mean(),
            "mean_eci_share": part["mean_eci_share"].mean(),
            "mean_asi": part["mean_asi"].mean(),
        })
    return pd.DataFrame(summary)


def exit_scenarios(h5: float, h7: float, share_stayers: float) -> pd.DataFrame:
    share_exit = 1.0 - share_stayers
    rows = []
    for name, factor in (("same_shift_as_stayers", 1.0), ("half_the_stayer_shift", 0.5),
                         ("no_shift", 0.0), ("opposite_shift", -1.0)):
        rows.append({
            "assumption_for_exiters": name,
            "h5_eci_share": share_stayers * h5 + share_exit * factor * h5,
            "h7_selected_asi": share_stayers * h7 + share_exit * factor * h7,
            "share_stayers": share_stayers,
        })
    return pd.DataFrame(rows)
