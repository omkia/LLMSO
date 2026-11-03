"""Train the AI-answerability index from pre-outcome question features.

Expert ratings are the target. The manuscript requires grouped
cross-validation by tag and time and a score scaled to [0, 1]. It does
not publish the feature dictionary or the learner. Both come from the
column list of the training file and from hparams.json.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.hparams import FORBIDDEN_ANSWER_FEATURES, require_hparams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--rating-max", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    hparams = require_hparams(args.hparams, "asi")
    frame = pd.read_csv(args.questions)
    if "expert_rating" not in frame.columns:
        raise SystemExit("questions file needs expert_rating")
    for column in ("net_vote", "accepted", "score"):
        if column in frame.columns:
            raise SystemExit(f"{column} is an answer outcome and cannot train ASI")
    feature_cols = [
        col for col in frame.columns
        if col not in FORBIDDEN_ANSWER_FEATURES
        and col not in ("expert_rating", "question_id", "primary_tag", "question_creation_date", "group_id")
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not feature_cols:
        raise SystemExit("no numeric pre-outcome features")
    y = frame["expert_rating"].to_numpy(dtype=float) / args.rating_max
    if np.nanmax(y) > 1 or np.nanmin(y) < 0:
        raise SystemExit("expert_rating / rating-max is outside [0, 1]")
    dates = pd.to_datetime(frame["question_creation_date"], utc=True, errors="coerce")
    half = dates.dt.year.astype(str) + "H" + ((dates.dt.month.sub(1) // 6) + 1).astype(str)
    groups = frame["primary_tag"].astype(str) + "|" + half.astype(str)
    learner = _learner(hparams)
    folds = GroupKFold(n_splits=min(5, groups.nunique()))
    pred = np.full(len(frame), np.nan)
    x = frame[feature_cols].to_numpy(dtype=float)
    for train_idx, test_idx in folds.split(x, y, groups):
        model = _learner(hparams)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = np.clip(model.predict(x[test_idx]), 0, 1)
    # Unused binding keeps a fitted full-sample model from being the reported score.
    del learner
    out = frame[["question_id"]].copy()
    out["asi"] = pred
    out.to_csv(args.out, index=False)


def _learner(hparams: dict):
    name = hparams["learner"]
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            learning_rate=float(hparams["learning_rate"]),
            max_depth=int(hparams["max_depth"]),
            max_iter=int(hparams["n_estimators"]),
            random_state=int(hparams["seed"]),
        )
    raise SystemExit("supported asi learner in this script: hist_gradient_boosting")


if __name__ == "__main__":
    main()
