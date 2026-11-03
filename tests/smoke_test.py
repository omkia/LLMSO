"""Checks for the cleaning rules, alpha, the fixed-effect estimator, and the sample drawer."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deleted_scenarios import deleted_counts, scenario_table
from analysis.fit_tables import valuation
from analysis.inference import bonferroni, wald_row
from analysis.outcome_robustness import BINS, ordinal_bins
from analysis.hdfe import fit_hdfe
from annotation.krippendorff_alpha import krippendorff_alpha
from annotation.splits import assert_groups_locked, assign_splits
from models.metrics import binary_report, youden_threshold
from preprocessing.run_pipeline import run
from replication_sample.draw_stratified_sample import draw_sample


def test_alpha_bounds() -> None:
    perfect = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    assert krippendorff_alpha(perfect, "nominal") == 1.0
    assert abs(krippendorff_alpha(perfect, "interval") - 1.0) < 1e-9
    one_disagreement = np.array([[0.0, 1.0]])
    assert abs(krippendorff_alpha(one_disagreement, "nominal")) < 1e-9


def test_youden_separates() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    threshold = youden_threshold(y, scores)
    report = binary_report(y, scores, threshold)
    assert report["sensitivity"] == 1
    assert report["specificity"] == 1


def test_pipeline_rules() -> None:
    rows = []
    # Account 1 meets the burst rule and the near-copy rule.
    for day in range(3):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=10 * day)
        for i in range(51):
            rows.append(_row(1, 1000 + day, start + timedelta(minutes=i), "use function alpha beta gamma", created_gap=10_000))
    # Account 2 is a fast unedited account with copied text: criteria ii and iii.
    created = datetime(2021, 3, 1, tzinfo=timezone.utc)
    for i in range(4):
        rows.append(_row(2, 2000, created + timedelta(seconds=30 + i), "identical boilerplate text here", created_gap=30))
    # Account 3 is an ordinary pair of distinct answers on one question, plus a near copy.
    rows.append(_row(3, 3000, datetime(2021, 6, 1, tzinfo=timezone.utc), "install package delta and restart", created_gap=10_000))
    rows.append(_row(3, 3000, datetime(2021, 6, 2, tzinfo=timezone.utc), "install package delta and restart", created_gap=10_000))
    rows.append(_row(3, 3001, datetime(2021, 6, 3, tzinfo=timezone.utc), "unrelated explanation of memory barriers", created_gap=10_000))
    # Outside the window.
    rows.append(_row(3, 3002, datetime(2018, 1, 1, tzinfo=timezone.utc), "old answer that leaves the window", created_gap=10_000))
    frame = pd.DataFrame(rows)
    cleaned, ledger = run(frame)
    remaining_users = set(cleaned["user_id"])
    assert 1 not in remaining_users
    assert 2 not in remaining_users
    assert 3 in remaining_users
    assert (cleaned["question_id"] == 3000).sum() == 1
    assert (cleaned["question_id"] == 3002).sum() == 0
    by_stage = {row["stage"]: row for row in ledger}
    assert by_stage["raw"]["n_remaining"] == len(frame)
    assert by_stage["raw"]["manuscript_remaining"] == 5847312
    assert by_stage["same_question_near_duplicates"]["n_removed_this_stage"] == 1


def test_splits_lock_groups() -> None:
    frame = pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5, 6],
        "near_dup_group": ["a", "a", "b", "b", "c", "d"],
        "period": ["pre"] * 6,
        "primary_tag": ["python"] * 6,
        "score_bin": ["low"] * 6,
        "length_bin": ["mid"] * 6,
        "code_share_bin": ["low"] * 6,
        "expertise_bin": ["mid"] * 6,
    })
    assigned = assign_splits(frame, (0.5, 0.25, 0.25), seed=1)
    assert_groups_locked(assigned)
    assert assigned.loc[assigned["near_dup_group"] == "a", "split"].nunique() == 1


def test_hdfe_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    user = rng.integers(0, 150, n)
    tag = rng.integers(0, 25, n)
    month = rng.integers(0, 20, n)
    x = rng.normal(size=n)
    y = 0.5 * x + user * 0.01 + tag * 0.02 + month * 0.03 + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame({
        "y": y, "x": x, "user_id": user, "primary_tag": tag, "answer_month": month,
        "sampling_weight": 1.0,
    })
    result = fit_hdfe(frame, "y", ["x"], ["user_id", "primary_tag", "answer_month"], "primary_tag", "sampling_weight")
    assert abs(result.beta[0] - 0.5) < 0.05
    assert result.se[0] > 0
    assert result.g == 25


def test_valuation_recovers_interactions() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    frame = pd.DataFrame({
        "answer_id": np.arange(n),
        "question_id": rng.integers(0, 800, n),
        "user_id": rng.integers(0, 120, n),
        "primary_tag": rng.integers(0, 20, n).astype(str),
        "answer_creation_date": pd.to_datetime("2019-01-01", utc=True) + pd.to_timedelta(rng.integers(0, 80 * 30, n), unit="D"),
        "ici": rng.random(n),
        "eci": rng.random(n),
        "asi": rng.random(n),
        "n_tags": rng.integers(1, 5, n),
        "duplicate_link": rng.integers(0, 2, n),
    })
    post = (frame["answer_creation_date"] >= "2023-01-01").astype(float)
    user = frame["user_id"].to_numpy()
    tag = frame["primary_tag"].astype("category").cat.codes.to_numpy()
    month = frame["answer_creation_date"].dt.strftime("%Y-%m").astype("category").cat.codes.to_numpy()
    signal = -0.40 * post * frame["ici"] + 0.30 * post * frame["eci"]
    frame["net_vote"] = signal + user * 0.01 + tag * 0.02 + month * 0.01 + rng.normal(scale=0.15, size=n)
    result = valuation(frame)
    table = result.as_frame().set_index("term")
    assert abs(table.loc["post_x_ici", "estimate"] - (-0.40)) < 0.08
    assert abs(table.loc["post_x_eci", "estimate"] - 0.30) < 0.08


def test_printed_inference_and_targets() -> None:
    import json
    h1 = wald_row(-0.153, 0.036, n_tests=12)
    assert abs(h1["z"] - (-4.25)) < 1e-12
    assert abs(h1["p_value"] - 2.1377051549868865e-05) < 1e-15
    assert abs(bonferroni(0.00002, 12) - 0.00024) < 1e-12
    h3 = wald_row(-0.139, 0.045, n_tests=12)
    assert abs(h3["p_value"] - 0.002009) < 1e-5
    assert abs(h3["bonferroni"] - 0.0241) < 1e-3
    h4 = wald_row(0.176, 0.052, n_tests=12)
    assert h4["p_value"] < 0.001
    assert h4["bonferroni"] < 0.05
    targets = json.loads((ROOT / "analysis" / "manuscript_targets.json").read_text(encoding="utf-8"))
    ses = {item["term"]: item["printed_se"] for item in targets["valuation_net_vote"]}
    assert ses == {
        "post_x_ici": 0.036,
        "post_x_eci": 0.041,
        "post_x_asi_x_ici": 0.045,
        "post_x_asi_x_eci": 0.052,
    }
    assert len(BINS) - 1 == 5
    coded = ordinal_bins(pd.Series([-3, 0, 1, 4, 9]))
    assert list(coded.astype(int)) == [0, 1, 2, 3, 4]
    grid = scenario_table().set_index("scenario")
    n_post, n_pre = deleted_counts(0.08, 0.03)
    assert n_post == 137003
    assert n_pre == 75076
    assert grid.loc["1_baseline", "printed_post_x_ici"] == -0.119
    assert grid.loc["5_worst_case", "printed_post_x_eci"] == 0.184
    assert (grid["printed_post_x_ici"] < 0).all()
    assert (grid["printed_post_x_eci"] > 0).all()
    assert (grid["kind"] == "hypothetical_bound").all()
    assert round(0.34 * 0.036, 4) == 0.0122
    assert round(0.077 / 0.153, 3) == 0.503


def test_sample_is_a_draw_not_a_synthesis() -> None:
    rng = np.random.default_rng(2)
    n = 400
    source = pd.DataFrame({
        "answer_id": np.arange(n),
        "question_id": rng.integers(0, 80, n),
        "user_id": rng.integers(0, 40, n),
        "answer_creation_date": pd.to_datetime("2020-01-01", utc=True) + pd.to_timedelta(rng.integers(0, 200, n), unit="D"),
        "primary_tag": rng.choice(["python", "r", "sql"], n),
        "asi": rng.random(n),
        "answer_body": ["secret post text"] * n,
        "net_vote": rng.normal(size=n),
    })
    sample, manifest = draw_sample(source, fraction=0.5, seed=7, salt="test-salt")
    assert "answer_body" not in sample.columns
    assert set(sample["answer_id"]).isdisjoint(set(source["answer_id"].astype(str)))
    assert sample["sampling_weight"].min() > 0
    assert manifest["n_source"].sum() == n


def _row(user_id, question_id, when, body, created_gap):
    created = when - timedelta(seconds=created_gap)
    return {
        "answer_id": f"{user_id}-{when.isoformat()}-{question_id}",
        "question_id": question_id,
        "user_id": user_id,
        "question_owner_id": 999,
        "answer_creation_date": when,
        "account_creation_date": created,
        "answer_body": f"<p>{body}</p>",
        "net_vote": 1,
        "comment_count": 0,
        "community_owned_date": pd.NaT,
        "answer_last_edit_date": pd.NaT,
        "question_creation_date": when - timedelta(days=1),
        "tags": "<python>",
        "accepted_answer_id": -1,
        "self_answer": 0,
    }


if __name__ == "__main__":
    test_alpha_bounds()
    test_youden_separates()
    test_splits_lock_groups()
    test_pipeline_rules()
    test_hdfe_recovers_slope()
    test_valuation_recovers_interactions()
    test_printed_inference_and_targets()
    test_sample_is_a_draw_not_a_synthesis()
    print("smoke tests passed")
