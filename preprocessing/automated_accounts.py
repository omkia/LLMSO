"""Likely-automated accounts: at least two of the three printed criteria.

(i)  More than 50 answers inside any 24-hour window on three or more
     distinct calendar days.
(ii) Pairwise Jaccard similarity above 0.95 for at least 80 percent of
     answer-text pairs.
(iii) Account creation and first answer less than 90 seconds apart, and
     no later edit on the account's answers.

The manuscript reports that this rule removed 47,218 answers from 3,841
accounts. Those counts are audit targets, not values this module inserts.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from manuscript_spec import (
    ACCOUNT_CREATION_SECONDS,
    ACCOUNT_JACCARD,
    ACCOUNT_PAIR_SHARE,
    BURST_ANSWERS,
    BURST_DISTINCT_DAYS,
    MIN_AUTOMATION_CRITERIA,
)
from preprocessing.textutil import jaccard


def flag_automated_accounts(frame: pd.DataFrame, token_sets: list[set[int]]) -> pd.DataFrame:
    """Return one row per user_id with the three criteria and the joint flag."""
    work = frame.copy()
    work["_row"] = np.arange(len(work))
    work["answer_creation_date"] = pd.to_datetime(work["answer_creation_date"], utc=True, errors="coerce")
    work["account_creation_date"] = pd.to_datetime(work["account_creation_date"], utc=True, errors="coerce")
    if "answer_last_edit_date" in work.columns:
        work["answer_last_edit_date"] = pd.to_datetime(work["answer_last_edit_date"], utc=True, errors="coerce")
    rows = []
    for user_id, group in work.groupby("user_id", dropna=True, sort=False):
        indexes = group["_row"].to_numpy()
        sets = [token_sets[i] for i in indexes]
        burst = burst_on_three_days(group["answer_creation_date"])
        similar = high_pairwise_jaccard(sets)
        fast = fast_creation_without_edits(group)
        criteria = int(burst) + int(similar) + int(fast)
        rows.append({
            "user_id": user_id,
            "criterion_burst": burst,
            "criterion_jaccard": similar,
            "criterion_fast_creation": fast,
            "n_criteria": criteria,
            "automated": criteria >= MIN_AUTOMATION_CRITERIA,
            "n_answers": int(len(group)),
        })
    return pd.DataFrame(rows)


def burst_on_three_days(timestamps: pd.Series) -> bool:
    times = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if len(times) <= BURST_ANSWERS:
        return False
    values = times.to_numpy()
    days = set()
    left = 0
    for right in range(len(values)):
        while values[right] - values[left] > np.timedelta64(24, "h"):
            left += 1
        if (right - left + 1) > BURST_ANSWERS:
            days.add(pd.Timestamp(values[right]).date())
            if len(days) >= BURST_DISTINCT_DAYS:
                return True
    return False


def high_pairwise_jaccard(sets: list[set[int]]) -> bool:
    n = len(sets)
    if n < 2:
        return False
    if n <= 400:
        return _exact_pair_share(sets) >= ACCOUNT_PAIR_SHARE
    return _minhash_pair_share(sets) >= ACCOUNT_PAIR_SHARE


def fast_creation_without_edits(group: pd.DataFrame) -> bool:
    created = group["account_creation_date"].dropna()
    posted = group["answer_creation_date"].dropna()
    if created.empty or posted.empty:
        return False
    gap = posted.min() - created.min()
    if gap < timedelta(0) or gap >= timedelta(seconds=ACCOUNT_CREATION_SECONDS):
        return False
    if "answer_last_edit_date" not in group.columns:
        return False
    return bool(group["answer_last_edit_date"].isna().all())


def _exact_pair_share(sets: list[set[int]]) -> float:
    high = 0
    pairs = 0
    for i in range(len(sets)):
        left = sets[i]
        for j in range(i + 1, len(sets)):
            pairs += 1
            if jaccard(left, sets[j]) > ACCOUNT_JACCARD:
                high += 1
    return high / pairs if pairs else 0.0


def _minhash_pair_share(sets: list[set[int]], n_perm: int = 32, bands: int = 8) -> float:
    """Estimate the pair share for accounts too large for exact comparison.

    Exact pairwise comparison is the manuscript rule. This estimate is used
    only above 400 answers and is labeled in the account table via the
    calling code's size threshold.
    """
    rng = np.random.default_rng(0)
    a = rng.integers(1, 2**31 - 1, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, 2**31 - 1, size=n_perm, dtype=np.uint64)
    prime = np.uint64(4294967311)
    signatures = np.vstack([_signature(tokens, a, b, prime) for tokens in sets])
    return _exact_pair_share_signatures(signatures, bands)


def _signature(tokens: set[int], a: np.ndarray, b: np.ndarray, prime: np.uint64) -> np.ndarray:
    if not tokens:
        return np.full(len(a), np.iinfo(np.uint64).max)
    vals = np.fromiter(tokens, dtype=np.uint64, count=len(tokens))
    hashed = (a.reshape(-1, 1) * vals.reshape(1, -1) + b.reshape(-1, 1)) % prime
    return hashed.min(axis=1)


def _exact_pair_share_signatures(signatures: np.ndarray, bands: int) -> float:
    n, n_perm = signatures.shape
    width = n_perm // bands
    high = 0
    pairs = 0
    # Compare signatures directly; agreement fraction estimates Jaccard.
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            agree = np.mean(signatures[i] == signatures[j])
            if agree > ACCOUNT_JACCARD:
                high += 1
    return high / pairs if pairs else 0.0
