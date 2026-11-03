"""Locked annotation splits.

The manuscript stratifies the annotation sample by period, tag, score,
length, code share, and contributor expertise, then keeps near-duplicates
inside one partition. It does not print the training, validation, and
test fractions or the training-set size. Fractions are required arguments.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


STRAIA = ("period", "primary_tag", "score_bin", "length_bin", "code_share_bin", "expertise_bin")


def assign_splits(
    frame: pd.DataFrame,
    fractions: tuple[float, float, float],
    group_col: str = "near_dup_group",
    seed: int = 20221130,
) -> pd.DataFrame:
    train_p, val_p, test_p = fractions
    total = train_p + val_p + test_p
    if abs(total - 1.0) > 1e-8 or min(fractions) < 0:
        raise ValueError("fractions must be non-negative and sum to 1")
    out = frame.copy()
    if group_col not in out.columns:
        out[group_col] = out["item_id"].astype(str)
    rng = np.random.default_rng(seed)
    groups = []
    for group_id, group in out.groupby(group_col, sort=False):
        stratum = _modal_stratum(group)
        groups.append((str(group_id), stratum, len(group)))
    # Allocate whole groups. Shuffle inside each stratum so the lock is reproducible.
    assignment = {}
    by_stratum: dict[tuple, list] = {}
    for group_id, stratum, _size in groups:
        by_stratum.setdefault(stratum, []).append(group_id)
    for stratum, group_ids in by_stratum.items():
        order = list(group_ids)
        rng.shuffle(order)
        cuts = _cuts(len(order), fractions)
        for group_id, split in zip(order, _labels(len(order), cuts)):
            assignment[group_id] = split
    out["split"] = out[group_col].astype(str).map(assignment)
    return out


def _modal_stratum(group: pd.DataFrame) -> tuple:
    values = []
    for column in STRAIA:
        if column not in group.columns or group[column].dropna().empty:
            values.append("missing")
        else:
            values.append(str(group[column].mode(dropna=True).iloc[0]))
    return tuple(values)


def _cuts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int]:
    train = int(round(n * fractions[0]))
    val = int(round(n * fractions[1]))
    if train + val > n:
        val = max(0, n - train)
    return train, train + val


def _labels(n: int, cuts: tuple[int, int]) -> list[str]:
    train_end, val_end = cuts
    labels = []
    for i in range(n):
        if i < train_end:
            labels.append("train")
        elif i < val_end:
            labels.append("validation")
        else:
            labels.append("test")
    return labels


def assert_groups_locked(frame: pd.DataFrame, group_col: str = "near_dup_group") -> None:
    counts = frame.groupby(group_col)["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise AssertionError("near-duplicate groups split across partitions: " + ", ".join(map(str, leaked.index[:10])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign locked annotation splits.")
    parser.add_argument("ratings")
    parser.add_argument("--train", type=float, required=True)
    parser.add_argument("--validation", type=float, required=True)
    parser.add_argument("--test", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20221130)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.ratings)
    assigned = assign_splits(frame, (args.train, args.validation, args.test), seed=args.seed)
    assert_groups_locked(assigned)
    assigned.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
