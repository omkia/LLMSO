"""Same-question near-duplicates.

The manuscript keeps community-closure duplicates and stores them as a
binary control. It drops exact-text near-duplicates posted to the same
question when Jaccard similarity exceeds 0.98. The earliest answer in a
near-duplicate component is kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from manuscript_spec import SAME_QUESTION_JACCARD
from preprocessing.textutil import jaccard


def same_question_duplicate_mask(frame: pd.DataFrame, token_sets: list[set[int]]) -> pd.Series:
    """True on the rows that the same-question rule removes."""
    drop = np.zeros(len(frame), dtype=bool)
    work = frame.reset_index(drop=True)
    work["_row"] = np.arange(len(work))
    work["answer_creation_date"] = pd.to_datetime(work["answer_creation_date"], utc=True, errors="coerce")
    for _, group in work.groupby("question_id", sort=False):
        rows = group["_row"].to_list()
        if len(rows) < 2:
            continue
        parent = {row: row for row in rows}

        def find(row: int) -> int:
            while parent[row] != row:
                parent[row] = parent[parent[row]]
                row = parent[row]
            return row

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a

        sets = {row: token_sets[row] for row in rows}
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if jaccard(sets[left], sets[right]) > SAME_QUESTION_JACCARD:
                    union(left, right)
        components: dict[int, list[int]] = {}
        for row in rows:
            components.setdefault(find(row), []).append(row)
        for members in components.values():
            if len(members) < 2:
                continue
            ordered = group.set_index("_row").loc[members].sort_values(
                ["answer_creation_date", "answer_id"], kind="mergesort"
            )
            drop[ordered.index.to_numpy()[1:]] = True
    return pd.Series(drop, index=frame.index)


def duplicate_link_indicator(answer_questions: pd.Series, duplicate_links: pd.DataFrame | None) -> pd.Series:
    """One if the parent question is either side of a community duplicate link."""
    if duplicate_links is None or duplicate_links.empty:
        return pd.Series(0, index=answer_questions.index, dtype=int)
    linked = set(duplicate_links["question_id"].dropna().astype(str))
    if "related_question_id" in duplicate_links.columns:
        linked.update(duplicate_links["related_question_id"].dropna().astype(str))
    return answer_questions.astype(str).isin(linked).astype(int)
