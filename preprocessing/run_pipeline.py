"""Apply the manuscript's sample accounting in the printed order.

Stages
1. Raw extract.
2. Drop answers outside 1 January 2019 through 31 December 2025.
3. Drop empty or non-machine-readable bodies.
4. Drop answers from accounts meeting at least two automation criteria.
5. Drop same-question near-duplicates with Jaccard similarity above 0.98.
6. Optional complete case, after ICI, ECI, and ASI have been merged.

Stage 6 is separate because the scores do not exist on the raw dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from manuscript_spec import SAMPLE_FLOW, WINDOW_END_EXCLUSIVE, WINDOW_START
from preprocessing.automated_accounts import flag_automated_accounts
from preprocessing.duplicates import duplicate_link_indicator, same_question_duplicate_mask
from preprocessing.textutil import primary_tag, tag_count, text_features, token_set

COMPLETE_CASE_COLUMNS = (
    "answer_id", "question_id", "user_id", "answer_creation_date",
    "net_vote", "ici", "eci", "asi", "primary_tag", "question_creation_date",
)


def run(answers: pd.DataFrame, duplicate_links: pd.DataFrame | None = None,
        retain_user_ids: set | None = None, apply_complete_case: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    ledger = []
    frame = answers.copy()
    _record(ledger, "raw", frame, len(answers))
    frame["answer_creation_date"] = pd.to_datetime(frame["answer_creation_date"], utc=True, errors="coerce")
    in_window = frame["answer_creation_date"].between(WINDOW_START, WINDOW_END_EXCLUSIVE, inclusive="left")
    frame = frame.loc[in_window].copy()
    _record(ledger, "outside_window", frame, len(answers))

    tokens, readable = [], []
    feature_rows = []
    for body in frame["answer_body"] if "answer_body" in frame.columns else [""] * len(frame):
        text = "" if body is None or (isinstance(body, float) and body != body) else str(body)
        broken = "\ufffd" in text or text.strip() == ""
        features = text_features(text)
        tokens.append(set() if broken else token_set(text))
        readable.append(not broken and features["answer_length"] > 0)
        feature_rows.append(features)
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    readable_mask = pd.Series(readable, index=frame.index, dtype=bool)
    frame = frame.loc[readable_mask].copy()
    tokens = [tok for tok, keep in zip(tokens, readable) if keep]
    _record(ledger, "empty_or_unreadable", frame, len(answers))

    frame["primary_tag"] = frame["tags"].map(primary_tag) if "tags" in frame.columns else None
    frame["n_tags"] = frame["tags"].map(tag_count) if "tags" in frame.columns else 0
    frame["duplicate_link"] = duplicate_link_indicator(frame["question_id"], duplicate_links)
    if "community_owned_date" in frame.columns:
        frame["community_wiki"] = frame["community_owned_date"].notna().astype(int)
    if "accepted_answer_id" in frame.columns:
        frame["accepted"] = (frame["accepted_answer_id"] == frame["answer_id"]).astype(int)

    accounts = flag_automated_accounts(frame, tokens)
    flagged = set(accounts.loc[accounts["automated"], "user_id"])
    if retain_user_ids:
        flagged -= set(retain_user_ids)
    keep_account = ~frame["user_id"].isin(flagged)
    # Deleted-user answers have no account id and are not removed by the account rule.
    keep_account = keep_account | frame["user_id"].isna()
    frame = frame.loc[keep_account].copy()
    tokens = [tok for tok, keep in zip(tokens, keep_account.to_numpy()) if keep]
    _record(ledger, "automated_accounts", frame, len(answers))

    drop_dup = same_question_duplicate_mask(frame, tokens)
    frame = frame.loc[~drop_dup.to_numpy()].copy()
    _record(ledger, "same_question_near_duplicates", frame, len(answers))

    if apply_complete_case:
        missing = [col for col in COMPLETE_CASE_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError("complete case requires columns: " + ", ".join(missing))
        before = len(frame)
        frame = frame.dropna(subset=list(COMPLETE_CASE_COLUMNS)).copy()
        ledger.append({
            "stage": "complete_case",
            "n_remaining": int(len(frame)),
            "n_removed_this_stage": int(before - len(frame)),
            "manuscript_remaining": _target("complete_case"),
        })
    frame = frame.drop(columns=["answer_body", "question_body"], errors="ignore")
    return frame, ledger


def _record(ledger: list[dict], stage: str, frame: pd.DataFrame, raw_n: int) -> None:
    previous = raw_n if not ledger else ledger[-1]["n_remaining"]
    ledger.append({
        "stage": stage,
        "n_remaining": int(len(frame)) if stage != "raw" else int(raw_n if stage == "raw" else len(frame)),
        "n_removed_this_stage": 0 if stage == "raw" else int(previous - len(frame)),
        "manuscript_remaining": _target(stage),
    })


def _target(stage: str):
    for name, remaining, _removed in SAMPLE_FLOW:
        if name == stage:
            return remaining
    return None


def write_ledger(ledger: list[dict], path: Path) -> None:
    frame = pd.DataFrame(ledger)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw Stack Overflow extract.")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--duplicates")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retain-users")
    parser.add_argument("--complete-case", action="store_true")
    args = parser.parse_args()
    answers = pd.read_csv(args.answers)
    links = pd.read_csv(args.duplicates) if args.duplicates else None
    retain = None
    if args.retain_users:
        retain = set(pd.read_csv(args.retain_users)["user_id"])
    cleaned, ledger = run(answers, links, retain, apply_complete_case=args.complete_case)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out / "cleaned_answers.csv", index=False)
    write_ledger(ledger, out / "sample_flow.csv")
    (out / "sample_flow.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
