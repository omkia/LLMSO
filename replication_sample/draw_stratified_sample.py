"""Draw the stratified anonymized replication sample from an analytical file.

The manuscript's redistribution note is a size and attribution choice.
Stack Overflow user content in the public dump is published under CC BY-SA;
this drawer removes post text and display names and replaces ids with
salted hashes. It does not synthesize rows. Sampling weights stay with
the drawn rows. The estimator does not rescale standard errors to the
printed table.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path

import numpy as np
import pandas as pd

from manuscript_spec import SAMPLE_FRACTION

ID_COLUMNS = ("answer_id", "question_id", "user_id", "question_owner_id")
TEXT_COLUMNS = {
    "answer_body", "question_body", "body", "title", "display_name",
    "owner_display_name", "about_me", "website_url", "location", "email",
    "comment_text",
}


def draw_sample(
    frame: pd.DataFrame,
    fraction: float = SAMPLE_FRACTION,
    seed: int = 20221130,
    salt: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not salt:
        raise ValueError("a salt is required so hashed ids are not reversible from the file alone")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    work = frame.copy()
    work["_stratum"] = _stratum_key(work)
    rng = np.random.default_rng(seed)
    pieces = []
    manifest_rows = []
    for stratum, group in work.groupby("_stratum", sort=False):
        n = len(group)
        expected = n * fraction
        n_take = int(np.floor(expected))
        if rng.random() < (expected - n_take):
            n_take += 1
        n_take = min(n, n_take)
        if n_take == 0:
            manifest_rows.append({"stratum": stratum, "n_source": n, "n_sample": 0, "weight": np.nan})
            continue
        chosen = group.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1)))
        chosen = chosen.copy()
        chosen["sampling_weight"] = n / n_take
        pieces.append(chosen)
        manifest_rows.append({"stratum": stratum, "n_source": n, "n_sample": n_take, "weight": n / n_take})
    if not pieces:
        raise ValueError("the draw kept zero rows; use a larger source or fraction")
    sample = pd.concat(pieces, ignore_index=True)
    sample = anonymize(sample, salt)
    manifest = pd.DataFrame(manifest_rows)
    return sample, manifest


def anonymize(frame: pd.DataFrame, salt: str) -> pd.DataFrame:
    out = frame.copy()
    drop = [col for col in out.columns if col.lower() in TEXT_COLUMNS or col == "_stratum"]
    out = out.drop(columns=drop, errors="ignore")
    for column in ID_COLUMNS:
        if column in out.columns:
            out[column] = [
                "" if pd.isna(value) else _hash(column, value, salt)
                for value in out[column]
            ]
    return out


def _hash(namespace: str, value: object, salt: str) -> str:
    message = f"{namespace}:{value}".encode("utf-8")
    return hmac.new(salt.encode("utf-8"), message, hashlib.sha256).hexdigest()[:16]


def _stratum_key(frame: pd.DataFrame) -> pd.Series:
    month = pd.to_datetime(frame["answer_creation_date"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    if "asi" in frame.columns:
        asi = pd.to_numeric(frame["asi"], errors="coerce")
        quartile = pd.qcut(asi, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
        quartile = quartile.astype(str)
    else:
        quartile = "no_asi"
    if "primary_tag" in frame.columns:
        top = set(frame["primary_tag"].value_counts().head(20).index.astype(str))
        tag = frame["primary_tag"].astype(str).where(frame["primary_tag"].astype(str).isin(top), "other")
    else:
        tag = "no_tag"
    return month.fillna("missing-month") + "|" + pd.Series(quartile, index=frame.index).fillna("missing-asi") + "|" + tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a stratified anonymized sample from the analytical file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fraction", type=float, default=SAMPLE_FRACTION)
    parser.add_argument("--seed", type=int, default=20221130)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    sample, manifest = draw_sample(frame, fraction=args.fraction, seed=args.seed, salt=args.salt)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "replication_sample.csv"
    sample.to_csv(sample_path, index=False)
    manifest.to_csv(out / "stratum_manifest.csv", index=False)
    meta = {
        "source_rows": int(len(frame)),
        "sample_rows": int(len(sample)),
        "fraction": args.fraction,
        "seed": args.seed,
        "anonymized": True,
        "text_fields_removed": sorted(TEXT_COLUMNS),
        "weighted_source_rows": float(sample["sampling_weight"].sum()) if len(sample) else 0,
    }
    (out / "sample_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
