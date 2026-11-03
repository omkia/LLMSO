"""Held-out ICI and ECI discrimination within primary-tag families.

The manuscript compares families that differ in surface syntax, including
Python and JavaScript. This script writes recall, precision, and F1 for
each requested family. It does not fill the manuscript with a number the
prediction file does not contain.
"""

from __future__ import annotations

import argparse

import pandas as pd

from models.metrics import binary_report


def family_report(frame: pd.DataFrame, families: list[str], threshold: float) -> pd.DataFrame:
    rows = []
    tags = frame["primary_tag"].astype(str).str.lower()
    for family in families:
        part = frame.loc[tags == family.lower()]
        if part.empty:
            rows.append({"primary_tag": family, "n": 0})
            continue
        metrics = binary_report(
            part["reference_positive"].to_numpy(),
            part["score"].to_numpy(),
            threshold,
        )
        metrics["primary_tag"] = family
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ICI or ECI within tag families.")
    parser.add_argument("predictions", help="CSV with primary_tag, reference_positive, and score.")
    parser.add_argument("--families", default="python,javascript")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    family_report(frame, families, args.threshold).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
