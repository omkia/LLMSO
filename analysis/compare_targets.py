"""Compare estimated coefficients with the values printed in the manuscript.

The printed values are an audit checklist. This script does not adjust
the sample, the coefficients, or the covariance matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimates", required=True, help="CSV with columns term, estimate, std_error")
    parser.add_argument("--targets", default=str(Path(__file__).with_name("manuscript_targets.json")))
    parser.add_argument("--table", required=True, help="Key inside the targets file")
    args = parser.parse_args()
    estimates = pd.read_csv(args.estimates)
    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))[args.table]
    rows = []
    lookup = estimates.set_index("term")
    for item in targets:
        term = item["term"]
        if term not in lookup.index:
            rows.append({**item, "estimated": None, "ratio": None})
            continue
        estimated = float(lookup.loc[term, "estimate"])
        printed = float(item["printed"])
        estimated_se = None
        se_ratio = None
        if "std_error" in lookup.columns and item.get("printed_se") not in (None, ""):
            estimated_se = float(lookup.loc[term, "std_error"])
            printed_se = float(item["printed_se"])
            se_ratio = estimated_se / printed_se if printed_se else None
        rows.append({
            **item,
            "estimated": estimated,
            "ratio_to_printed": estimated / printed if printed else None,
            "estimated_se": estimated_se,
            "se_ratio_to_printed": se_ratio,
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
