"""Detector performance on the 1,000-row human validation file.

Columns required: reference_positive (1 if the majority vote is the
positive AI class used for the printed sensitivity) and score.
The operating threshold printed in the manuscript is 0.72.
"""

from __future__ import annotations

import argparse

import pandas as pd

from manuscript_spec import DETECTOR_THRESHOLD
from models.metrics import binary_report, youden_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation")
    parser.add_argument("--threshold", type=float, default=DETECTOR_THRESHOLD)
    parser.add_argument("--select-youden", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.validation)
    y = frame["reference_positive"].to_numpy()
    scores = frame["score"].to_numpy()
    threshold = youden_threshold(y, scores) if args.select_youden else args.threshold
    report = binary_report(y, scores, threshold)
    for key, value in report.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
