"""Write the estimated tables and their covariance matrices.

Definitional tables (notation, construct definitions, coding anchors) are
not estimated. Detector metrics and the three-factor measurement model
need the annotation files and are produced by models/detector_metrics.py
and annotation/cfa_three_factor.R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.deleted_scenarios import fit_deleted_scenarios, scenario_table
from analysis.design import VALUATION_FOCAL
from analysis.fit_tables import (
    baseline,
    descriptives,
    event_study,
    generalized_vif,
    outside_copilot_languages,
    prewindow,
    sensitivity,
    time_placebos,
    two_event,
    valuation,
    with_supply,
    with_tag_slope,
)
from analysis.inference import bonferroni
from analysis.outcome_robustness import fit_ordered_linear, trim_top_percent
from analysis.hdfe import OLSResult


def save_result(result: OLSResult, path: Path) -> None:
    result.as_frame().to_csv(path.with_suffix(".csv"), index=False)
    npz = {
        "names": result.names,
        "beta": result.beta,
        "std_error": result.se,
        "cov": result.cov,
        "n": result.n,
        "g": result.g,
        "rank": result.rank,
    }
    pd.Series(npz["names"]).to_csv(path.with_suffix(".names.csv"), index=False, header=["term"])
    # NumPy is imported locally so a missing binary extension fails at save time.
    import numpy as np
    np.savez(path.with_suffix(".npz"), **{key: value for key, value in npz.items() if key != "names"})
    meta = {"n": result.n, "clusters": result.g, "rank": result.rank, "absorbed": result.absorbed}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(frame: pd.DataFrame, out_dir: Path, languages: set[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptives(frame).to_csv(out_dir / "descriptives.csv", index=False)
    save_result(baseline(frame, "net_vote"), out_dir / "baseline_net_vote")
    for outcome in ("accepted", "future_edits", "survival"):
        if outcome in frame.columns:
            save_result(baseline(frame, outcome), out_dir / f"baseline_{outcome}_linear")
    valuation_result = valuation(frame)
    save_result(valuation_result, out_dir / "valuation")
    _focal_inference(valuation_result).to_csv(out_dir / "valuation_inference.csv", index=False)
    trimmed = trim_top_percent(frame, 1.0)
    save_result(valuation(trimmed), out_dir / "valuation_trim_top1")
    (out_dir / "valuation_trim_top1_cutoff.txt").write_text(
        f"dropped net_vote above {trimmed.attrs['trim_cutoff']}\n",
        encoding="utf-8",
    )
    save_result(fit_ordered_linear(frame), out_dir / "valuation_ordinal_bins_linear")
    scenario_table().to_csv(out_dir / "deleted_scenarios.csv", index=False)
    fit_deleted_scenarios(frame).to_csv(out_dir / "deleted_scenario_fits.csv", index=False)
    length = tuple(col for col in ("answer_length",) if col in frame.columns)
    if length:
        save_result(valuation(frame, extra=length), out_dir / "valuation_length_controls")
    text = tuple(col for col in ("answer_length", "flesch_kincaid", "code_ratio", "formatting_score") if col in frame.columns)
    if text:
        save_result(valuation(frame, extra=text), out_dir / "valuation_text_controls")
    event_result, event_tests = event_study(frame)
    save_result(event_result, out_dir / "event_study")
    pd.DataFrame(event_tests).to_csv(out_dir / "event_study_wald.csv", index=False)
    pre_result, pre_tests = prewindow(frame)
    save_result(pre_result, out_dir / "prewindow")
    pd.DataFrame(pre_tests).to_csv(out_dir / "prewindow_wald.csv", index=False)
    time_placebos(frame).to_csv(out_dir / "time_placebo.csv", index=False)
    sensitivity(frame).to_csv(out_dir / "sensitivity.csv", index=False)
    save_result(with_tag_slope(frame), out_dir / "tag_slope")
    save_result(with_supply(frame), out_dir / "supply")
    generalized_vif(frame).to_csv(out_dir / "vif.csv", index=False)
    two_result, two_tests = two_event(frame)
    save_result(two_result, out_dir / "two_event")
    pd.DataFrame(two_tests).to_csv(out_dir / "two_event_wald.csv", index=False)
    if languages:
        save_result(outside_copilot_languages(frame, languages), out_dir / "outside_copilot_languages")
    (out_dir / "ESTIMATOR.txt").write_text(
        "Net-vote estimates are HDFE-OLS with primary-tag clustered Liang-Zeger "
        "standard errors. p-values are the two-sided normal tail of estimate/SE. "
        "They are estimates from the file passed to this script. "
        "Joint F tests report numerator degrees of freedom, F, and the "
        "cluster-robust p-value. A Satterthwaite denominator is not printed. "
        "Deleted-answer rows in the manuscript are hypothetical bounds, not "
        "the imputation refit written beside them. "
        "Acceptance, counts, and the exit hazard are specified for R/Stata in "
        "analysis/r and analysis/stata because those families are not the linear "
        "within estimator used here. The linear files for those outcomes, when "
        "written, are labeled _linear.\n",
        encoding="utf-8",
    )


def _focal_inference(result: OLSResult) -> pd.DataFrame:
    table = result.as_frame().set_index("term")
    rows = []
    for term in VALUATION_FOCAL:
        if term not in table.index:
            continue
        p_value = float(table.loc[term, "p_value"])
        rows.append({
            "term": term,
            "estimate": float(table.loc[term, "estimate"]),
            "std_error": float(table.loc[term, "std_error"]),
            "z": float(table.loc[term, "z"]),
            "p_value": p_value,
            "bonferroni_4": bonferroni(p_value, 4),
            "bonferroni_8": bonferroni(p_value, 8),
            "bonferroni_12": bonferroni(p_value, 12),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate manuscript tables from the replication sample.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copilot-languages", help="Text file with one primary tag per line.")
    args = parser.parse_args()
    frame = pd.read_csv(args.sample)
    languages = None
    if args.copilot_languages:
        languages = {
            line.strip()
            for line in Path(args.copilot_languages).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    run(frame, Path(args.out_dir), languages)


if __name__ == "__main__":
    main()
