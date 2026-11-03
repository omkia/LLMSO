"""Top-1% trim and five-category ordered logit for the valuation specification.

The ordered outcome uses five vote categories: negative, 0, 1, 2-4, and
5 or higher. Ordered-logit coefficients are on the latent index and are
not in net-vote units. The script writes each fitted coefficient vector
and covariance. It does not replace those fits with the printed table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.fit_tables import valuation
from manuscript_spec import N_ORDINAL_CATEGORIES


# Edges give five categories: [-inf, 0), [0, 1), [1, 2), [2, 5), [5, inf).
BINS = [-np.inf, 0, 1, 2, 5, np.inf]


def trim_top_percent(frame: pd.DataFrame, percent: float = 1.0) -> pd.DataFrame:
    cutoff = float(np.nanpercentile(frame["net_vote"].to_numpy(dtype=float), 100 - percent))
    kept = frame.loc[frame["net_vote"] <= cutoff].copy()
    kept.attrs["trim_cutoff"] = cutoff
    return kept


def ordinal_bins(scores: pd.Series) -> pd.Series:
    coded = pd.cut(scores, bins=BINS, right=False, labels=False).astype(float)
    if coded.nunique(dropna=True) > N_ORDINAL_CATEGORIES:
        raise ValueError("ordinal coding produced more than five categories")
    return coded


def fit_ordered_linear(frame: pd.DataFrame):
    from analysis.design import LOWER_ORDER, add_top_tag_indicators, prepare
    from analysis.hdfe import fit_hdfe

    ordered = frame.copy()
    ordered["net_vote_bin"] = ordinal_bins(ordered["net_vote"])
    prepared = prepare(ordered.dropna(subset=["net_vote_bin"]))
    controls = add_top_tag_indicators(prepared)
    return fit_hdfe(
        prepared,
        "net_vote_bin",
        list(LOWER_ORDER) + controls,
        ["user_id", "primary_tag", "answer_month"],
        "primary_tag",
        "sampling_weight",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    from analysis.run_tables import save_result

    frame = pd.read_csv(args.sample)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trimmed = trim_top_percent(frame, 1.0)
    save_result(valuation(trimmed), out / "valuation_trim_top1")
    (out / "valuation_trim_top1_cutoff.txt").write_text(
        f"dropped net_vote above {trimmed.attrs['trim_cutoff']}\n",
        encoding="utf-8",
    )
    save_result(fit_ordered_linear(frame), out / "valuation_ordinal_bins_linear")
    from analysis.design import LOWER_ORDER, add_top_tag_indicators, prepare

    ordered = frame.copy()
    ordered["net_vote_bin"] = ordinal_bins(ordered["net_vote"])
    prepared = prepare(ordered.dropna(subset=["net_vote_bin"]))
    controls = add_top_tag_indicators(prepared)
    try:
        _ordered_logit(prepared, list(LOWER_ORDER) + controls, out / "valuation_ordinal_logit")
    except Exception as exc:
        (out / "valuation_ordinal_logit_NOTE.txt").write_text(
            "Ordinal logit was not fit in this run: "
            + str(exc)
            + "\nThe binned linear covariance is valuation_ordinal_bins_linear.npz.\n",
            encoding="utf-8",
        )


def _ordered_logit(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    y = frame["net_vote_bin"].astype(int)
    x = frame[columns].astype(float)
    # Month, tag, and contributor dummies are not saturated here: with
    # hundreds of thousands of rows those fixed effects belong in a
    # specialized ordinal estimator. The exported linear-bin model carries
    # the fixed effects. This logit is the latent-index companion without
    # claiming those dummies were absorbed.
    model = OrderedModel(y, x, distr="logit")
    fitted = model.fit(method="bfgs", disp=False, maxiter=100)
    names = list(columns)
    beta = np.asarray(fitted.params)[: len(names)]
    cov = np.asarray(fitted.cov_params())[: len(names), : len(names)]
    pd.DataFrame({"term": names, "estimate": beta, "std_error": np.sqrt(np.diag(cov))}).to_csv(
        path.with_suffix(".csv"), index=False
    )
    np.savez(path.with_suffix(".npz"), beta=beta, cov=cov)


if __name__ == "__main__":
    main()
