# Replication package

This directory implements the sample rules, annotation checks, classifier
constraints, and estimators printed in `cas-sc-sample.tex`.

The raw Stack Overflow extract is not in this repository. Stack Overflow
content in the public data dump is available under CC BY-SA. The full
analytical file is available on request. `replication_sample/draw_stratified_sample.py`
draws the stratified 5 percent sample named in the manuscript, removes
post text and display names, and replaces identifiers with salted hashes.
Sampling noise prevents a 5 percent draw from reproducing the exact
three-decimal point estimates in the tables. The estimators fit the file
they are given. They do not overwrite coefficients or covariance matrices
with the printed tables. Printed standard errors, including 0.036 for H1,
are full-sample covariance standard errors when those matrices are
supplied. They are not the standard errors of the 5 percent fit.
`analysis/manuscript_targets.json` is the audit checklist, including the
H1--H4 standard errors, the trim and ordered-logit targets, and the
supply-adjusted informational coefficient -0.077 (SE 0.038). Inference
for a printed coefficient uses the two-sided normal probability of
estimate / SE. The H1 checklist value is t = -0.153 / 0.036 = -4.25,
reported p = 0.00002, and a twelve-test Bonferroni value of 0.00024.
The informational interquartile standard error is 0.34 * 0.036 = 0.01224,
printed as approximately 0.0122. The supply coefficient retains
0.077 / 0.153 = 0.503 of the baseline effect.

The event-study clock sets k = -1 to November 2022 and omits it. k = 0 is
December 2022. That month is plotted and included in the joint test from
k = 0 through k = +12. The baseline Post indicator starts on 1 January
2023. The two-event LLM set is {-6, ..., -2, 0, ..., +12}, eighteen
coefficients. Joint F output reports the numerator degrees of freedom,
F, and a cluster-robust p-value. It does not print a Satterthwaite
denominator. The identifier a3f9c21e is a short label for the frozen
pre-break scoring model, not a cryptographic file hash. ASI validation
in the manuscript is the expert-rating correlation r = 0.76, not a
binary rate on the AI-text detector labels.

## Order of operations

1. Restore a Stack Exchange dump and run `data_extraction/01_extract_answers.sql` and `02_duplicate_links.sql`. The SEDE website caps results at 50,000 rows, so it cannot return the 5,847,312-row extract. `00_count_answers_by_year.sql` is the aggregate check.
2. `python preprocessing/run_pipeline.py --answers answers.csv --duplicates links.csv --out-dir outputs/clean`
3. Code the annotation file using `annotation/codebook.md`. Compute alpha and assign splits.
4. Fill `models/hparams.template.json`. The manuscript names the detector checkpoint and the 0.72 threshold. It does not name the ICI/ECI encoder, the ASI learner, or the optimization settings.
5. Score ICI, ECI, and ASI, merge them onto the cleaned answers, and rerun the cleaner with `--complete-case`.
6. `python replication_sample/draw_stratified_sample.py --input cleaned_answers.csv --out-dir replication_sample/out --salt <private salt>`
7. `python analysis/run_tables.py --sample replication_sample/out/replication_sample.csv --out-dir outputs/tables`

That command writes the valuation fit, `valuation_inference.csv` (normal-tail p-values and Bonferroni factors of 4, 8, and 12), the top-1% trim, the five-category ordered outcome, and the six hypothetical deleted-answer bounds. The ordered outcome is negative, 0, 1, 2-4, and 5 or higher. Ordered-logit coefficients, when statsmodels is installed, are written by `analysis/outcome_robustness.py` and are on the latent index. An imputation refit of the deleted-answer scenarios is written separately and is not the manuscript table.

R and Stata specifications for the same answer-level equations are in `analysis/r/valuation.R` and `analysis/stata/valuation.do`. Acceptance is a logit there, and nonnegative counts use Poisson pseudo-maximum likelihood. The Python tables for the net vote are ordinary least squares with contributor, primary-tag, and month fixed effects, clustered by primary tag.

## What is not invented

The Copilot language list, the future-edit horizon, the survival coding rule, the five quality-item stems, and the numeric annotation cut are not printed in the manuscript. The corresponding inputs are arguments or blank fields. The pipeline writes the sample-flow counts it observes and places the printed counts beside them.
