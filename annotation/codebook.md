# Annotation codebook

This codebook records the coding rules printed in the manuscript and the
item slots implied by the construct definitions. Where the manuscript
does not print the administered wording, the stem is left blank for the
wording that was actually shown to coders.

## Unit and blinding

The unit is one Stack Overflow answer. Coders do not see net votes,
acceptance, author reputation, or the calendar date. Near-duplicate
answers stay in the same train, validation, or test partition
(`annotation/splits.py`).

The measurement sample named in the manuscript has 2,500 items and carries
three latent factors: epistemic contribution, informational usefulness,
and overall answer quality. ICI and ECI are coded by three independent
raters, with a 15 percent overlap for reliability calibration. Quality is
five 7-point Likert items, rated independently, with reported
Krippendorff alpha 0.87. ECI alpha is 0.84 and ICI alpha is 0.79. Those
alphas are computed by `annotation/krippendorff_alpha.py`; the manuscript
does not state whether the difference function was nominal, ordinal, or
interval. A separate check of 500 random post-period answers is reported
at Cohen's kappa 0.78.

Expert ASI ratings have an intraclass correlation of 0.88. The
historical-feature classifier's out-of-sample correlation with those
ratings is 0.76. The manuscript does not report binary ASI rates on
the detector's AI-text labels. Detector sensitivity, specificity,
precision, and F1 apply only to that detector. The
frozen-model correctness correlation on 600 expert-rated questions is
0.58.

## Quadrant anchors

These are the anchors in the manuscript's coding-anchor table.

| Pattern | Code | Illustrative form |
| --- | --- | --- |
| High ICI, low ECI | Direct command, code fragment, or standard procedure without visible tests, assumptions, or limits | "Use function X with argument Y." |
| Low ICI, high ECI | Diagnoses uncertainty, compares explanations, or specifies tests and boundary conditions without a complete drop-in solution | "This depends on the runtime version; test A before choosing between B and C." |
| High ICI, high ECI | Usable solution whose warrant, scope, and failure conditions are inspectable | "Use X; the example verifies condition Y, but it fails under Z and requires version V." |
| Low ICI, low ECI | Neither a usable solution nor inspectable reasoning | Unsupported assertion, social comment, or restatement of the question |

## Informational items

Score each item from 1 to 7. The binary training target is not a
hand-weighted sum of surface features. `models/train_ici_eci.py` sets
the positive class when the mean of these items is at least the cut in
`hparams.json`. That cut is not printed in the manuscript.

| Column | Dimension named in the manuscript | Administered stem |
| --- | --- | --- |
| `ici_solution_directness` | Solution directness | |
| `ici_code_or_procedure` | Code or procedure provision | |
| `ici_routine_structure` | Routine structure | |

## Epistemic items

ECI records visible warranting. It is not correctness, usefulness, or
pre-period reputation.

| Column | Dimension named in the manuscript | Administered stem |
| --- | --- | --- |
| `eci_verification` | Verification | |
| `eci_evidence` | Evidence | |
| `eci_explanation` | Explanation | |
| `eci_boundary` | Boundary conditions | |
| `eci_alternatives` | Alternatives | |
| `eci_uncertainty` | Uncertainty | |
| `eci_provenance` | Provenance | |

## Overall quality

Five 7-point items, rated independently of the ICI and ECI items. The
manuscript does not print the five stems.

| Column | Administered stem |
| --- | --- |
| `quality_1` | |
| `quality_2` | |
| `quality_3` | |
| `quality_4` | |
| `quality_5` | |

## AI-text validation labels

On the separate 1,000-answer detector sample, three annotators use one of
three labels: predominantly human, AI-assisted editing, or predominantly
AI-generated. Annotators are blinded to detector scores. The validation
reference is the majority vote.

## ASI expert rating

On the 600-question panel, the expert rating is whether a useful candidate
answer could be produced from the question as written. Raters do not see
model output. The classifier target is the mean expert rating, later
scaled to [0, 1]. The raw rating scale is not printed; pass its maximum
to `models/train_asi.py` as `--rating-max`.
