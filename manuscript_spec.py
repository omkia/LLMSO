"""Quantities printed in the manuscript.

Only values stated in cas-sc-sample.tex belong here. Optimization settings,
the ICI/ECI encoder, the ASI feature list, the future-edit horizon, and the
Copilot tag list are not stated there and are not invented here.
"""

WINDOW_START = "2019-01-01"
WINDOW_END_EXCLUSIVE = "2026-01-01"
POST_START = "2023-01-01"
LLM_PUBLIC = "2022-11-30"
COPILOT_GA = "2022-06-21"
# Event-study index: k = -1 is November 2022 and is omitted.
# k = 0 is December 2022. It is plotted and included in the joint test
# from k = 0 through k = +12. The baseline Post indicator is separate
# and starts on POST_START (January 2023, k >= 1).
EVENT_K0_YEAR = 2022
EVENT_K0_MONTH = 12
EVENT_OMITTED_K = -1
EVENT_PLOT_K_MIN = -6
EVENT_PLOT_K_MAX = 13
EVENT_POST_TEST_K_MAX = 12
# Short label for the frozen pre-break ICI/ECI scoring model.
# Not a cryptographic file hash.
SCORE_MODEL_LABEL = "a3f9c21e"

BURST_ANSWERS = 50
BURST_DISTINCT_DAYS = 3
ACCOUNT_JACCARD = 0.95
ACCOUNT_PAIR_SHARE = 0.80
ACCOUNT_CREATION_SECONDS = 90
MIN_AUTOMATION_CRITERIA = 2

SAME_QUESTION_JACCARD = 0.98

DETECTOR_CHECKPOINT = "microsoft/deberta-v3-large"
DETECTOR_TRANSFORMERS = "4.38.1"
DETECTOR_THRESHOLD = 0.72
DETECTOR_HUMAN_N = 9000
DETECTOR_SYNTHETIC_N = 9000
DETECTOR_VALIDATION_N = 3000

SUPPLY_ICI_MIN = 0.70
SUPPLY_ECI_MAX = 0.30

SAMPLE_FRACTION = 0.05
N_ORDINAL_CATEGORIES = 5
FULL_SAMPLE_N = 4_215_082
TRIM_TOP1_N = 4_172_931
POST_PERIOD_ANSWERS = 1_712_539
PRE_PERIOD_ANSWERS = 2_502_543
N_RATERS = 3
RATER_OVERLAP = 0.15
ICI_ALPHA = 0.79
ECI_ALPHA = 0.84
ASI_ICC = 0.88
ASI_EXPERT_R = 0.76
FROZEN_CORRECTNESS_R = 0.58
EXPERT_QUESTIONS_N = 600
POSTPERIOD_AUDIT_N = 500
POSTPERIOD_KAPPA = 0.78
# 0.34 * 0.036 = 0.01224, printed as approximately 0.0122.
ICI_IQR = 0.34
ICI_IQR_CONTRAST_SE = 0.01224
# Supply-adjusted informational coefficient printed in the manuscript.
SUPPLY_POST_X_ICI = -0.077
SUPPLY_POST_X_ICI_SE = 0.038

# Printed full-sample ledger, used only as an audit checklist.
SAMPLE_FLOW = (
    ("raw", 5847312, None),
    ("outside_window", 4956067, 891245),
    ("empty_or_unreadable", 4837664, 118403),
    ("automated_accounts", 4790446, 47218),
    ("same_question_near_duplicates", 4777599, 12847),
    ("complete_case", 4215082, 562517),
)
