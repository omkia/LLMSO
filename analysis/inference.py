"""Two-sided normal tests from a printed coefficient and its standard error.

The manuscript reports H1--H4 from t = estimate / SE, not from a separate
fitted p-value. Twelve-test Bonferroni values are that probability times 12.
"""

from __future__ import annotations

import math


def two_sided_normal_p(z: float) -> float:
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def normal_interval(estimate: float, std_error: float, z: float = 1.96) -> tuple[float, float]:
    width = z * float(std_error)
    return float(estimate) - width, float(estimate) + width


def bonferroni(p_value: float, n_tests: int) -> float:
    if n_tests < 1:
        raise ValueError("n_tests must be positive")
    return min(1.0, float(p_value) * n_tests)


def wald_row(estimate: float, std_error: float, n_tests: int = 12) -> dict[str, float]:
    if std_error <= 0:
        raise ValueError("std_error must be positive")
    z = float(estimate) / float(std_error)
    p_value = two_sided_normal_p(z)
    low, high = normal_interval(estimate, std_error)
    return {
        "estimate": float(estimate),
        "std_error": float(std_error),
        "z": z,
        "p_value": p_value,
        "ci_low": low,
        "ci_high": high,
        "bonferroni": bonferroni(p_value, n_tests),
        "n_tests": n_tests,
    }
