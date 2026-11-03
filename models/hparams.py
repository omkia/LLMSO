"""Shared checks for the classifier scripts."""

from __future__ import annotations

import json
from pathlib import Path

FORBIDDEN_ANSWER_FEATURES = {
    "net_vote", "score", "accepted", "accepted_answer_id", "reputation",
    "reputation_pre", "status_pre", "post", "ai_flagged", "comment_count",
}


def require_hparams(path: str, section: str) -> dict:
    block = json.loads(Path(path).read_text(encoding="utf-8"))[section]
    missing = [key for key, value in block.items() if value is None]
    if missing:
        raise SystemExit(
            "The manuscript does not state these settings. "
            f"Fill {path} section '{section}' before training: " + ", ".join(missing)
        )
    return block


def assert_text_only(columns: list[str]) -> None:
    used = FORBIDDEN_ANSWER_FEATURES.intersection(columns)
    if used:
        raise SystemExit("ICI/ECI training cannot use outcome or status columns: " + ", ".join(sorted(used)))
