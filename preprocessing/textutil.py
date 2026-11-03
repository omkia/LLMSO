"""Text helpers used by cleaning, duplicate detection, and text controls."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_TOKEN = re.compile(r"[a-z0-9_+#]+", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")
_SENTENCE = re.compile(r"[.!?]+")
_TAG = re.compile(r"<([^>]+)>")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.code_chars = 0
        self.format_chars = 0
        self._code = 0
        self._format = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("code", "pre"):
            self._code += 1
        if tag in ("code", "pre", "ul", "ol", "li", "blockquote", "h1", "h2", "h3"):
            self._format += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("code", "pre") and self._code:
            self._code -= 1
        if tag in ("code", "pre", "ul", "ol", "li", "blockquote", "h1", "h2", "h3") and self._format:
            self._format -= 1

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        n = len(data)
        if self._code:
            self.code_chars += n
        if self._format:
            self.format_chars += n


def parse_html(raw: str) -> tuple[str, int, int]:
    parser = _Extractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return "", 0, 0
    return "".join(parser.parts), parser.code_chars, parser.format_chars


def plain_text(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    text, _, _ = parse_html(str(raw))
    return text


def text_features(raw: object) -> dict[str, float]:
    """Length, code share, and a formatting share.

    ``formatting_score`` is the share of characters inside code, list,
    quote, or heading markup. The manuscript names formatting as a
    control and does not publish this formula.
    """
    if raw is None or (isinstance(raw, float) and raw != raw):
        return {"answer_length": float("nan"), "code_ratio": float("nan"),
                "formatting_score": float("nan"), "flesch_kincaid": float("nan")}
    text, code_chars, format_chars = parse_html(str(raw))
    stripped = text.strip()
    length = float(len(stripped))
    denom = length if length else float("nan")
    readable = _without_code(str(raw))
    return {
        "answer_length": length,
        "code_ratio": (code_chars / denom) if denom == denom and denom else float("nan"),
        "formatting_score": (format_chars / denom) if denom == denom and denom else float("nan"),
        "flesch_kincaid": flesch_kincaid_grade(readable),
    }


def _without_code(raw: str) -> str:
    no_code = re.sub(r"<(pre|code)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text, _, _ = parse_html(no_code)
    return text


def flesch_kincaid_grade(text: str) -> float:
    """Flesch–Kincaid grade with a vowel-group syllable count.

    The manuscript names Flesch–Kincaid and does not name the implementation.
    """
    words = _WORD.findall(text or "")
    if not words:
        return float("nan")
    sentences = max(1, len(_SENTENCE.findall(text)))
    syllables = 0
    for word in words:
        groups = re.findall(r"[aeiouy]+", word.lower())
        syllables += max(1, len(groups))
    n_words = len(words)
    return 0.39 * (n_words / sentences) + 11.8 * (syllables / n_words) - 15.59


def token_set(raw: object) -> set[int]:
    text = plain_text(raw).lower()
    out: set[int] = set()
    for token in _TOKEN.findall(text):
        out.add(_fnv(token))
        if len(out) >= 5000:
            break
    return out


def _fnv(token: str) -> int:
    h = 2166136261
    for byte in token.encode("utf-8", errors="ignore"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def primary_tag(tags: object) -> str | None:
    if tags is None or (isinstance(tags, float) and tags != tags):
        return None
    match = _TAG.search(str(tags))
    return match.group(1).lower() if match else None


def tag_count(tags: object) -> int:
    if tags is None or (isinstance(tags, float) and tags != tags):
        return 0
    return len(_TAG.findall(str(tags)))


def tag_set(tags: object) -> set[str]:
    if tags is None or (isinstance(tags, float) and tags != tags):
        return set()
    return {item.lower() for item in _TAG.findall(str(tags))}
