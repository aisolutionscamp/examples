"""Local answer checking: numbers and fractions are compared by value, words case-insensitively."""
import re
from fractions import Fraction

_MIXED = re.compile(r"^(-?)(\d+)\s+(\d+)\s*/\s*(\d+)$")  # "1 1/2"


def as_number(text: str) -> Fraction | None:
    text = text.strip()
    try:
        mixed = _MIXED.match(text)
        if mixed:
            value = int(mixed[2]) + Fraction(int(mixed[3]), int(mixed[4]))
            return -value if mixed[1] else value
        return Fraction(text.replace(" ", "").replace(",", "."))
    except (ValueError, ZeroDivisionError):
        return None


def normalize_words(text: str) -> str:
    return re.sub(r"[^\w/]+", " ", text.lower()).strip()


def is_correct(learner_answer: str, expected: str) -> bool:
    a, b = as_number(learner_answer), as_number(expected)
    if a is not None and b is not None:
        return a == b
    return bool(normalize_words(learner_answer)) and normalize_words(learner_answer) == normalize_words(expected)
