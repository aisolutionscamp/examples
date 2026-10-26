"""Check a password the way NIST SP 800-63B (rev. 4) asks verifiers to.

- Length is what counts: at least 15 characters for a password used on its own, at least 8 when it
  is only one factor of MFA. Accept at least 64 characters, spaces and Unicode.
- No composition rules ("one uppercase, one digit, ...").
- Reject passwords that appear in breach corpora or block lists.
- Strength estimate with zxcvbn (pattern matching on dictionaries, keyboard walks, dates,
  repeats, l33t substitutions) to explain *why* a password is weak.
"""
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from zxcvbn import zxcvbn

MAX_LENGTH = 128          # NIST: permit at least 64
ZXCVBN_MAX_INPUT = 100    # zxcvbn gets slow on very long input; it only advises, it never truncates the check
MIN_ZXCVBN_SCORE = 3      # 0..4; 3 = "safely unguessable: moderate protection from offline attacks"


class BreachChecker(Protocol):
    def pwned_count(self, password: str) -> int: ...


@dataclass
class CheckResult:
    accepted: bool
    length: int
    score: int | None = None
    guesses_log10: float | None = None
    pwned_count: int | None = None
    problems: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def check_password(
    password: str,
    *,
    min_length: int = 15,
    user_inputs: list[str] | None = None,
    breach_checker: BreachChecker | None = None,
) -> CheckResult:
    """Return a verdict with reasons. `user_inputs` are words tied to the user (name, email, site)."""
    password = unicodedata.normalize("NFC", password)
    length = len(password)  # code points, as NIST asks
    result = CheckResult(accepted=False, length=length)

    if length < min_length:
        result.problems.append(f"Too short: {length} characters, at least {min_length} needed.")
    if length > MAX_LENGTH:
        result.problems.append(f"Too long: at most {MAX_LENGTH} characters are accepted.")

    if breach_checker is not None:
        result.pwned_count = breach_checker.pwned_count(password)
        if result.pwned_count:
            result.problems.append(
                f"Found {result.pwned_count} times in known data breaches. Choose a different password."
            )

    estimate = zxcvbn(password[:ZXCVBN_MAX_INPUT], user_inputs=user_inputs or [])
    result.score = estimate["score"]
    result.guesses_log10 = round(float(estimate["guesses_log10"]), 1)
    if estimate["score"] < MIN_ZXCVBN_SCORE:
        warning = estimate["feedback"]["warning"] or "Easy to guess."
        result.problems.append(f"Guessable pattern (strength {estimate['score']}/4): {warning}")
    result.suggestions = list(estimate["feedback"]["suggestions"])
    if length < min_length and not result.suggestions:
        result.suggestions.append("Use a passphrase of several random words.")

    result.accepted = not result.problems
    return result
