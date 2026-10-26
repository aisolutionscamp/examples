"""Bayesian Knowledge Tracing (Corbett & Anderson, 1994).

Each skill has a probability that the learner has mastered it. After every answer:
1. Bayes update with slip (knows it, answers wrong) and guess (does not know it, answers right).
2. Learning transition: chance to learn the skill from this practice opportunity.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2    # P(L0): mastered before the first question
    p_learn: float = 0.15  # P(T): learned during one practice opportunity
    p_slip: float = 0.1    # P(S): wrong answer although mastered
    p_guess: float = 0.2   # P(G): right answer although not mastered

    def __post_init__(self):
        for name in ("p_init", "p_learn", "p_slip", "p_guess"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")
        if self.p_slip + self.p_guess >= 1.0:
            raise ValueError("p_slip + p_guess must be < 1, otherwise answers carry no information")


DEFAULT_PARAMS = BKTParams()


def update(p_mastery: float, correct: bool, params: BKTParams = DEFAULT_PARAMS) -> float:
    s, g = params.p_slip, params.p_guess
    if correct:
        posterior = p_mastery * (1 - s) / (p_mastery * (1 - s) + (1 - p_mastery) * g)
    else:
        posterior = p_mastery * s / (p_mastery * s + (1 - p_mastery) * (1 - g))
    return posterior + (1 - posterior) * params.p_learn


def p_correct(p_mastery: float, params: BKTParams = DEFAULT_PARAMS) -> float:
    """Predicted chance of a correct answer, useful to evaluate the parameters on real logs."""
    return p_mastery * (1 - params.p_slip) + (1 - p_mastery) * params.p_guess
