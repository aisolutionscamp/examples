import math

import pytest

from adaptive_tutor.bkt import BKTParams, p_correct, update

P = BKTParams(p_init=0.2, p_learn=0.15, p_slip=0.1, p_guess=0.2)


def test_correct_answer_matches_hand_calculation():
    # posterior = 0.2*0.9 / (0.2*0.9 + 0.8*0.2) = 0.18/0.34; then + (1-post)*0.15
    post = 0.18 / 0.34
    assert math.isclose(update(0.2, True, P), post + (1 - post) * 0.15)


def test_wrong_answer_matches_hand_calculation():
    post = 0.02 / (0.02 + 0.8 * 0.8)
    assert math.isclose(update(0.2, False, P), post + (1 - post) * 0.15)


def test_correct_raises_wrong_lowers():
    assert update(0.5, True, P) > 0.5
    assert update(0.9, False, P) < 0.9


def test_mastery_reached_after_a_few_correct_answers():
    p, steps = 0.2, 0
    while p < 0.95:
        p, steps = update(p, True, P), steps + 1
    assert steps == 3  # 0.2 -> 0.60 -> 0.89 -> 0.98


def test_p_correct():
    assert math.isclose(p_correct(0.5, P), 0.5 * 0.9 + 0.5 * 0.2)


def test_invalid_params():
    with pytest.raises(ValueError):
        BKTParams(p_slip=0.6, p_guess=0.5)
    with pytest.raises(ValueError):
        BKTParams(p_learn=1.5)
