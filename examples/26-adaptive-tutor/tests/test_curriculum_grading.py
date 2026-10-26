from pathlib import Path

import pytest

from adaptive_tutor.curriculum import Curriculum
from adaptive_tutor.grading import as_number, is_correct

FRACTIONS = Path(__file__).parents[1] / "curricula" / "fractions.json"


def test_first_skill_has_no_prerequisites():
    c = Curriculum.load(FRACTIONS)
    assert c.next_skill({}).id == "frac_concept"


def test_prerequisites_gate_skills():
    c = Curriculum.load(FRACTIONS)
    mastery = {"frac_concept": 0.97, "frac_equivalent": 0.5, "frac_add_same": 0.3}
    # frac_compare/multiply need frac_equivalent; frac_add_diff needs both -> lowest of the ready ones
    assert c.next_skill(mastery).id == "frac_add_same"
    mastery.update(frac_equivalent=0.96, frac_add_same=0.99)
    assert c.next_skill(mastery).id == "frac_compare"
    assert c.next_skill({s: 0.99 for s in c.skills}) is None


def test_unknown_prerequisite_is_rejected(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text('{"subject": "x", "skills": [{"id": "a", "name": "A", "prerequisites": ["b"]}]}')
    with pytest.raises(ValueError):
        Curriculum.load(bad)


@pytest.mark.parametrize("given, expected, ok", [
    ("3/4", "0.75", True), ("6/8", "3/4", True), ("0,75", "3/4", True), ("1 1/2", "3/2", True),
    ("-1 1/2", "-1.5", True), ("2/3", "3/4", False), ("Numerator", "numerator", True),
    ("denominator.", "numerator", False), ("", "", False), ("1/0", "0", False),
])
def test_is_correct(given, expected, ok):
    assert is_correct(given, expected) is ok


def test_as_number_rejects_text():
    assert as_number("three") is None
