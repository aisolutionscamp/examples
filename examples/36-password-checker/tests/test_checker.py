import unicodedata

from password_tool.checker import check_password


class FakeBreaches:
    def __init__(self, counts=None):
        self.counts = counts or {}
        self.calls = []

    def pwned_count(self, password):
        self.calls.append(password)
        return self.counts.get(password, 0)


def test_short_password_is_rejected():
    result = check_password("Xk7#qL9!", breach_checker=FakeBreaches())
    assert not result.accepted
    assert any("Too short" in p for p in result.problems)


def test_same_password_ok_as_mfa_factor_if_not_guessable():
    result = check_password("vT9#qmZ2!rWx", min_length=8)
    assert result.accepted, result.problems


def test_no_composition_rules_for_long_random_words():
    # only lowercase letters and spaces: fine under NIST, length does the work
    result = check_password("ripple banjo cactus mural tundra", breach_checker=FakeBreaches())
    assert result.accepted, result.problems
    assert result.score >= 3


def test_long_but_predictable_password_is_rejected():
    result = check_password("passwordpassword123")
    assert not result.accepted
    assert result.score < 3


def test_breached_password_is_rejected_even_if_long():
    pw = "ripple banjo cactus mural tundra"
    result = check_password(pw, breach_checker=FakeBreaches({pw: 12}))
    assert not result.accepted
    assert result.pwned_count == 12
    assert any("breaches" in p for p in result.problems)


def test_user_inputs_lower_the_score():
    pw = "aliceexample2026!!"
    plain = check_password(pw)
    personal = check_password(pw, user_inputs=["alice", "example"])
    assert personal.guesses_log10 < plain.guesses_log10


def test_unicode_is_normalized_and_counted_by_code_point():
    decomposed = unicodedata.normalize("NFD", "café crème brûlée façade")
    breaches = FakeBreaches()
    result = check_password(decomposed, breach_checker=breaches)
    assert breaches.calls == [unicodedata.normalize("NFC", decomposed)]
    assert result.length == len("café crème brûlée façade")
