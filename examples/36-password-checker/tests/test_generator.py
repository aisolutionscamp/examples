import math

import pytest

from password_tool.generator import (
    DEFAULT_ALPHABET,
    entropy_bits,
    generate_passphrase,
    generate_password,
    load_wordlist,
)


def test_wordlist_is_the_full_eff_list():
    words = load_wordlist()
    assert len(words) == 7776
    assert math.isclose(entropy_bits(1, len(words)), 12.92, abs_tol=0.01)


def test_password_has_requested_length_and_alphabet():
    pw = generate_password(24)
    assert len(pw) == 24
    assert set(pw) <= set(DEFAULT_ALPHABET)


def test_passwords_are_not_repeated():
    assert len({generate_password(20) for _ in range(200)}) == 200


def test_passphrase_uses_words_from_list():
    words = load_wordlist()
    phrase = generate_passphrase(6, separator=" ")
    parts = phrase.split(" ")
    assert len(parts) == 6
    assert all(p in words for p in parts)


def test_six_words_give_more_than_77_bits():
    assert entropy_bits(6, 7776) > 77


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate_password(0)
    with pytest.raises(ValueError):
        generate_password(10, alphabet="aab")
    with pytest.raises(ValueError):
        generate_passphrase(0)
