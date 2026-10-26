"""Random passwords and passphrases from the operating system's CSPRNG (`secrets`)."""
import math
import secrets
import string
from functools import lru_cache
from pathlib import Path

WORDLIST = Path(__file__).with_name("eff_large_wordlist.txt")
DEFAULT_ALPHABET = string.ascii_letters + string.digits + "-_.!@#%+=?"


@lru_cache(maxsize=1)
def load_wordlist(path: Path = WORDLIST) -> tuple[str, ...]:
    words = tuple(w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())
    if len(set(words)) != len(words):
        raise ValueError("wordlist contains duplicates, entropy would be overstated")
    return words


def generate_password(length: int = 20, alphabet: str = DEFAULT_ALPHABET) -> str:
    """Uniformly random characters. No composition rules: every character is drawn independently."""
    if length < 1:
        raise ValueError("length must be positive")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet contains duplicate characters")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_passphrase(words: int = 6, separator: str = "-", wordlist: tuple[str, ...] | None = None) -> str:
    """Diceware-style passphrase, easier to type and remember than random characters."""
    if words < 1:
        raise ValueError("words must be positive")
    pool = wordlist or load_wordlist()
    return separator.join(secrets.choice(pool) for _ in range(words))


def entropy_bits(symbols: int, pool_size: int) -> float:
    """Entropy of a secret made of `symbols` independent uniform draws from `pool_size` options."""
    return symbols * math.log2(pool_size)
