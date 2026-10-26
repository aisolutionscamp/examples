# Password generator and checker (NIST SP 800-63B)

Generates passwords and passphrases with Python's `secrets` module and checks existing passwords the way
[NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b.html) (revision 4, section 3.1.1.2) asks
password verifiers to:

- **Length over composition.** At least 15 characters for a password used on its own, at least 8 if it is
  only one factor of multi-factor authentication. Spaces and Unicode are allowed (normalized to NFC,
  counted per code point). No "one uppercase, one digit, one symbol" rules.
- **Breached passwords are rejected.** Lookup in the Have I Been Pwned *Pwned Passwords* corpus through the
  k-anonymity range API: only the first 5 hex characters of the SHA-1 hash are sent.
- **Strength estimate with explanations.** [zxcvbn](https://github.com/dwolfhub/zxcvbn-python) finds
  dictionary words, keyboard walks, dates, repeats and l33t substitutions and estimates the number of
  guesses. Passwords below score 3 of 4 are rejected, with zxcvbn's warning and suggestions.

No machine-learning model is used or needed: the checks are deterministic, and zxcvbn's pattern matching
explains its result, which a trained classifier would not.

## Architecture

```
password ──► NFC normalize ──► length check ─┐
                              ├─► SHA-1 ──► first 5 chars ──► api.pwnedpasswords.com/range/XXXXX
                              │                               (suffix compared locally)
                              └─► zxcvbn (patterns, guesses) ┴─► verdict + reasons
secrets.choice ──► EFF large wordlist (7,776 words) ──► passphrase
```

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m password_tool generate            # 6-word passphrase, about 77 bits
python -m password_tool generate --chars 20 # random characters
python -m password_tool check               # hidden prompt, online breach lookup
python -m password_tool check --offline --mfa --user-input alice
```

`check` prints JSON with `accepted`, `length`, zxcvbn `score` (0 to 4), `guesses_log10`, `pwned_count`,
`problems` and `suggestions`. No API key is needed for the Pwned Passwords range API.

Tests (offline, the HIBP API is mocked with `httpx.MockTransport`): `pytest`.

## Configuration

- `check_password(min_length=15 | 8, user_inputs=[...], breach_checker=...)`: `user_inputs` are words tied to
  the account (name, email, site name); zxcvbn penalizes them.
- `MIN_ZXCVBN_SCORE` in `checker.py` (default 3).
- `generate_password(alphabet=...)` if a target system restricts characters.

## Limitations

- zxcvbn's dictionaries are English-centric; the estimate is a guide, not a guarantee.
- zxcvbn only analyzes the first 100 characters (for speed). The length and breach checks always use the
  whole password.
- The breach check needs network access; with `--offline` only length and zxcvbn are applied.
- The wordlist is the EFF large wordlist (CC BY 3.0 US, https://www.eff.org/dice).

## For production

- Run the check server-side at registration and password change, and store passwords only as salted,
  slow hashes (Argon2id, scrypt or bcrypt).
- Rate-limit failed logins instead of forcing periodic password changes (NIST: do not require them).
- For high volume or no outbound traffic, download the Pwned Passwords hash list and query it locally.
- Add your own block list (product name, company name) via `user_inputs` or a separate set lookup.
