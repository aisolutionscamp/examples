"""Breached-password check with the Have I Been Pwned "Pwned Passwords" range API.

k-anonymity: only the first 5 hex characters of the SHA-1 hash leave the machine. The API returns
all hash suffixes with that prefix, and the comparison happens locally.
Docs: https://haveibeenpwned.com/API/v3#PwnedPasswords
"""
import hashlib

import httpx

API_URL = "https://api.pwnedpasswords.com/range/"


class PwnedPasswordsClient:
    def __init__(self, client: httpx.Client | None = None, timeout: float = 5.0):
        self._client = client or httpx.Client(timeout=timeout)

    def pwned_count(self, password: str) -> int:
        """How often the password appears in known breaches (0 = not found)."""
        # SHA-1 is what the Pwned Passwords range API is keyed on; it is used for a
        # k-anonymity lookup (only the first 5 hex chars are sent), not for storage.
        digest = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
        prefix, suffix = digest[:5], digest[5:]
        response = self._client.get(
            API_URL + prefix,
            # Padding adds fake entries (count 0) so the response size does not leak the prefix.
            headers={"Add-Padding": "true", "User-Agent": "aisolutionscamp-password-example"},
        )
        response.raise_for_status()
        for line in response.text.splitlines():
            candidate, _, count = line.strip().partition(":")
            if candidate == suffix:
                return int(count)
        return 0
