"""Emergency-contact escalation.

This is NOT a replacement for emergency services. It sends a message to a person the user
chose (family, carer) and always tells the user to call the local emergency number.
Detection is a deterministic phrase match that runs before the LLM, so it does not depend
on the model understanding the situation. It will miss some phrasings and trigger on others;
a physical help button is still needed.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

import httpx2

EMERGENCY_PATTERNS = [
    r"\bhelp me\b", r"\bi (have |'ve )?fallen\b", r"\bi fell\b", r"\bcan(\s?no|')?t get up\b",
    r"\bchest pain\b", r"\bcan(\s?no|')?t breathe\b", r"\bcall (an )?ambulance\b", r"\bemergency\b",
    r"\bi'?m bleeding\b", r"\bstroke\b",
]
_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)


def is_emergency(text: str) -> bool:
    return bool(_RE.search(text))


class EmergencyEscalation:
    def __init__(self, webhook_url: str, contact_name: str, emergency_number: str,
                 post: Callable[..., httpx2.Response] = httpx2.post):
        self.webhook_url = webhook_url
        self.contact_name = contact_name
        self.emergency_number = emergency_number
        self._post = post

    def escalate(self, what_was_said: str, now: datetime) -> str:
        """Notify the contact and return the sentence the assistant says out loud."""
        sent = False
        if self.webhook_url:
            try:
                resp = self._post(self.webhook_url, timeout=10, json={
                    "event": "possible_emergency",
                    "time": now.isoformat(timespec="minutes"),
                    "transcript": what_was_said,
                })
                sent = resp.status_code < 300
            except httpx2.HTTPError:
                sent = False
        call = f"If you are hurt or in danger, call {self.emergency_number} now."
        if not self.webhook_url:
            return f"No emergency contact is set up. {call}"
        if sent:
            return f"I have sent a message to {self.contact_name}. {call}"
        return f"I could not reach {self.contact_name}. {call}"
