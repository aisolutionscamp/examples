"""Email cleanup and rule-based extraction that runs before the LLM call."""

import re
from dataclasses import dataclass, field
from email import message_from_string, policy

ORDER_RE = re.compile(r"\b(?:order|ord)\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Z0-9]{2,}-?\d{4,})\b", re.IGNORECASE)
ACCOUNT_RE = re.compile(r"\baccount\s*(?:id|no\.?|number|#)?\s*[:#]?\s*([A-Z0-9-]{5,})\b", re.IGNORECASE)
ERROR_RE = re.compile(r"\b(?:error|err|code)\s*[:#]?\s*([A-Z]{0,4}-?\d{3,5})\b", re.IGNORECASE)
QUOTE_START_RE = re.compile(r"^(On .+ wrote:|-{2,}\s*Original Message\s*-{2,}|From: .+)$", re.IGNORECASE)
SIGNATURE_RE = re.compile(r"^(--\s*|Sent from my .+)$")
MAX_CHARS = 6000


@dataclass
class ParsedEmail:
    subject: str
    sender: str
    body: str
    order_numbers: list[str] = field(default_factory=list)
    account_ids: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)

    def as_prompt_text(self) -> str:
        return f"Subject: {self.subject}\nFrom: {self.sender}\n\n{self.body}"


def strip_quotes_and_signature(body: str) -> str:
    """Drop quoted earlier messages and the signature block."""
    kept = []
    for line in body.splitlines():
        stripped = line.strip()
        if QUOTE_START_RE.match(stripped) or SIGNATURE_RE.match(stripped):
            break
        if stripped.startswith(">"):
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", text)[:MAX_CHARS]


def _unique(matches: list[str]) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in matches))


def parse_email(raw: str) -> ParsedEmail:
    """Parse an RFC 822 message (or plain text) and extract identifiers."""
    if re.match(r"^[A-Za-z-]+: ", raw):
        msg = message_from_string(raw, policy=policy.default)
        part = msg.get_body(preferencelist=("plain",)) if msg.is_multipart() else msg
        body = part.get_content() if part is not None else ""
        subject, sender = str(msg.get("Subject", "")), str(msg.get("From", ""))
    else:
        body, subject, sender = raw, "", ""
    body = strip_quotes_and_signature(body)
    text = f"{subject}\n{body}"
    return ParsedEmail(
        subject=subject,
        sender=sender,
        body=body,
        order_numbers=_unique(ORDER_RE.findall(text)),
        account_ids=_unique(ACCOUNT_RE.findall(text)),
        error_codes=_unique(ERROR_RE.findall(text)),
    )
