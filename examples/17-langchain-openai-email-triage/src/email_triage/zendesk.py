"""Map a triage result to a Zendesk ticket payload (POST /api/v2/tickets.json)."""

from .pipeline import TriageResult

PRIORITY = {"HIGH": "urgent", "MEDIUM": "normal", "LOW": "low"}


def to_zendesk_ticket(result: TriageResult, subject: str, requester_email: str) -> dict:
    c = result.classification
    comment = result.draft_reply or f"Escalated: {result.escalation_reason}"
    return {
        "ticket": {
            "subject": subject or "(no subject)",
            "requester": {"email": requester_email},
            "priority": PRIORITY[c.urgency],
            "tags": [f"dept_{c.department.lower()}", f"intent_{c.intent}", f"lang_{c.language}"]
                    + (["needs_human"] if result.escalate else ["ai_draft"]),
            # Internal note: the agent reviews the draft before anything goes to the customer.
            "comment": {"body": comment, "public": False},
        }
    }
