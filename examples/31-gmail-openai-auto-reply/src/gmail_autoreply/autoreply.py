"""Categorize unread Gmail messages with OpenAI and save reply drafts for review."""

import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from . import gmail

DEFAULT_MODEL = "gpt-6-luna"
PROCESSED_LABEL = "AI/processed"
# Unread mail in the inbox that this tool has not handled yet; no newsletters or promotions.
DEFAULT_QUERY = f'is:unread in:inbox -category:promotions -category:social -label:"{PROCESSED_LABEL}"'
NO_REPLY_CATEGORIES = {"newsletter", "spam", "personal", "automated"}

Category = Literal["pricing", "availability", "order_status", "support", "partnership",
                   "newsletter", "spam", "personal", "automated", "other"]


class Decision(BaseModel):
    category: Category
    needs_reply: bool = Field(description="False for newsletters, spam, automated notifications, personal mail")
    reply: str | None = Field(description="Reply text without subject line, or null if no reply is needed")
    reason: str = Field(description="One short sentence explaining the decision")


INSTRUCTIONS = """You help a small business answer its email inbox.
Categorize the email and, if it is a business inquiry, write a reply draft.
Rules:
- Use only facts from the business information below. Never invent prices, stock levels or dates.
  If the information is missing, say that you will check and get back to the sender.
- Reply in the language of the email, in a friendly and concise tone (under 120 words), signed with the business name.
- Do not reply to newsletters, spam, automated notifications or personal mail (needs_reply = false, reply = null).

Business information:
{business_info}"""


@dataclass
class Outcome:
    message_id: str
    subject: str
    category: str
    draft_id: str | None


def decide(client: OpenAI, email: gmail.Email, business_info: str, model: str = DEFAULT_MODEL) -> Decision:
    response = client.responses.parse(
        model=model,
        instructions=INSTRUCTIONS.format(business_info=business_info),
        input=f"From: {email.sender}\nSubject: {email.subject}\n\n{email.body[:8000]}",
        text_format=Decision,
    )
    decision = response.output_parsed
    if decision is None:
        raise RuntimeError(f"model returned no parsable decision for message {email.id}")
    if decision.category in NO_REPLY_CATEGORIES:  # enforce the rule even if the model drafted something
        decision.needs_reply, decision.reply = False, None
    return decision


def process_inbox(service, client: OpenAI, business_info: str, from_address: str,
                  query: str = DEFAULT_QUERY, max_messages: int = 20, dry_run: bool = False) -> list[Outcome]:
    """Handle each matching message once: label it, and create a draft when a reply is needed."""
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    processed_label = None if dry_run else gmail.ensure_label(service, PROCESSED_LABEL)
    outcomes = []
    for message_id in gmail.list_unread(service, query, max_messages):
        email = gmail.get_email(service, message_id)
        if email.sender_address.lower() == from_address.lower():
            continue  # never answer our own mail
        decision = decide(client, email, business_info, model)
        draft_id = None
        if not dry_run:
            if decision.needs_reply and decision.reply:
                draft_id = gmail.create_reply_draft(service, email, decision.reply, from_address)
            category_label = gmail.ensure_label(service, f"AI/{decision.category}")
            gmail.add_labels(service, message_id, [processed_label, category_label])
        outcomes.append(Outcome(message_id, email.subject, decision.category, draft_id))
    return outcomes
