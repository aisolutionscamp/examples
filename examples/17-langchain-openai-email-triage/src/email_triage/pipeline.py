"""Classify a support email with LangChain + OpenAI, then draft a reply or escalate."""

import os
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from .preprocess import ParsedEmail, parse_email

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class Classification(BaseModel):
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    department: Literal["TECHNICAL", "BILLING", "SALES", "GENERAL"]
    intent: Literal["outage", "bug_report", "refund_request", "billing_question",
                    "account_access", "product_question", "feature_request", "other"]
    language: str = Field(description="ISO 639-1 code of the email language, e.g. en, es, de")
    confidence: float = Field(ge=0, le=1, description="Your confidence in this classification")
    reasoning: str = Field(description="One sentence explaining the decision")


CLASSIFY_SYSTEM = (
    "You triage customer support emails.\n"
    "URGENCY: HIGH = service outage, security issue, failed payment; "
    "MEDIUM = account access, billing dispute, bug; LOW = feature request, general question.\n"
    "DEPARTMENT: TECHNICAL = bugs, integrations, API errors; BILLING = payments, invoices, refunds, pricing; "
    "SALES = upgrades, new features, enterprise; GENERAL = everything else.\n"
    "Be honest with the confidence value: use < 0.6 when the email is ambiguous."
)
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([("system", CLASSIFY_SYSTEM), ("human", "{email}")])

REPLY_SYSTEM = (
    "You write reply drafts for support agents. Answer in the customer's language ({language}). "
    "Use only facts from the policy excerpts. If they do not answer the question, say an agent will follow up. "
    "Do not promise refunds or dates that the policies do not state. Keep it under 150 words."
)
REPLY_HUMAN = (
    "Policy excerpts:\n{policies}\n\nClassification: {department} / {intent}\n"
    "Known identifiers: {identifiers}\n\nCustomer email:\n{email}"
)
REPLY_PROMPT = ChatPromptTemplate.from_messages([("system", REPLY_SYSTEM), ("human", REPLY_HUMAN)])

ESCALATE_INTENTS = {"outage", "refund_request"}


class TriageResult(BaseModel):
    classification: Classification
    order_numbers: list[str]
    error_codes: list[str]
    escalate: bool
    escalation_reason: str | None
    draft_reply: str | None


class EmailTriage:
    def __init__(self, classifier: Runnable, writer: Runnable, knowledge_base: VectorStore,
                 min_confidence: float = 0.7, top_k: int = 3):
        self.classify_chain = CLASSIFY_PROMPT | classifier
        self.reply_chain = REPLY_PROMPT | writer | StrOutputParser()
        self.knowledge_base = knowledge_base
        self.min_confidence = min_confidence
        self.top_k = top_k

    def escalation_reason(self, c: Classification) -> str | None:
        if c.confidence < self.min_confidence:
            return f"low confidence ({c.confidence:.2f})"
        if c.urgency == "HIGH":
            return "high urgency"
        if c.intent in ESCALATE_INTENTS:
            return f"intent '{c.intent}' needs a human decision"
        return None

    def process(self, raw_email: str) -> TriageResult:
        parsed: ParsedEmail = parse_email(raw_email)
        text = parsed.as_prompt_text()
        c: Classification = self.classify_chain.invoke({"email": text})
        reason = self.escalation_reason(c)

        draft = None
        if reason is None:
            docs = self.knowledge_base.similarity_search(text, k=self.top_k)
            draft = self.reply_chain.invoke({
                "language": c.language,
                "policies": "\n".join(f"- {d.page_content}" for d in docs) or "- (none found)",
                "department": c.department,
                "intent": c.intent,
                "identifiers": ", ".join(parsed.order_numbers + parsed.account_ids + parsed.error_codes) or "none",
                "email": text,
            })
        return TriageResult(classification=c, order_numbers=parsed.order_numbers,
                            error_codes=parsed.error_codes, escalate=reason is not None,
                            escalation_reason=reason, draft_reply=draft)


def from_env(policies: list[str]) -> EmailTriage:
    """Build the pipeline with real OpenAI models (needs OPENAI_API_KEY)."""
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    classifier = ChatOpenAI(model=model).with_structured_output(Classification)
    writer = ChatOpenAI(model=model)
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    kb = InMemoryVectorStore.from_texts(policies, embeddings)  # swap for PineconeVectorStore in production
    return EmailTriage(classifier, writer, kb, min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")))
