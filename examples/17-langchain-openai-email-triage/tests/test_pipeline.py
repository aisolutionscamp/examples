from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from email_triage import Classification, EmailTriage
from email_triage.zendesk import to_zendesk_ticket

POLICIES = ["Password reset: use https://app.example.com/reset.", "Refunds within 30 days of delivery."]
VOCAB = ["password", "reset", "login", "refund", "refunds", "delivery", "invoice", "invoices"]


class KeywordEmbeddings(Embeddings):
    """Offline stand-in for OpenAIEmbeddings: one dimension per keyword."""

    def embed_query(self, text):
        words = text.lower().replace(".", " ").replace(":", " ").split()
        return [float(words.count(v)) + 1e-6 for v in VOCAB]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def make_triage(classification: Classification, seen: list):
    def fake_classifier(prompt_value):
        seen.append(prompt_value.to_string())
        return classification

    fake_llm = FakeListChatModel(responses=["Hi Jane, you can reset your password at https://app.example.com/reset."])

    def record_then_answer(prompt_value):
        seen.append(prompt_value.to_string())
        return prompt_value

    writer = RunnableLambda(record_then_answer) | fake_llm
    kb = InMemoryVectorStore.from_texts(POLICIES, KeywordEmbeddings())
    return EmailTriage(RunnableLambda(fake_classifier), writer, kb, min_confidence=0.7, top_k=1)


def c(**overrides) -> Classification:
    base = {"urgency": "MEDIUM", "department": "GENERAL", "intent": "account_access", "language": "en",
            "confidence": 0.9, "reasoning": "login problem"}
    return Classification(**(base | overrides))


def test_confident_routine_email_gets_draft():
    seen = []
    result = make_triage(c(), seen).process("From: a@example.com\nSubject: Login\n\nI forgot my password.")
    assert not result.escalate
    assert "reset your password" in result.draft_reply
    # The cleaned email reached the classification prompt together with the system rules.
    assert "I forgot my password." in seen[0] and "URGENCY: HIGH" in seen[0]
    # The retriever picked the password policy (top_k=1), not the refund policy.
    assert "Password reset" in seen[1] and "Refunds" not in seen[1]


def test_low_confidence_escalates_without_draft():
    result = make_triage(c(confidence=0.4), []).process("hm?")
    assert result.escalate and result.draft_reply is None
    assert result.escalation_reason.startswith("low confidence")


def test_high_urgency_and_refund_escalate():
    t = make_triage(c(urgency="HIGH", intent="outage", department="TECHNICAL"), [])
    assert t.process("Everything is down").escalation_reason == "high urgency"
    t = make_triage(c(intent="refund_request", department="BILLING"), [])
    assert "refund_request" in t.process("Refund order AB-123456 please").escalation_reason


def test_zendesk_payload():
    result = make_triage(c(urgency="LOW"), []).process("Where are my invoices?")
    ticket = to_zendesk_ticket(result, "Invoices", "a@example.com")["ticket"]
    assert ticket["priority"] == "low"
    assert "ai_draft" in ticket["tags"] and "dept_general" in ticket["tags"]
    assert ticket["comment"]["public"] is False
