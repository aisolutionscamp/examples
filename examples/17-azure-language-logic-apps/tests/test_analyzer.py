import pytest
from fakes import FakeChatClient, FakeLanguageClient

from azure_triage import analyze_email, draft_reply
from azure_triage.analyzer import classify_category


def test_billing_question_is_routine():
    client = FakeLanguageClient(phrases=["invoice", "March"], entities=[("March", "DateTime")])
    a = analyze_email(client, "Where can I download the invoice for March?")
    assert (a.category, a.urgency, a.escalate) == ("billing", "medium", False)
    assert a.entities == [{"text": "March", "category": "DateTime", "confidence": 0.9}]
    # Detected language is passed to the follow-up calls.
    assert ("analyze_sentiment", {"language": "en"}) in client.calls


def test_outage_is_high_urgency_and_escalated():
    a = analyze_email(FakeLanguageClient(sentiment="negative", negative=0.7), "Your API is down since 9:00")
    assert a.category == "technical" and a.urgency == "high"
    assert a.escalate and a.escalation_reason == "high urgency"


def test_german_and_spanish_keywords():
    assert classify_category("Die Rechnung ist doppelt abgebucht, bitte Erstattung", []) == "billing"
    assert classify_category("La aplicación no funciona desde ayer", []) == "technical"
    assert classify_category("Can I get a quote for an enterprise license?", []) == "sales"
    assert classify_category("Thanks for the quick help", []) == "general"
    # Word boundaries: "rapid" must not count as "api".
    assert classify_category("rapid delivery", []) == "general"


def test_very_negative_general_email_escalates():
    a = analyze_email(FakeLanguageClient(sentiment="negative", negative=0.92), "I am really disappointed with you")
    assert a.urgency == "medium" and a.escalation_reason.startswith("very negative")


def test_unsupported_language_falls_back_and_escalates():
    client = FakeLanguageClient(language="fr")
    a = analyze_email(client, "Bonjour, une question")
    assert a.escalate and "unsupported language" in a.escalation_reason
    assert ("extract_key_phrases", {"language": "en"}) in client.calls


def test_service_error_raises():
    with pytest.raises(RuntimeError, match="InvalidDocument"):
        analyze_email(FakeLanguageClient(error="InvalidDocument"), "text")


def test_draft_only_for_routine_emails(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "support-drafts")
    chat = FakeChatClient("Hello, here is how to download it.")
    routine = analyze_email(FakeLanguageClient(language="de"), "Wo finde ich meine Rechnung?")
    assert draft_reply(chat, "Rechnung", "Wo finde ich meine Rechnung?", routine) == "Hello, here is how to download it."
    assert chat.requests[0]["model"] == "support-drafts"
    assert "ISO code de" in chat.requests[0]["messages"][0]["content"]

    urgent = analyze_email(FakeLanguageClient(), "URGENT: security problem")
    assert draft_reply(chat, "", "URGENT", urgent) is None
    assert draft_reply(None, "", "x", routine) is None
