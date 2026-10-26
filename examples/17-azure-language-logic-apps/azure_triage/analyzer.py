"""Analyze a support email with Azure AI Language and turn the result into routing decisions."""

import os
import re
from dataclasses import asdict, dataclass

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

SUPPORTED_LANGUAGES = {"en", "es", "de"}

# Category keywords per language. Matched against key phrases and the raw text (lowercase).
CATEGORY_KEYWORDS = {
    "billing": ["invoice", "refund", "charge", "payment", "price", "billing",
                "factura", "reembolso", "pago", "cobro",
                "rechnung", "erstattung", "zahlung", "abbuchung", "preis"],
    "technical": ["error", "bug", "crash", "api", "outage", "down", "not working", "login",
                  "fallo", "no funciona", "caído",
                  "fehler", "absturz", "funktioniert nicht", "ausfall", "anmeldung"],
    "sales": ["upgrade", "enterprise", "quote", "demo", "license",
              "presupuesto", "licencia", "angebot", "lizenz"],
}
URGENT_PATTERNS = re.compile(
    r"\b(urgent|asap|immediately|outage|down|security|hacked|urgente|inmediatamente|dringend|sofort|ausfall)\b",
    re.IGNORECASE)
ESCALATE_NEGATIVE_SCORE = 0.8


@dataclass
class EmailAnalysis:
    language: str
    sentiment: str
    negative_score: float
    key_phrases: list[str]
    entities: list[dict]
    category: str
    urgency: str
    escalate: bool
    escalation_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def make_client() -> TextAnalyticsClient:
    """Client for an Azure AI Language resource (endpoint + key from the environment)."""
    return TextAnalyticsClient(endpoint=os.environ["AZURE_LANGUAGE_ENDPOINT"],
                               credential=AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]))


def _first(results, what: str):
    result = results[0]
    if result.is_error:
        raise RuntimeError(f"Azure AI Language {what} failed: {result.error.code} {result.error.message}")
    return result


def classify_category(text: str, key_phrases: list[str]) -> str:
    haystack = " ".join([text, *key_phrases]).lower()
    scores = {cat: sum(len(re.findall(rf"\b{re.escape(w)}\b", haystack)) for w in words)
              for cat, words in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def classify_urgency(text: str, sentiment: str, negative_score: float, category: str) -> str:
    if URGENT_PATTERNS.search(text):
        return "high"
    if sentiment == "negative" and negative_score >= 0.6:
        return "medium" if category == "general" else "high"
    if category in {"billing", "technical"}:
        return "medium"
    return "low"


def analyze_email(client: TextAnalyticsClient, text: str) -> EmailAnalysis:
    """Run language detection, sentiment, entities and key phrases, then apply the routing rules."""
    docs = [text[:5000]]  # synchronous API limit is 5,120 characters per document
    lang = _first(client.detect_language(docs), "language detection").primary_language.iso6391_name
    lang_for_calls = lang if lang in SUPPORTED_LANGUAGES else "en"

    sentiment = _first(client.analyze_sentiment(docs, language=lang_for_calls), "sentiment")
    entities = _first(client.recognize_entities(docs, language=lang_for_calls), "entities")
    phrases = _first(client.extract_key_phrases(docs, language=lang_for_calls), "key phrases")

    key_phrases = list(phrases.key_phrases)
    negative = float(sentiment.confidence_scores.negative)
    category = classify_category(text, key_phrases)
    urgency = classify_urgency(text, sentiment.sentiment, negative, category)

    reason = None
    if urgency == "high":
        reason = "high urgency"
    elif negative >= ESCALATE_NEGATIVE_SCORE:
        reason = f"very negative sentiment ({negative:.2f})"
    elif lang not in SUPPORTED_LANGUAGES:
        reason = f"unsupported language '{lang}'"

    return EmailAnalysis(
        language=lang,
        sentiment=sentiment.sentiment,
        negative_score=round(negative, 3),
        key_phrases=key_phrases,
        entities=[{"text": e.text, "category": e.category, "confidence": round(e.confidence_score, 3)}
                  for e in entities.entities],
        category=category,
        urgency=urgency,
        escalate=reason is not None,
        escalation_reason=reason,
    )
