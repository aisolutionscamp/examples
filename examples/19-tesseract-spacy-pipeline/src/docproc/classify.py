"""Document type classifier: TF-IDF + logistic regression.

The bundled examples only make the pipeline run. Train on a few hundred of
your own labelled documents with train_classifier() for real use.
"""
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

SEED_EXAMPLES = {
    "invoice": [
        "Invoice No INV-1001 invoice date due date bill to subtotal VAT total amount due payment terms",
        "INVOICE number date of issue customer item quantity unit price amount subtotal tax total due",
        "Tax invoice vendor address invoice number net amount VAT 19% gross total please pay within 30 days",
        "Invoice billing address order number line items qty price total balance due bank transfer IBAN",
    ],
    "contract": [
        "This Agreement is made between the parties effective date term termination governing law signature",
        "Service agreement between provider and client obligations confidentiality liability term of contract",
        "Contract parties hereby agree terms and conditions effective as of renewal termination notice signed",
        "Lease agreement landlord tenant term rent clauses obligations governing law signatures witness",
    ],
    "receipt": [
        "Receipt store cashier thank you for your purchase card payment change total paid",
        "Sales receipt items cash paid change VAT included thank you visit again",
        "Receipt transaction id terminal card ending approved total EUR thank you",
        "Store receipt date time items total paid by card no refunds without receipt",
    ],
    "report": [
        "Quarterly report summary results analysis revenue growth outlook conclusions recommendations",
        "Annual report executive summary key figures charts discussion risks outlook",
        "Project status report milestones progress risks next steps recommendations",
        "Analysis report methodology findings tables figures conclusion appendix",
    ],
}


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
                          LogisticRegression(max_iter=1000, C=10))
    return model.fit(texts, labels)


@lru_cache(maxsize=1)
def default_classifier() -> Pipeline:
    texts = [t for examples in SEED_EXAMPLES.values() for t in examples]
    labels = [label for label, examples in SEED_EXAMPLES.items() for _ in examples]
    return train_classifier(texts, labels)


def classify(text: str, model: Pipeline | None = None, min_confidence: float = 0.4) -> tuple[str, float]:
    model = model or default_classifier()
    probs = model.predict_proba([text])[0]
    best = int(probs.argmax())
    label = str(model.classes_[best]) if probs[best] >= min_confidence else "unknown"
    return label, round(float(probs[best]), 3)
