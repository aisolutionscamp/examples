"""Rule-based entity extraction with spaCy: no model download needed.

A blank English pipeline gets an EntityRuler with patterns for dates, money
and e-mail addresses. Field extraction then looks at the words around each
entity ("Total", "Tax", "effective", ...). Set SPACY_MODEL=en_core_web_sm
(after installing it) to add the statistical NER for ORG/PERSON as well.
"""
import os
import re
from datetime import date, datetime
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
          "sept", "oct", "nov", "dec"]
CURRENCY = ["$", "€", "£", "EUR", "USD", "GBP", "CHF"]
PATTERNS = [
    {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dd"}]},
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"^\d{1,2}[./]\d{1,2}[./]\d{4}$"}}]},
    {"label": "DATE", "pattern": [{"LOWER": {"IN": MONTHS}}, {"SHAPE": {"IN": ["d", "dd"]}},
                                  {"ORTH": ",", "OP": "?"}, {"SHAPE": "dddd"}]},
    {"label": "DATE", "pattern": [{"SHAPE": {"IN": ["d", "dd"]}}, {"LOWER": {"IN": MONTHS}}, {"SHAPE": "dddd"}]},
    {"label": "MONEY", "pattern": [{"ORTH": {"IN": CURRENCY}}, {"LIKE_NUM": True}]},
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"ORTH": {"IN": CURRENCY}}]},
    {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"]


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    name = os.environ.get("SPACY_MODEL")
    nlp = spacy.load(name) if name else spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
    ruler.add_patterns(PATTERNS)
    return nlp


def parse_date(text: str, day_first: bool = True) -> date | None:
    clean = re.sub(r"\s+", " ", text.replace(",", "")).replace(" - ", "-").replace(" -", "-").strip()
    clean = clean.replace("Sept ", "Sep ")
    formats = DATE_FORMATS if day_first else [f.replace("%d/%m", "%m/%d") for f in DATE_FORMATS]
    for fmt in formats:
        try:
            return datetime.strptime(clean, fmt).date()
        except ValueError:
            continue
    return None


def parse_money(span: Span) -> float | None:
    number = next((t.text for t in span if t.like_num), None)
    if number is None:
        return None
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{2}", number):  # 1.234,56
        number = number.replace(".", "").replace(",", ".")
    return float(number.replace(",", ""))


def context(doc: Doc, span: Span, window: int = 5) -> str:
    """Lower-case words before the entity on the same line."""
    start = span.start
    while start > 0 and span.start - start < window and "\n" not in doc[start - 1].text_with_ws:
        start -= 1
    return doc[start:span.start].text.lower()


def token_after(doc: Doc, keywords: tuple[str, ...]) -> str | None:
    """Text glued together after e.g. 'Invoice No:' -> 'INV-2026-0042'."""
    for i, tok in enumerate(doc):
        if tok.lower_ in keywords:
            j = i + 1
            while j < len(doc) and (doc[j].lower_ in {"no", "nr", "number", "#", ":", "."} or doc[j].is_space):
                j += 1
            parts = []
            while j < len(doc) and not doc[j].is_space:
                parts.append(doc[j].text)
                if doc[j].whitespace_:
                    break
                j += 1
            if parts:
                return "".join(parts)
    return None


def extract_fields(text: str, doc_type: str) -> dict:
    doc = get_nlp()(text)
    dates = [(e, parse_date(e.text)) for e in doc.ents if e.label_ == "DATE"]
    money = [(e, parse_money(e)) for e in doc.ents if e.label_ == "MONEY"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def money_near(*words, exclude=()):
        hits = [v for e, v in money if any(w in context(doc, e) for w in words)
                and not any(x in context(doc, e) for x in exclude)]
        return hits[-1] if hits else None

    def date_near(*words):
        return next((v for e, v in dates if v and any(w in context(doc, e) for w in words)), None)

    fields = {"emails": [e.text for e in doc.ents if e.label_ == "EMAIL"],
              "dates": [v.isoformat() for _, v in dates if v],
              "amounts": [v for _, v in money if v is not None]}
    if doc_type in ("invoice", "receipt"):
        fields.update({
            "vendor": lines[0] if lines else None,
            "invoice_number": token_after(doc, ("invoice", "inv")),
            "invoice_date": date_near("date", "dated", "issued") or (dates[0][1] if dates else None),
            "due_date": date_near("due"),
            "subtotal": money_near("subtotal", "net"),
            "tax": money_near("tax", "vat"),
            "total": money_near("total", "amount due", "balance due", exclude=("subtotal",)),
        })
    elif doc_type == "contract":
        m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:,|\(|\.|\n)", text, re.IGNORECASE | re.DOTALL)
        fields.update({
            "parties": [m.group(1).strip(), m.group(2).strip()] if m else [],
            "effective_date": date_near("effective", "commence"),
            "end_date": date_near("until", "expire", "terminat"),
        })
    if "ner" in get_nlp().pipe_names:
        fields["organizations"] = sorted({e.text for e in doc.ents if e.label_ == "ORG"})
    return fields
