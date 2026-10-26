"""Text extraction -> classification -> entity extraction -> validation."""
from pathlib import Path

from .classify import classify
from .entities import extract_fields
from .text import extract_text
from .validate import validate


def process_document(path: str | Path, lang: str = "eng") -> dict:
    text, method = extract_text(path, lang=lang)
    doc_type, confidence = classify(text)
    fields = extract_fields(text, doc_type)
    return {
        "document_type": doc_type,
        "classifier_confidence": confidence,
        "text_method": method,
        "fields": fields,
        "validation": validate(doc_type, fields, confidence),
        "text": text,
    }
