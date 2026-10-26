"""Completeness and plausibility checks on extracted fields."""

REQUIRED = {"invoice": ["vendor", "invoice_number", "invoice_date", "total"],
            "receipt": ["total"], "contract": ["parties", "effective_date"], "report": [], "unknown": []}


def validate(doc_type: str, fields: dict, classifier_confidence: float) -> dict:
    errors = [f"missing {name}" for name in REQUIRED.get(doc_type, []) if fields.get(name) in (None, "", [])]
    warnings = []
    if classifier_confidence < 0.6:
        warnings.append(f"document type uncertain ({classifier_confidence:.2f})")
    sub, tax, total = fields.get("subtotal"), fields.get("tax"), fields.get("total")
    if None not in (sub, tax, total) and abs(sub + tax - total) > 0.02:
        errors.append(f"subtotal {sub} + tax {tax} != total {total}")
    d1, d2 = fields.get("invoice_date"), fields.get("due_date")
    if d1 and d2 and d2 < d1:
        errors.append("due date before invoice date")
    status = "invalid" if errors else "requires_review" if warnings or doc_type == "unknown" else "valid"
    return {"status": status, "errors": errors, "warnings": warnings}
