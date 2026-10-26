"""Business checks on extracted data. Anything doubtful goes to a person."""
from datetime import date

REQUIRED = {"invoice": ["vendor_name", "invoice_id", "invoice_date", "total"],
            "contract": ["parties", "effective_date"], "general": []}


def validate(record: dict, min_confidence: float = 0.8, tolerance: float = 0.02) -> dict:
    fields, conf = record["fields"], record["confidence"]
    errors, warnings = [], []
    for name in REQUIRED[record["document_type"]]:
        if fields.get(name) in (None, "", []):
            errors.append(f"missing {name}")
    low = sorted(name for name, c in conf.items() if c < min_confidence)
    if low:
        warnings.append("low confidence: " + ", ".join(low))

    if record["document_type"] == "invoice":
        sub, tax, total = fields.get("subtotal"), fields.get("tax"), fields.get("total")
        if None not in (sub, tax, total) and abs(sub + tax - total) > tolerance:
            errors.append(f"subtotal {sub} + tax {tax} != total {total}")
        amounts = [i["amount"] for i in fields.get("items", []) if i.get("amount") is not None]
        if amounts and sub is not None and abs(sum(amounts) - sub) > tolerance:
            warnings.append(f"line items sum to {sum(amounts):.2f}, subtotal is {sub}")
        d1, d2 = fields.get("invoice_date"), fields.get("due_date")
        if isinstance(d1, date) and isinstance(d2, date) and d2 < d1:
            errors.append("due date before invoice date")
    if record["document_type"] == "contract":
        d1, d2 = fields.get("effective_date"), fields.get("expiration_date")
        if isinstance(d1, date) and isinstance(d2, date) and d2 <= d1:
            errors.append("expiration date not after effective date")

    status = "invalid" if errors else "requires_review" if warnings else "valid"
    return {"status": status, "errors": errors, "warnings": warnings}
