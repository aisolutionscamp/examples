"""Turn an Azure AI Document Intelligence AnalyzeResult into flat, typed records."""
import re
from typing import Any

from azure.ai.documentintelligence.models import AnalyzeResult, DocumentField

MODELS = {"invoice": "prebuilt-invoice",
          "contract": "prebuilt-contract", "general": "prebuilt-layout"}


def field_value(field: DocumentField | None) -> Any:
    """Typed value of a field (currency -> amount, address -> text, arrays/objects recursively)."""
    if field is None:
        return None
    kind = field.type
    if kind == "currency" and field.value_currency is not None:
        return field.value_currency.amount
    if kind == "address":
        return field.content
    if kind == "array":
        return [field_value(f) for f in field.value_array or []]
    if kind == "object":
        return {k: field_value(v) for k, v in (field.value_object or {}).items()}
    # "phoneNumber" -> value_phone_number, "date" -> value_date, ...
    attr = "value_" + re.sub(r"(?<!^)(?=[A-Z])", "_", kind).lower() if kind else ""
    value = getattr(field, attr, None) if attr else None
    return value if value is not None else field.content


def _fields(result: AnalyzeResult, keys: dict[str, str]) -> tuple[dict, dict]:
    doc = (result.documents or [None])[0]
    found = doc.fields if doc and doc.fields else {}
    data = {name: field_value(found.get(key)) for name, key in keys.items()}
    confidence = {name: found[key].confidence for name, key in keys.items()
                  if key in found and found[key].confidence is not None}
    return data, confidence


def extract_invoice(result: AnalyzeResult) -> dict:
    data, conf = _fields(result, {
        "vendor_name": "VendorName", "vendor_address": "VendorAddress", "customer_name": "CustomerName",
        "invoice_id": "InvoiceId", "invoice_date": "InvoiceDate", "due_date": "DueDate",
        "subtotal": "SubTotal", "tax": "TotalTax", "total": "InvoiceTotal", "items": "Items"})
    doc = (result.documents or [None])[0]
    total = doc.fields.get("InvoiceTotal") if doc and doc.fields else None
    data["currency"] = total.value_currency.currency_code if total and total.value_currency else None
    data["items"] = [{"description": i.get("Description"), "quantity": i.get("Quantity"),
                      "unit_price": i.get("UnitPrice"), "amount": i.get("Amount")}
                     for i in data["items"] or []]
    return {"document_type": "invoice", "fields": data, "confidence": conf}


def extract_contract(result: AnalyzeResult) -> dict:
    data, conf = _fields(result, {
        "title": "Title", "contract_id": "ContractId", "parties": "Parties",
        "execution_date": "ExecutionDate", "effective_date": "EffectiveDate",
        "expiration_date": "ExpirationDate", "renewal_date": "RenewalDate"})
    data["parties"] = [p.get("Name") if isinstance(p, dict) else p for p in data["parties"] or []]
    return {"document_type": "contract", "fields": data, "confidence": conf}


def extract_general(result: AnalyzeResult) -> dict:
    tables = [{"rows": t.row_count, "columns": t.column_count} for t in result.tables or []]
    return {"document_type": "general", "confidence": {},
            "fields": {"pages": len(result.pages or []), "tables": tables, "text": result.content or ""}}


EXTRACTORS = {"invoice": extract_invoice,
              "contract": extract_contract, "general": extract_general}
