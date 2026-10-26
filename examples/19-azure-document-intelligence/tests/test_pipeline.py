import copy
import json
from datetime import date
from pathlib import Path

import pytest
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult

from doc_pipeline import pipeline
from doc_pipeline.extract import extract_contract, extract_general, extract_invoice, field_value
from doc_pipeline.routing import route
from doc_pipeline.validate import validate

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FakePoller:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class FakeClient:
    """Records calls like DocumentIntelligenceClient.begin_analyze_document and returns a fixture."""

    def __init__(self, payload: dict):
        self.payload, self.calls = payload, []

    def begin_analyze_document(self, model_id, body):
        self.calls.append((model_id, body))
        return FakePoller(AnalyzeResult(self.payload))


class FakeHttp:
    def __init__(self):
        self.posts = []

    def post(self, url, data, headers, timeout):
        self.posts.append((url, json.loads(data), headers))
        return type("R", (), {"raise_for_status": lambda self: None})()


def test_invoice_fields_are_typed():
    rec = extract_invoice(AnalyzeResult(load("invoice")))
    f = rec["fields"]
    assert f["vendor_name"] == "Example Corp" and f["invoice_id"] == "INV-1042"
    assert f["invoice_date"] == date(2026, 10, 1) and f["total"] == 14280.0 and f["currency"] == "EUR"
    assert f["vendor_address"] == "1 Example Street, Springfield"
    assert f["items"][0] == {"description": "Consulting days", "quantity": 10, "unit_price": 1000.0,
                             "amount": 10000.0}
    assert rec["confidence"]["customer_name"] == pytest.approx(0.62)


def test_camel_case_field_types_resolve():
    doc = AnalyzeResult(load("invoice")).documents[0]
    assert field_value(doc.fields["VendorPhone"]) == "+15555550100"


def test_validation_flags_low_confidence_and_bad_totals():
    rec = extract_invoice(AnalyzeResult(load("invoice")))
    v = validate(rec)
    assert v["status"] == "requires_review" and "customer_name" in v["warnings"][0]
    broken = copy.deepcopy(rec)
    broken["fields"]["total"] = 9999.0
    broken["fields"]["due_date"] = date(2026, 9, 1)
    v = validate(broken)
    assert v["status"] == "invalid" and len(v["errors"]) == 2
    missing = copy.deepcopy(rec)
    missing["fields"]["invoice_id"] = None
    assert "missing invoice_id" in validate(missing)["errors"]


def test_routing_by_amount_and_review():
    rec = extract_invoice(AnalyzeResult(load("invoice")))
    wf = route(rec, {"status": "valid"})
    assert wf["steps"] == ["finance_manager", "cfo"] and wf["priority"] == "high"
    assert all(a.endswith("@example.com") for a in wf["approvers"])
    small = copy.deepcopy(rec)
    small["fields"]["total"] = 120.0
    assert route(small, {"status": "requires_review"})["steps"] == ["document_review", "accounts_payable"]


def test_contract_and_layout_extraction():
    c = extract_contract(AnalyzeResult(load("contract")))
    assert c["fields"]["parties"] == ["Example Corp", "Sample Supplier GmbH"]
    assert validate(c)["status"] == "valid"
    assert route(c, validate(c))["steps"] == ["legal_review"]
    g = extract_general(AnalyzeResult(load("layout")))
    assert g["fields"]["pages"] == 2 and g["fields"]["tables"] == [{"rows": 3, "columns": 2}]


def test_process_local_file_writes_audit_and_calls_webhook(tmp_path):
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake")
    client, http = FakeClient(load("invoice")), FakeHttp()
    out = pipeline.process(client, str(pdf), "invoice", webhook_url="https://example.com/hook",
                           audit_path=tmp_path / "audit.jsonl", http=http)
    model_id, body = client.calls[0]
    assert model_id == "prebuilt-invoice" and hasattr(body, "read")
    assert out["workflow"]["steps"][0] == "document_review"
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert audit[0]["document_id"] == out["document_id"]
    url, payload, headers = http.posts[0]
    assert url == "https://example.com/hook" and payload["fields"]["total"] == 14280.0
    assert headers["Content-Type"] == "application/json"
    assert "items" not in payload["fields"]


def test_process_url_uses_url_source(tmp_path):
    client = FakeClient(load("contract"))
    pipeline.process(client, "https://example.com/contract.pdf", "contract", audit_path=tmp_path / "a.jsonl")
    model_id, body = client.calls[0]
    assert model_id == "prebuilt-contract"
    assert isinstance(body, AnalyzeDocumentRequest) and body.url_source == "https://example.com/contract.pdf"
    with pytest.raises(ValueError):
        pipeline.process(client, "x.pdf", "resume")


def test_make_client_uses_key_when_set(monkeypatch):
    monkeypatch.setenv("DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example-resource.cognitiveservices.azure.com/")
    monkeypatch.setenv("DOCUMENT_INTELLIGENCE_KEY", "not-a-real-key")
    assert isinstance(pipeline.make_client(), DocumentIntelligenceClient)
