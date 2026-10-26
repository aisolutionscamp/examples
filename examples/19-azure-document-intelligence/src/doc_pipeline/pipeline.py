"""Analyze a document with Azure AI Document Intelligence, validate it and route it.

    python -m doc_pipeline.pipeline invoice.pdf --type invoice
    python -m doc_pipeline.pipeline https://example.com/contract.pdf --type contract
"""
import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from .extract import EXTRACTORS, MODELS
from .routing import route
from .validate import validate


def make_client() -> DocumentIntelligenceClient:
    endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
    key = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")
    if key:
        return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    from azure.identity import DefaultAzureCredential  # Entra ID: managed identity, az login, ...
    return DocumentIntelligenceClient(endpoint, DefaultAzureCredential())


def analyze(client: DocumentIntelligenceClient, source: str, doc_type: str) -> AnalyzeResult:
    model_id = MODELS[doc_type]
    if source.startswith(("https://", "http://")):
        poller = client.begin_analyze_document(model_id, AnalyzeDocumentRequest(url_source=source))
    else:
        with open(source, "rb") as f:
            poller = client.begin_analyze_document(model_id, body=f)
    return poller.result()  # long-running operation: polls until the service is done


def process(client, source: str, doc_type: str, webhook_url: str | None = None,
            audit_path: str | Path = "audit.jsonl", http=requests) -> dict:
    if doc_type not in MODELS:
        raise ValueError(f"doc_type must be one of {sorted(MODELS)}")
    record = EXTRACTORS[doc_type](analyze(client, source, doc_type))
    validation = validate(record)
    output = {
        "document_id": str(uuid.uuid4()),
        "source": source,
        "model_id": MODELS[doc_type],
        "processed_at": datetime.now(UTC).isoformat(),
        **record,
        "validation": validation,
        "workflow": route(record, validation),
    }
    with open(audit_path, "a", encoding="utf-8") as audit:  # one JSON line per document
        audit.write(json.dumps(output, default=str) + "\n")
    if webhook_url and doc_type != "general":
        payload = {k: output[k] for k in ("document_id", "document_type", "validation", "workflow")}
        payload["fields"] = {k: v for k, v in record["fields"].items() if k != "items"}
        resp = http.post(webhook_url, data=json.dumps(payload, default=str),
                         headers={"Content-Type": "application/json"}, timeout=10)
        resp.raise_for_status()
    return output


def main() -> None:  # pragma: no cover - needs an Azure resource
    p = argparse.ArgumentParser()
    p.add_argument("source", help="local file or https URL")
    p.add_argument("--type", default="invoice", choices=sorted(MODELS))
    args = p.parse_args()
    result = process(make_client(), args.source, args.type, os.environ.get("APPROVAL_WEBHOOK_URL") or None)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
