import json

import azure.functions as func
from fakes import FakeChatClient, FakeLanguageClient

import function_app


def call(body: bytes, monkeypatch, language_client=None):
    monkeypatch.setattr(function_app, "clients", lambda: (language_client or FakeLanguageClient(), FakeChatClient()))
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "d")
    req = func.HttpRequest(method="POST", url="/api/triage", body=body, headers={"Content-Type": "application/json"})
    handler = function_app.triage.build().get_user_function()
    return handler(req)


def test_triage_endpoint_returns_analysis_and_draft(monkeypatch):
    payload = {"subject": "Invoice", "body": "<p>Where is my <b>invoice</b>?</p><br>Thanks &amp; regards"}
    resp = call(json.dumps(payload).encode(), monkeypatch)
    data = json.loads(resp.get_body())
    assert resp.status_code == 200
    assert data["category"] == "billing" and data["draft"] == "Draft reply"


def test_html_to_text():
    assert function_app.html_to_text("<style>p{}</style><p>Hi&nbsp;there</p><p>Line 2</p>") == "Hi\xa0there\nLine 2"


def test_bad_requests(monkeypatch):
    assert call(b"not json", monkeypatch).status_code == 400
    resp = call(json.dumps({"subject": "x", "body": " "}).encode(), monkeypatch)
    assert resp.status_code == 400 and "body is required" in resp.get_body().decode()
