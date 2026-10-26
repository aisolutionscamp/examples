"""Azure Functions (Python v2 model) HTTP endpoint that the Logic App calls for every new email."""

import html
import json
import logging
import re

import azure.functions as func

from azure_triage import analyze_email, draft_reply, make_client, make_openai_client

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger("email_triage")
_clients: dict = {}


def clients():
    if not _clients:
        _clients["language"] = make_client()
        _clients["openai"] = make_openai_client()
    return _clients["language"], _clients["openai"]


def html_to_text(value: str) -> str:
    """The Outlook trigger delivers the body as HTML; Azure AI Language needs plain text."""
    value = re.sub(r"(?is)<(script|style).*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", value)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def handle(payload: dict, language_client, openai_client) -> dict:
    subject = str(payload.get("subject") or "")
    body = html_to_text(str(payload.get("body") or ""))
    if not body.strip():
        raise ValueError("body is required")
    analysis = analyze_email(language_client, f"{subject}\n\n{body}")
    result = analysis.to_dict() | {"draft": draft_reply(openai_client, subject, body, analysis)}
    # Structured log line; query it in Application Insights / Azure Monitor for dashboards.
    logger.info("triage %s", json.dumps({k: result[k] for k in ("language", "category", "urgency",
                                                                        "sentiment", "escalate")}))
    return result


@app.route(route="triage", methods=["POST"])
def triage(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse('{"error": "invalid JSON"}', status_code=400, mimetype="application/json")
    try:
        result = handle(payload, *clients())
    except ValueError as exc:
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=400, mimetype="application/json")
    return func.HttpResponse(json.dumps(result), mimetype="application/json")
