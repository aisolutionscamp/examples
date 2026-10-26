"""Decide who has to look at a document next. Thresholds and addresses are configuration."""

APPROVERS = {
    "accounts_payable": "ap@example.com",
    "finance_manager": "finance.manager@example.com",
    "cfo": "cfo@example.com",
    "legal_review": "legal@example.com",
    "document_review": "doc-review@example.com",
}


def route(record: dict, validation: dict, high: float = 10_000, medium: float = 1_000) -> dict:
    steps: list[str] = []
    if validation["status"] != "valid":
        steps.append("document_review")  # a person checks the extraction first
    if record["document_type"] == "invoice":
        total = record["fields"].get("total") or 0
        steps += ["finance_manager", "cfo"] if total > high else ["finance_manager"] if total > medium \
            else ["accounts_payable"]
    elif record["document_type"] == "contract":
        steps.append("legal_review")
    priority = "high" if validation["status"] == "invalid" or "cfo" in steps else "normal"
    return {"steps": steps, "approvers": [APPROVERS[s] for s in steps], "priority": priority}
