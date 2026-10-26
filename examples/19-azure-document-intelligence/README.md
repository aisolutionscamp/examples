# Azure AI Document Intelligence pipeline

Sends invoices, contracts or any other document to **Azure AI Document Intelligence** (formerly Azure Form Recognizer), turns the result into a flat record, checks it with business rules, decides who has to approve it, writes an audit line and optionally starts an approval flow in **Power Automate** or **Azure Logic Apps** through an HTTP trigger.

## How it works

```
file or URL ─► Document Intelligence (prebuilt-invoice | prebuilt-contract | prebuilt-layout)
            ─► extract.py   typed fields + per-field confidence
            ─► validate.py  required fields, confidence >= 0.8, subtotal + tax = total, date order
            ─► routing.py   review step if needed, approvers by amount
            ─► audit.jsonl  one JSON line per document
            ─► webhook      Power Automate / Logic Apps HTTP trigger (optional)
```

- **Models**: `prebuilt-invoice` for invoices, `prebuilt-contract` for contracts (parties, dates), `prebuilt-layout` for everything else (text, tables, pages). All are prebuilt models of the Document Intelligence v4.0 API (`2024-11-30`), the default of `azure-ai-documentintelligence` 1.0.x.
- **Validation status**: `valid`, `requires_review` (low-confidence fields, line items that do not add up) or `invalid` (missing required fields, totals that do not match, due date before invoice date).
- **Routing**: anything not `valid` first goes to `document_review`. Invoices over 10,000 go to the finance manager and the CFO, over 1,000 to the finance manager, the rest to accounts payable. Contracts go to legal review. Addresses in `routing.py` are placeholders (`@example.com`).

## Setup (needs an Azure subscription)

1. Create a Document Intelligence resource (or an Azure AI services resource) in the Azure portal. The free F0 tier allows a limited number of pages per month.
2. Configure:

   ```bash
   pip install -r requirements.txt
   cp .env.example .env   # endpoint, and either a key or nothing for Microsoft Entra ID
   set -a; . ./.env; set +a
   ```

   With an empty key, `DefaultAzureCredential` is used (managed identity, `az login`, ...). The identity needs the role *Cognitive Services User* on the resource.
3. Run:

   ```bash
   cd src
   python -m doc_pipeline.pipeline ../invoice.pdf --type invoice
   python -m doc_pipeline.pipeline https://example.com/contract.pdf --type contract
   ```

4. Optional approval flow: create a Power Automate flow with the trigger *When an HTTP request is received* (or a Logic App with an HTTP trigger), put its URL in `APPROVAL_WEBHOOK_URL`. The pipeline posts `document_id`, `document_type`, the header fields, the validation result and the approvers.

## Tests (offline)

```bash
scripts/test-example.sh examples/19-azure-document-intelligence   # from the repo root, runs in Docker
```

The tests build real `AnalyzeResult` objects of the SDK from JSON fixtures in `tests/fixtures/` (same shape as the REST response), pass them through a fake client, and check field typing, validation, routing, the audit file, the webhook payload and that local files and URLs are sent the right way.

## Limitations

- The document type is given by the caller. To detect it automatically, train a custom classification model in Document Intelligence Studio and call it before extraction.
- Field names are those of the prebuilt models; other languages and layouts may return fewer fields. Check the confidence values on your own documents.
- The webhook is called synchronously and without retries.

## For production

- Store originals and results in Blob Storage or SharePoint (Microsoft Graph) instead of a local audit file; keep the `document_id` as the link.
- Run the pipeline from a queue (Azure Functions with a Blob or Service Bus trigger) so uploads do not wait for analysis.
- Keep the key in Key Vault or use managed identity, and restrict network access to the resource.
- Tune the confidence threshold and amount limits with your finance team, and measure extraction accuracy on a labelled sample of your own documents.
