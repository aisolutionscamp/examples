# Support email triage with Azure AI Language + Logic Apps

A Logic App watches a shared Outlook mailbox. For every new email it calls an Azure Function that analyzes the text with Azure AI Language (language detection, sentiment, named entities, key phrases), applies simple routing rules, and optionally writes a reply draft with Azure OpenAI. The Logic App then sends the draft to the agents' mailbox, or alerts the escalation mailbox.

## Architecture

```
Outlook (Office 365 connector, "When a new email arrives (V3)", polls every 3 min)
  -> HTTP action  POST /api/triage  (Azure Function, function_app.py)
       html_to_text -> Azure AI Language: detect_language, analyze_sentiment, recognize_entities, extract_key_phrases
       -> rules (azure_triage/analyzer.py): category, urgency, escalate?
       -> Azure OpenAI draft (azure_triage/drafts.py, only for routine emails, optional)
  -> Condition "escalate"
       true  -> email to escalation mailbox (importance High)
       false -> email with the draft to the agents' mailbox (nothing is sent to the customer)
Logs: one structured log line per email ("triage {...}") in Application Insights
```

Files:

- `function_app.py`: HTTP endpoint (Azure Functions Python v2 model)
- `azure_triage/analyzer.py`: Azure AI Language calls and routing rules
- `azure_triage/drafts.py`: optional Azure OpenAI reply draft
- `logic-app/workflow.json`: Logic App (Consumption) workflow definition

## What you need

An Azure subscription with:

- an **Azure AI Language** resource (endpoint and key)
- a **Function App** (Python 3.12, Linux)
- a **Logic App (Consumption)** with an Office 365 Outlook connection to a work or school mailbox
- optional: an **Azure OpenAI** resource with a chat model deployment

## Setup

1. Create the Language resource and copy endpoint and key.
2. Deploy the function: `func azure functionapp publish <app-name>` (Azure Functions Core Tools), then set the app settings from `local.settings.example.json`.
3. Create the Logic App, open the code view and paste `logic-app/workflow.json`. Create the Office 365 connection in the designer, fill in the `$connections` IDs, `triageFunctionUrl` and `triageFunctionKey` (function key of the `triage` function). Check the trigger and the two "Send an email (V2)" actions in the designer once; it shows the connector's current fields.
4. Send a test email to the mailbox and check the run history.

Run locally:

```bash
pip install -r requirements.txt
cp local.settings.example.json local.settings.json   # fill in the keys
func start
curl -X POST localhost:7071/api/triage -H 'Content-Type: application/json' \
  -d '{"subject": "Invoice", "body": "Where can I download my March invoice?"}'
```

Tests (offline, the Azure clients are replaced with fakes):

```bash
pytest
```

## Configuration

| Setting | Meaning |
|---|---|
| `AZURE_LANGUAGE_ENDPOINT`, `AZURE_LANGUAGE_KEY` | Azure AI Language resource |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | optional; leave the endpoint empty to skip drafts |
| `AZURE_OPENAI_DEPLOYMENT` | name of your chat model deployment |
| `AZURE_OPENAI_API_VERSION` | default `2024-10-21` |

Routing rules (keywords per language, urgency patterns, escalation thresholds) are constants at the top of `azure_triage/analyzer.py`.

## Limitations

- The category comes from keyword rules over the text and key phrases, not from a trained classifier. It is easy to read and change, but it misses phrasings that are not in the lists. For better routing, train an Azure AI Language **custom text classification** project on your labeled emails and call `begin_single_label_classify`.
- Keywords exist for English, Spanish and German. Other languages are analyzed in English mode and escalated.
- The function makes four Language API calls per email (one per feature). `begin_analyze_actions` can bundle them, but it is asynchronous and slower for single emails.
- Only the first 5,000 characters of each email are analyzed.
- The Logic App definition has not been deployed as part of this repository's tests. Connector field names can change; verify them in the designer.

## For production

- Use managed identity (`DefaultAzureCredential` from `azure-identity`) instead of keys, and Key Vault references for secrets.
- Store the triage result on the ticket (Dynamics 365, Zendesk, Freshdesk) instead of sending an internal email.
- Build reports in Power BI or Azure Monitor workbooks from the `triage` log lines.
- Log agent corrections to measure routing accuracy, and use them as training data for custom classification.
