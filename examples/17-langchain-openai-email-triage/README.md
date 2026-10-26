# Support email triage with LangChain + OpenAI

Classifies an incoming support email by urgency, department and intent, then either drafts a reply from your policy texts or flags the email for a human. The output includes a ready-to-send Zendesk ticket payload with the draft as an internal note.

## Architecture

```
email (.eml or text)
  -> preprocess.py   strip quotes/signature, regex-extract order numbers, account IDs, error codes
  -> classify        ChatPromptTemplate | ChatOpenAI.with_structured_output(Classification)
  -> escalation      confidence < MIN_CONFIDENCE, urgency HIGH, or intent outage/refund_request -> human
  -> retrieve        vector store with your policy texts (InMemoryVectorStore + OpenAI embeddings)
  -> draft reply     ChatPromptTemplate | ChatOpenAI | StrOutputParser
  -> zendesk.py      ticket payload (priority, tags, internal note)
```

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
PYTHONPATH=src python -m email_triage data/samples/password.eml
```

Tests (offline, no API key; the LLM and embeddings are replaced with fakes):

```bash
pytest
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | - | required for real runs |
| `OPENAI_MODEL` | `gpt-6-luna` | chat model for classification and drafting |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | embeddings for the policy search |
| `MIN_CONFIDENCE` | `0.7` | below this the email goes to a human |

Policy texts live in `data/policies.json` (one string per policy). Replace them with your own.

## Limitations

- The `confidence` value is the model's self-assessment, not a calibrated probability. Check it against agent corrections before you tune `MIN_CONFIDENCE`.
- Classification quality has not been measured here. Label a few hundred of your own emails and compare before you route automatically.
- The in-memory vector store is rebuilt on every start and is lost on exit.
- Nothing is sent to customers. The draft is an internal note for an agent.

## For production

- Swap `InMemoryVectorStore` for a persistent store, for example `PineconeVectorStore` from `langchain-pinecone`, and index your help-center articles.
- Post the payload to `https://<subdomain>.zendesk.com/api/v2/tickets.json` (API token auth), or update the existing ticket instead of creating one.
- Log the classification and the agent's final category to measure accuracy and to build a few-shot set from corrections.
- Remove personal data you don't need before sending emails to the API, and check your data processing agreement with OpenAI.
