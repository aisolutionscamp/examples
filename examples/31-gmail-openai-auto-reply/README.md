# Gmail auto-reply drafts with OpenAI

Reads unread messages from a Gmail inbox, asks an OpenAI model to categorize each one and, for business inquiries, write a reply based on your business information. The reply is saved as a **draft in the same thread**. Nothing is sent automatically: you open Gmail, check the draft, and press send.

## Architecture

```
Gmail API  users.messages.list (unread, inbox, not yet labeled AI/processed)
  -> users.messages.get        sender, subject, Message-ID, first text/plain part
  -> OpenAI Responses API      responses.parse(..., text_format=Decision)
                               category, needs_reply, reply, reason
  -> rules                     no draft for newsletter / spam / personal / automated, never answer own mail
  -> users.drafts.create       reply draft with In-Reply-To / References and threadId
  -> users.messages.modify     labels AI/processed and AI/<category>
```

## What you need

- A Google account (Gmail or Google Workspace) and a Google Cloud project with the **Gmail API** enabled.
- An OAuth client of type **Desktop app**; download it as `credentials.json` into this folder. While the OAuth consent screen is in testing mode, add your address as a test user.
- An OpenAI API key.

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env              # OPENAI_API_KEY, GMAIL_ADDRESS
# edit business_info.md with your real prices, hours and policies
PYTHONPATH=src python -m gmail_autoreply --dry-run   # first run opens the browser for Google login
PYTHONPATH=src python -m gmail_autoreply             # create drafts and labels
PYTHONPATH=src python -m gmail_autoreply --interval 10   # repeat every 10 minutes
```

The OAuth token is cached in `token.json`. Both `credentials.json` and `token.json` are in `.gitignore`.

Tests (offline; Gmail and OpenAI are replaced with in-memory fakes):

```bash
pytest
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | - | required |
| `OPENAI_MODEL` | `gpt-6-luna` | model for categorizing and drafting |
| `GMAIL_ADDRESS` | - | your address; used as From and to skip your own mail |
| `GMAIL_QUERY` | unread inbox mail without promotions/social and without `AI/processed` | Gmail search query |
| `BUSINESS_INFO_FILE` | `business_info.md` | the only facts the model may use in replies |

## Limitations

- The model only knows what is in `business_info.md`. It is told not to invent prices or dates, but check every draft before sending.
- Categorization quality has not been measured. Review the labels for a week before you trust them.
- Attachments and HTML-only emails are reduced to the text part or the Gmail snippet.
- Email content is sent to the OpenAI API. Check that this fits your privacy policy and data processing agreement.
- The `gmail.modify` scope gives the script read and label access to the whole mailbox.

## For production

- Run it on a schedule (cron, Cloud Run job) instead of `--interval`, or use Gmail push notifications (`users.watch` + Pub/Sub).
- Store the token in a secret manager instead of `token.json`.
- Add a small review UI or a Google Sheet log if several people share the inbox.
- Only after a long review phase, consider sending replies automatically for one narrow category, for example opening hours.
