"""Thin wrapper around the Gmail API: read unread messages, create reply drafts, set labels."""

import base64
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

# gmail.modify allows reading, labeling and creating drafts, but not deleting messages permanently.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


@dataclass
class Email:
    id: str
    thread_id: str
    sender: str
    subject: str
    message_id: str
    references: str
    body: str

    @property
    def sender_address(self) -> str:
        return parseaddr(self.sender)[1]


def get_service(credentials_file: str = "credentials.json", token_file: str = "token.json"):
    """OAuth2 desktop flow. The first run opens a browser; the token is cached in token.json."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if Path(token_file).exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES).run_local_server(port=0)
        Path(token_file).write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _plain_text(payload: dict) -> str:
    """Return the first text/plain part of a (possibly multipart) message payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return _decode(payload["body"]["data"])
    for part in payload.get("parts", []):
        text = _plain_text(part)
        if text:
            return text
    return ""


def list_unread(service, query: str, max_results: int = 20) -> list[str]:
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    return [m["id"] for m in resp.get("messages", [])]


def get_email(service, message_id: str) -> Email:
    msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
    return Email(
        id=msg["id"],
        thread_id=msg["threadId"],
        sender=headers.get("from", ""),
        subject=headers.get("subject", ""),
        message_id=headers.get("message-id", ""),
        references=headers.get("references", ""),
        body=_plain_text(msg["payload"]) or msg.get("snippet", ""),
    )


def create_reply_draft(service, email: Email, reply_text: str, from_address: str) -> str:
    """Create a draft in the same thread. Returns the draft id. Nothing is sent."""
    reply = EmailMessage()
    reply["To"] = email.sender
    reply["From"] = from_address
    reply["Subject"] = email.subject if email.subject.lower().startswith("re:") else f"Re: {email.subject}"
    if email.message_id:
        reply["In-Reply-To"] = email.message_id
        reply["References"] = f"{email.references} {email.message_id}".strip()
    reply.set_content(reply_text)
    raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw, "threadId": email.thread_id}}).execute()
    return draft["id"]


def ensure_label(service, name: str) -> str:
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"] == name:
            return label["id"]
    created = service.users().labels().create(
        userId="me", body={"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    ).execute()
    return created["id"]


def add_labels(service, message_id: str, label_ids: list[str]) -> None:
    service.users().messages().modify(userId="me", id=message_id, body={"addLabelIds": label_ids}).execute()
