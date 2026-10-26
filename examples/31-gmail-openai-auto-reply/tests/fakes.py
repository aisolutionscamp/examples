"""In-memory fake of the parts of the Gmail API client used here (users().messages()/drafts()/labels())."""

import base64
from types import SimpleNamespace as NS


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def gmail_message(mid, sender, subject, body, html=False):
    part = {"mimeType": "text/plain", "body": {"data": b64(body)}}
    payload = {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/html", "body": {"data": b64(f"<p>{body}</p>")}}, part]} if html else part
    payload["headers"] = [{"name": "From", "value": sender}, {"name": "Subject", "value": subject},
                          {"name": "Message-ID", "value": f"<{mid}@mail.example.com>"}]
    return {"id": mid, "threadId": f"t-{mid}", "snippet": body[:50], "payload": payload}


class _Call:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeGmailService:
    def __init__(self, messages):
        self.messages_by_id = {m["id"]: m for m in messages}
        self.labels = [{"id": "INBOX", "name": "INBOX"}]
        self.created_drafts, self.modified, self.queries = [], [], []

    def users(self):
        return NS(
            messages=lambda: NS(list=self._list, get=self._get, modify=self._modify),
            drafts=lambda: NS(create=self._create_draft),
            labels=lambda: NS(list=lambda userId: _Call(lambda: {"labels": self.labels}), create=self._create_label),
        )

    def _list(self, userId, q, maxResults):
        self.queries.append(q)
        return _Call(lambda: {"messages": [{"id": i} for i in list(self.messages_by_id)[:maxResults]]})

    def _get(self, userId, id, format):
        return _Call(lambda: self.messages_by_id[id])

    def _modify(self, userId, id, body):
        self.modified.append((id, body["addLabelIds"]))
        return _Call(lambda: {"id": id})

    def _create_draft(self, userId, body):
        self.created_drafts.append(body)
        return _Call(lambda: {"id": f"d{len(self.created_drafts)}"})

    def _create_label(self, userId, body):
        label = {"id": f"L{len(self.labels)}", "name": body["name"]}
        self.labels.append(label)
        return _Call(lambda: label)


class FakeOpenAI:
    """Returns queued Decision objects from responses.parse and records the requests."""

    def __init__(self, decisions):
        self.decisions, self.requests = list(decisions), []
        self.responses = NS(parse=self._parse)

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        return NS(output_parsed=self.decisions.pop(0))
