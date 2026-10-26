from types import SimpleNamespace

import pytest


class FakeChatClient:
    """Mimics client.chat.completions.create of the OpenAI SDK and records every request."""

    def __init__(self, reply=lambda messages: "summary"):
        self.requests = []
        self._reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = self._reply(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_client():
    return FakeChatClient
