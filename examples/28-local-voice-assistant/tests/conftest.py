import json

import pytest
from openai.types.chat import ChatCompletion

from voice_assistant.assistant import Assistant
from voice_assistant.emergency import EmergencyEscalation
from voice_assistant.reminders import ReminderStore


def completion(content=None, tool_calls=None):
    """A real openai ChatCompletion object, as Ollama's OpenAI-compatible API would return it."""
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "llama3.2:3b",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                     "message": {"role": "assistant", "content": content, "tool_calls": [
                         {"id": f"call_{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args)}}
                         for i, (name, args) in enumerate(tool_calls or [])] or None}}],
    })


class FakeLLM:
    """Stands in for openai.OpenAI: returns scripted completions and records the requests."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture
def make_assistant(tmp_path):
    def make(*responses, webhook="https://hooks.example.com/family", post=None):
        posts = []

        def fake_post(url, **kwargs):
            posts.append((url, kwargs))
            return FakeResponse(200)

        esc = EmergencyEscalation(webhook, "Anna", "112", post=post or fake_post)
        a = Assistant(FakeLLM(*responses), "llama3.2:3b", ReminderStore(tmp_path / "r.json"), esc)
        a.posts = posts
        return a
    return make
