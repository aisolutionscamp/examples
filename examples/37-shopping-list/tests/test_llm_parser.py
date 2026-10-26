import json
from types import SimpleNamespace

from shopping_list.llm_parser import parse_with_llm


class FakeClient:
    def __init__(self, content):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def test_llm_items_are_used_and_categories_validated():
    content = json.dumps({"items": [
        {"name": "Flour", "quantity": 1, "unit": "kg", "category": "pantry"},
        {"name": "eggs", "quantity": "3", "unit": None, "category": "dairy"},
        {"name": "milk", "quantity": 0.5, "unit": "l", "category": "snacks"},  # invalid category
        {"name": "", "quantity": 1},  # dropped
    ]})
    client = FakeClient(content)
    items = parse_with_llm("stuff for pancakes", client=client, model="test-model")
    assert [(i.name, i.quantity, i.unit, i.category) for i in items] == [
        ("flour", 1.0, "kg", "pantry"), ("eggs", 3.0, None, "dairy"), ("milk", 0.5, "l", "dairy")]
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][1]["content"] == "stuff for pancakes"


def test_invalid_json_falls_back_to_rules():
    items = parse_with_llm("2 kg potatoes, milk", client=FakeClient("Sure! Here is your list"), model="m")
    assert [i.name for i in items] == ["potatoes", "milk"]


def test_model_default_comes_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "llama3.2")
    client = FakeClient('{"items": []}')
    parse_with_llm("milk", client=client)
    assert client.requests[0]["model"] == "llama3.2"
