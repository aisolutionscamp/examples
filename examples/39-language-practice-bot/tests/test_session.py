import json

from language_bot.scenarios import SCENARIOS
from language_bot.session import PracticeSession, parse_answer


def answer(reply, corrections=(), translation="t"):
    return json.dumps({"reply": reply, "corrections": list(corrections), "translation": translation})


def test_system_prompt_contains_scenario_and_level(fake_client):
    session = PracticeSession(SCENARIOS["train"], "French", "B1", "German", client=fake_client(), model="m")
    assert "ticket agent" in session.system
    assert "French" in session.system and "B1" in session.system and "German" in session.system


def test_turn_with_corrections(fake_client):
    content = answer("¡Claro! ¿Algo para beber?", [
        {"original": "Yo quiero un mesa", "corrected": "Quiero una mesa", "explanation": "mesa is feminine"},
        {"original": "same", "corrected": "same", "explanation": "no change"},  # dropped
    ], "Sure. Anything to drink?")
    client = fake_client(lambda messages: content)
    session = PracticeSession(SCENARIOS["restaurant"], client=client, model="m")
    turn = session.send("Yo quiero un mesa")
    assert turn.reply == "¡Claro! ¿Algo para beber?"
    assert turn.translation == "Sure. Anything to drink?"
    assert [c["corrected"] for c in turn.corrections] == ["Quiero una mesa"]
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert session.review() == turn.corrections


def test_history_holds_replies_only_and_is_trimmed(fake_client):
    replies = iter(answer(f"reply {i}", [{"original": "x", "corrected": "y", "explanation": "e"}])
                   for i in range(10))
    client = fake_client(lambda messages: next(replies))
    session = PracticeSession(SCENARIOS["smalltalk"], client=client, model="m", max_history=2)
    for i in range(4):
        session.send(f"msg {i}")
    messages = client.requests[-1]["messages"]
    assert [m["content"] for m in messages[1:]] == ["msg 1", "reply 1", "msg 2", "reply 2", "msg 3"]
    assert all("corrections" not in m["content"] for m in messages[1:])
    assert len(session.review()) == 4


def test_parse_answer_handles_fences_and_plain_text():
    fenced = "```json\n" + answer("Hola") + "\n```"
    assert parse_answer(fenced).reply == "Hola"
    plain = parse_answer("Hola, ¿qué tal?")
    assert plain.reply == "Hola, ¿qué tal?" and plain.corrections == []
    assert parse_answer("[1, 2]").reply == "[1, 2]"


def test_model_from_env(monkeypatch, fake_client):
    monkeypatch.setenv("OPENAI_MODEL", "llama3.2")
    client = fake_client(lambda messages: answer("ok"))
    PracticeSession(SCENARIOS["hotel"], client=client).send("Hola")
    assert client.requests[0]["model"] == "llama3.2"
