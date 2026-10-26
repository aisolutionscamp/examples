from datetime import datetime

import httpx2
import pytest
from conftest import completion

from voice_assistant.emergency import is_emergency

NOW = datetime(2026, 3, 1, 10, 0)


@pytest.mark.parametrize("text", ["I have fallen and can't get up", "Help me please", "I've got chest pain",
                                  "I cannot breathe", "I can not get up"])
def test_emergency_phrases_are_detected(text):
    assert is_emergency(text)


@pytest.mark.parametrize("text", ["What is the weather today?", "Remind me to call Peter"])
def test_normal_phrases_are_not_emergencies(text):
    assert not is_emergency(text)


def test_emergency_notifies_contact_and_never_calls_the_llm(make_assistant):
    a = make_assistant()  # no scripted LLM responses: a call would fail
    reply = a.handle_text("I fell in the kitchen, I can't get up", NOW)
    assert "sent a message to Anna" in reply and "call 112" in reply
    url, kwargs = a.posts[0]
    assert url == "https://hooks.example.com/family"
    assert kwargs["json"]["event"] == "possible_emergency"
    assert a.llm.requests == []


def test_failed_escalation_still_tells_user_to_call_emergency_number(make_assistant):
    def broken_post(url, **kwargs):
        raise httpx2.ConnectError("offline")

    reply = make_assistant(post=broken_post).handle_text("help me", NOW)
    assert reply.startswith("I could not reach Anna") and "call 112" in reply


def test_llm_tool_call_creates_reminder(make_assistant):
    a = make_assistant(
        completion(tool_calls=[("add_reminder", {"text": "take the heart tablet", "at": "20:00",
                                                 "kind": "medication"})]),
        completion("All right. I will remind you every day at 8 in the evening."),
    )
    reply = a.handle_text("Please remind me to take my heart tablet at 8 in the evening", NOW)
    assert reply.startswith("All right")
    assert [(r.text, r.at) for r in a.reminders.items] == [("take the heart tablet", "20:00")]
    second = a.llm.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and '"ok": true' in second[-1]["content"]
    assert "Do not give medical advice" in second[0]["content"]
    assert a.llm.requests[0]["model"] == "llama3.2:3b"


def test_invalid_tool_arguments_are_returned_to_the_model(make_assistant):
    a = make_assistant(
        completion(tool_calls=[("add_reminder", {"text": "pills", "at": "8 pm", "kind": "medication"})]),
        completion("What time exactly? For example 20:00."),
    )
    a.handle_text("remind me about pills at 8 pm", NOW)
    assert a.reminders.items == []
    assert '"ok": false' in a.llm.requests[1]["messages"][-1]["content"]


def test_reminder_announcement_and_done(make_assistant):
    a = make_assistant()
    a.reminders.add("take the tablet", "08:00")
    lines = a.reminder_announcements(datetime(2026, 3, 1, 8, 1))
    assert lines == ["It is 08:00. Time to take the tablet. Say 'done' when you have."]
    assert "noted" in a.handle_text("Done, I took them", datetime(2026, 3, 1, 8, 5))
    assert a.reminder_announcements(datetime(2026, 3, 1, 9, 0)) == []


def test_history_is_sent_and_capped(make_assistant):
    a = make_assistant(*[completion(f"answer {i}") for i in range(8)])
    a.max_history = 4
    for i in range(8):
        a.handle_text(f"question {i}", NOW)
    assert len(a.history) == 4
    last_messages = a.llm.requests[-1]["messages"]
    assert [m["content"] for m in last_messages[1:-1]] == ["question 5", "answer 5", "question 6", "answer 6"]


def test_audio_turn_uses_stt_and_tts(make_assistant, tmp_path):
    class FakeSTT:
        def transcribe(self, path):
            return "What day is it today?"

    class FakeTTS:
        def __init__(self):
            self.spoken = []

        def speak(self, text, out):
            self.spoken.append((text, out))
            return out

    a = make_assistant(completion("Today is Sunday, the first of March."))
    tts = FakeTTS()
    a.transcriber, a.speaker = FakeSTT(), tts
    heard, reply = a.handle_audio("in.wav", str(tmp_path / "out.wav"), NOW)
    assert heard == "What day is it today?"
    assert tts.spoken == [(reply, str(tmp_path / "out.wav"))]


def test_without_configured_contact_only_points_to_emergency_number(make_assistant):
    reply = make_assistant(webhook="").handle_text("I can't breathe", NOW)
    assert reply == "No emergency contact is set up. If you are hurt or in danger, call 112 now."
