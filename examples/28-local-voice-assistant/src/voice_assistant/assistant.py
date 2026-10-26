"""The conversation loop: emergency check first, then reminders, then the local LLM with tools."""
from __future__ import annotations

import json
import re
from datetime import datetime

from .emergency import EmergencyEscalation, is_emergency
from .reminders import ReminderStore

SYSTEM_PROMPT = """You are a friendly voice assistant for an older person living at home.
Current local time: {now}.
Rules:
- Answer in one to three short, simple sentences. Ask at most one question at a time.
- You can set and list reminders with the tools. Confirm the time back to the user.
- Do not give medical advice and never suggest changing a medication or dose; suggest asking
  the doctor or pharmacist instead.
- You cannot call emergency services. If the user may be in danger, tell them to call {number}."""

TOOLS = [
    {"type": "function", "function": {
        "name": "add_reminder",
        "description": "Create a reminder, daily unless a date is given.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to remind about, e.g. 'take the blood pressure tablet'"},
            "at": {"type": "string", "description": "24h local time HH:MM"},
            "kind": {"type": "string", "enum": ["medication", "appointment", "other"]},
            "on_date": {"type": "string", "description": "YYYY-MM-DD for a one-off reminder"},
        }, "required": ["text", "at", "kind"]},
    }},
    {"type": "function", "function": {
        "name": "list_reminders",
        "description": "List today's reminders.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

_DONE = re.compile(r"\b(done|i took (them|it|my)|i have taken|taken (them|it))\b", re.IGNORECASE)


class Assistant:
    def __init__(self, llm, model: str, reminders: ReminderStore, escalation: EmergencyEscalation,
                 transcriber=None, speaker=None, max_history: int = 10):
        self.llm, self.model = llm, model  # llm: an openai.OpenAI client (pointed at Ollama)
        self.reminders, self.escalation = reminders, escalation
        self.transcriber, self.speaker = transcriber, speaker
        self.history: list[dict] = []
        self.max_history = max_history

    def handle_text(self, text: str, now: datetime) -> str:
        if is_emergency(text):  # deterministic, never waits for the model
            return self.escalation.escalate(text, now)
        if _DONE.search(text) and (done := self.reminders.acknowledge_open(now)):
            return "Thank you. I have noted it: " + ", ".join(r.text for r in done) + "."
        reply = self._chat(text, now)
        self.history = (self.history + [{"role": "user", "content": text},
                                        {"role": "assistant", "content": reply}])[-self.max_history:]
        return reply

    def handle_audio(self, wav_in: str, wav_out: str, now: datetime) -> tuple[str, str]:
        heard = self.transcriber.transcribe(wav_in)
        if not heard:
            reply = "Sorry, I did not catch that. Could you say it again?"
        else:
            reply = self.handle_text(heard, now)
        self.speaker.speak(reply, wav_out)
        return heard, reply

    def reminder_announcements(self, now: datetime) -> list[str]:
        """Sentences to say now: new reminders and repeats of unconfirmed ones."""
        out = [f"It is {r.at}. Time to {r.text}. Say 'done' when you have." for r in self.reminders.due(now)]
        out += [f"Just a reminder: {r.text}. Have you done it?" for r in self.reminders.repeat_unacknowledged(now)]
        return out

    def _chat(self, text: str, now: datetime) -> str:
        system = SYSTEM_PROMPT.format(now=now.strftime("%A %Y-%m-%d %H:%M"),
                                      number=self.escalation.emergency_number)
        messages = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": text}]
        for _ in range(3):  # at most a few tool rounds per turn
            msg = self.llm.chat.completions.create(model=self.model, messages=messages, tools=TOOLS,
                                                   temperature=0.3).choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip() or "Sorry, could you say that again?"
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls]})
            for call in msg.tool_calls:
                result = self._run_tool(call.function.name, call.function.arguments, now)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        return "Sorry, I could not finish that. Please try again."

    def _run_tool(self, name: str, arguments: str, now: datetime) -> dict:
        try:
            args = json.loads(arguments or "{}")
            if name == "add_reminder":
                r = self.reminders.add(args["text"], args["at"], args.get("kind", "other"), args.get("on_date"))
                return {"ok": True, "id": r.id, "at": r.at, "on_date": r.on_date}
            if name == "list_reminders":
                return {"ok": True, "reminders": [{"text": r.text, "at": r.at, "kind": r.kind}
                                                  for r in self.reminders.for_day(now.date())]}
            return {"ok": False, "error": f"unknown tool {name}"}
        except (KeyError, ValueError) as e:  # bad JSON or a time like "8 pm": let the model ask again
            return {"ok": False, "error": str(e)}
