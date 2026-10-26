"""One practice conversation: role-play plus corrections of the learner's last message.

The model answers in JSON so the app can show the in-character reply and the corrections separately.
Works with OpenAI or any OpenAI-compatible server (Ollama, vLLM).
"""
import json
import os
from dataclasses import dataclass, field

from .scenarios import Scenario

SYSTEM_PROMPT = """You are a patient {language} tutor playing the {bot_role} in {setting}. \
The learner is the {learner_role}, level {level} (CEFR), native language {native}.
Rules:
- Stay in character and answer in simple {language} suited to level {level}, at most 3 sentences, \
and end with a question that keeps the conversation going.
- Check only the learner's last message for grammar, vocabulary and word-order mistakes. \
Ignore missing accents and punctuation if the meaning is clear.
- Explain each mistake briefly in {native}.
Answer with JSON only: {{"reply": str, "corrections": [{{"original": str, "corrected": str, \
"explanation": str}}], "translation": str}} where "translation" is your reply in {native}."""


@dataclass
class Turn:
    learner: str
    reply: str
    corrections: list[dict] = field(default_factory=list)
    translation: str = ""


def make_client():
    from openai import OpenAI

    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)


def parse_answer(content: str) -> Turn:
    """Tolerates answers wrapped in ```json fences; plain text becomes a reply without corrections."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Turn(learner="", reply=content.strip())
    if not isinstance(data, dict):
        return Turn(learner="", reply=content.strip())
    corrections = [
        {k: str(c.get(k, "")) for k in ("original", "corrected", "explanation")}
        for c in data.get("corrections") or []
        if isinstance(c, dict) and c.get("original") and c.get("original") != c.get("corrected")
    ]
    return Turn(learner="", reply=str(data.get("reply", "")).strip(), corrections=corrections,
                translation=str(data.get("translation", "")).strip())


class PracticeSession:
    def __init__(self, scenario: Scenario, language: str = "Spanish", level: str = "A2",
                 native: str = "English", client=None, model: str | None = None, max_history: int = 12):
        self.client = client or make_client()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        self.system = SYSTEM_PROMPT.format(language=language, level=level, native=native,
                                           setting=scenario.setting, bot_role=scenario.bot_role,
                                           learner_role=scenario.learner_role)
        self.max_history = max_history
        self.turns: list[Turn] = []

    def _messages(self, learner_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system}]
        for turn in self.turns[-self.max_history:]:
            messages.append({"role": "user", "content": turn.learner})
            # only the in-character reply goes back into the history, not the corrections
            messages.append({"role": "assistant", "content": turn.reply})
        messages.append({"role": "user", "content": learner_text})
        return messages

    def send(self, learner_text: str) -> Turn:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(learner_text),
            response_format={"type": "json_object"},
        )
        turn = parse_answer(response.choices[0].message.content)
        turn.learner = learner_text
        self.turns.append(turn)
        return turn

    def review(self) -> list[dict]:
        """All corrections of the session, for a review at the end."""
        return [c for t in self.turns for c in t.corrections]
