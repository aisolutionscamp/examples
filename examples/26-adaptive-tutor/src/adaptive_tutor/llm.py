"""LLM part: generate practice questions at the right difficulty and explain mistakes.

Uses an OpenAI-compatible chat API (OpenAI, or a local model through Ollama).
"""
import json
import os
from dataclasses import dataclass

from .curriculum import Skill

QUESTION_PROMPT = (
    "You write one practice question for the skill \"{skill}\" in the subject \"{subject}\". "
    "Difficulty: {difficulty}. The answer must be short and unambiguous (a number, a fraction like 3/4, "
    "or one word). Avoid these recent questions: {recent}. "
    "Answer with JSON only: {{\"question\": str, \"answer\": str, \"hint\": str}}."
)
EXPLAIN_PROMPT = (
    "You are a friendly tutor. A learner practising \"{skill}\" answered a question wrongly. "
    "Explain step by step how to get the correct answer, in at most 5 short sentences, and name the "
    "likely misconception behind the learner's answer. Do not just repeat the answer."
)


@dataclass
class Question:
    skill_id: str
    text: str
    answer: str
    hint: str = ""


def difficulty_for(p_mastery: float) -> str:
    if p_mastery < 0.4:
        return "easy, one step"
    if p_mastery < 0.75:
        return "medium, two steps"
    return "hard, several steps or a short word problem"


def make_client():
    from openai import OpenAI

    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)


class TutorLLM:
    def __init__(self, client=None, model: str | None = None):
        self.client = client or make_client()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")

    def _chat(self, system: str, user: str, json_output: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_output else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    def question(self, skill: Skill, subject: str, p_mastery: float, recent: list[str]) -> Question:
        prompt = QUESTION_PROMPT.format(skill=skill.name, subject=subject, difficulty=difficulty_for(p_mastery),
                                        recent=json.dumps(recent[-5:]))
        for _ in range(2):  # one retry if the model breaks the format
            try:
                data = json.loads(self._chat(prompt, "Next question, please.", json_output=True))
                if data.get("question") and str(data.get("answer", "")).strip():
                    return Question(skill.id, str(data["question"]), str(data["answer"]).strip(),
                                    str(data.get("hint", "")))
            except (json.JSONDecodeError, AttributeError):
                pass
        raise RuntimeError("model did not return a usable question")

    def explain(self, skill: Skill, question: Question, learner_answer: str) -> str:
        user = f"Question: {question.text}\nCorrect answer: {question.answer}\nLearner's answer: {learner_answer}"
        return self._chat(EXPLAIN_PROMPT.format(skill=skill.name), user)
