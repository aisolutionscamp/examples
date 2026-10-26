"""The tutoring loop: pick a skill, ask a generated question, grade locally, update mastery."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import bkt
from .curriculum import Curriculum, Skill
from .grading import is_correct
from .llm import Question, TutorLLM


@dataclass
class LearnerState:
    mastery: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    recent_questions: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "LearnerState":
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


@dataclass
class Feedback:
    correct: bool
    mastery_before: float
    mastery_after: float
    explanation: str | None = None


class Tutor:
    def __init__(self, curriculum: Curriculum, llm: TutorLLM, state: LearnerState | None = None,
                 params: bkt.BKTParams = bkt.DEFAULT_PARAMS):
        self.curriculum, self.llm, self.params = curriculum, llm, params
        self.state = state or LearnerState()

    def mastery(self, skill_id: str) -> float:
        return self.state.mastery.get(skill_id, self.params.p_init)

    def next_skill(self) -> Skill | None:
        known = {sid: self.mastery(sid) for sid in self.curriculum.skills}
        return self.curriculum.next_skill(known)

    def ask(self, skill: Skill) -> Question:
        question = self.llm.question(skill, self.curriculum.subject, self.mastery(skill.id),
                                     self.state.recent_questions)
        self.state.recent_questions = (self.state.recent_questions + [question.text])[-20:]
        return question

    def submit(self, question: Question, answer: str) -> Feedback:
        before = self.mastery(question.skill_id)
        correct = is_correct(answer, question.answer)
        after = bkt.update(before, correct, self.params)
        self.state.mastery[question.skill_id] = round(after, 4)
        self.state.attempts[question.skill_id] = self.state.attempts.get(question.skill_id, 0) + 1
        explanation = None
        if not correct:
            explanation = self.llm.explain(self.curriculum.skills[question.skill_id], question, answer)
        return Feedback(correct, before, after, explanation)
