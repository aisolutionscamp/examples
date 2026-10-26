import json
from dataclasses import dataclass
from pathlib import Path

MASTERY_THRESHOLD = 0.95  # common choice in BKT-based tutors


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    prerequisites: tuple[str, ...] = ()


@dataclass
class Curriculum:
    subject: str
    skills: dict[str, Skill]

    @classmethod
    def load(cls, path: str | Path) -> "Curriculum":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        skills = {s["id"]: Skill(s["id"], s["name"], tuple(s.get("prerequisites", []))) for s in data["skills"]}
        for skill in skills.values():
            missing = [p for p in skill.prerequisites if p not in skills]
            if missing:
                raise ValueError(f"{skill.id}: unknown prerequisites {missing}")
        return cls(data["subject"], skills)

    def next_skill(self, mastery: dict[str, float], threshold: float = MASTERY_THRESHOLD) -> Skill | None:
        """The least-mastered skill whose prerequisites are all mastered; None when everything is done."""
        ready = [
            s for s in self.skills.values()
            if mastery.get(s.id, 0.0) < threshold
            and all(mastery.get(p, 0.0) >= threshold for p in s.prerequisites)
        ]
        # ties keep curriculum order
        return min(ready, key=lambda s: mastery.get(s.id, 0.0), default=None)
