import json
from pathlib import Path

import pytest

from adaptive_tutor.curriculum import Curriculum
from adaptive_tutor.llm import TutorLLM, difficulty_for
from adaptive_tutor.tutor import LearnerState, Tutor

FRACTIONS = Path(__file__).parents[1] / "curricula" / "fractions.json"


def question_json(q="What is 1/2 + 1/4?", a="3/4"):
    return json.dumps({"question": q, "answer": a, "hint": "Use quarters."})


def scripted(*answers):
    it = iter(answers)
    return lambda messages: next(it)


def test_difficulty_follows_mastery():
    assert difficulty_for(0.1).startswith("easy")
    assert difficulty_for(0.5).startswith("medium")
    assert difficulty_for(0.9).startswith("hard")


def test_correct_answer_updates_mastery_without_explanation(fake_client):
    client = fake_client(scripted(question_json()))
    tutor = Tutor(Curriculum.load(FRACTIONS), TutorLLM(client, model="m"))
    skill = tutor.next_skill()
    question = tutor.ask(skill)
    feedback = tutor.submit(question, "0.75")
    assert feedback.correct and feedback.explanation is None
    assert feedback.mastery_after > feedback.mastery_before == 0.2
    assert tutor.state.attempts == {"frac_concept": 1}
    prompt = client.requests[0]["messages"][0]["content"]
    assert "What a fraction means" in prompt and "easy" in prompt
    assert len(client.requests) == 1


def test_wrong_answer_gets_explanation(fake_client):
    client = fake_client(scripted(question_json(), "Turn 1/2 into 2/4 first, then add the quarters."))
    tutor = Tutor(Curriculum.load(FRACTIONS), TutorLLM(client, model="m"))
    question = tutor.ask(tutor.next_skill())
    feedback = tutor.submit(question, "2/6")
    assert not feedback.correct
    assert feedback.mastery_after < feedback.mastery_before
    assert feedback.explanation.startswith("Turn 1/2")
    assert "Learner's answer: 2/6" in client.requests[1]["messages"][1]["content"]


def test_broken_json_is_retried_once(fake_client):
    client = fake_client(scripted("not json", question_json()))
    tutor = Tutor(Curriculum.load(FRACTIONS), TutorLLM(client, model="m"))
    assert tutor.ask(tutor.next_skill()).answer == "3/4"
    with pytest.raises(RuntimeError):
        Tutor(Curriculum.load(FRACTIONS), TutorLLM(fake_client(scripted("x", "{}")), model="m")).ask(
            tutor.next_skill())


def test_recent_questions_are_passed_back(fake_client):
    client = fake_client(scripted(question_json("Q1"), question_json("Q2")))
    tutor = Tutor(Curriculum.load(FRACTIONS), TutorLLM(client, model="m"))
    skill = tutor.next_skill()
    tutor.ask(skill)
    tutor.ask(skill)
    assert '["Q1"]' in client.requests[1]["messages"][0]["content"]


def test_learner_moves_to_next_skill_after_mastery(fake_client):
    client = fake_client(lambda messages: question_json(a="1/2"))
    tutor = Tutor(Curriculum.load(FRACTIONS), TutorLLM(client, model="m"))
    for _ in range(4):
        tutor.submit(tutor.ask(tutor.next_skill()), "1/2")
    assert tutor.mastery("frac_concept") >= 0.95
    assert tutor.next_skill().id in {"frac_equivalent", "frac_add_same"}


def test_state_roundtrip(tmp_path):
    path = tmp_path / "alice.json"
    LearnerState({"frac_concept": 0.7}, {"frac_concept": 3}, ["Q"]).save(path)
    assert LearnerState.load(path) == LearnerState({"frac_concept": 0.7}, {"frac_concept": 3}, ["Q"])
    assert LearnerState.load(tmp_path / "new.json") == LearnerState()
