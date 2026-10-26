"""python -m adaptive_tutor ../curricula/fractions.json --learner alice"""
import argparse
from pathlib import Path

from .curriculum import Curriculum
from .llm import TutorLLM
from .tutor import LearnerState, Tutor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adaptive_tutor")
    parser.add_argument("curriculum")
    parser.add_argument("--learner", default="learner", help="progress is stored in <learner>.json")
    args = parser.parse_args(argv)

    state_path = Path(f"{args.learner}.json")
    tutor = Tutor(Curriculum.load(args.curriculum), TutorLLM(), LearnerState.load(state_path))
    print("Empty answer = stop, '?' = hint.")
    while (skill := tutor.next_skill()) is not None:
        question = tutor.ask(skill)
        print(f"\n[{skill.name}, mastery {tutor.mastery(skill.id):.0%}] {question.text}")
        answer = input("> ").strip()
        if answer == "?":
            print(f"Hint: {question.hint}")
            answer = input("> ").strip()
        if not answer:
            break
        feedback = tutor.submit(question, answer)
        print("Correct." if feedback.correct else f"Not quite. The answer is {question.answer}.")
        if feedback.explanation:
            print(feedback.explanation)
        print(f"Mastery: {feedback.mastery_before:.0%} -> {feedback.mastery_after:.0%}")
        tutor.state.save(state_path)
    else:
        print("All skills mastered.")
    tutor.state.save(state_path)


if __name__ == "__main__":
    main()
