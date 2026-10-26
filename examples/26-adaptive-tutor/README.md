# Adaptive tutor with Bayesian knowledge tracing

A terminal tutor that keeps a mastery estimate per skill, always practises the weakest skill whose
prerequisites are mastered, and uses an LLM to write practice questions at a matching difficulty and to
explain wrong answers.

- **Skill model:** a curriculum JSON lists skills and their prerequisites (example: fractions, 6 skills).
- **Mastery tracking:** [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing)
  (Corbett and Anderson, 1994). Every answer updates P(mastered) with slip and guess probabilities, then
  adds the chance of learning from the practice. A skill counts as mastered at 0.95.
- **Next skill:** the least-mastered skill whose prerequisites are all mastered.
- **Questions:** an OpenAI-compatible chat model (OpenAI, or a local model via Ollama) writes one question
  with a short answer and a hint as JSON. Difficulty follows mastery (easy below 0.4, medium below 0.75,
  hard above). The last questions are passed back to avoid repeats.
- **Grading is local:** numbers and fractions are compared by value (`6/8` = `3/4` = `0.75`, `1 1/2`),
  words case-insensitively. The LLM does not decide whether an answer is right.
- **Explanations:** after a wrong answer the model explains the steps and the likely misconception.

Progress is stored in `<learner>.json`.

## Architecture

```
curriculum.json ─► next skill (prerequisites + mastery) ─► LLM: question JSON ─► learner answers
                         ▲                                                          │
                learner.json ◄── BKT update ◄── local grading ◄──────────────────────┘
                                                     └─ wrong ─► LLM: explanation
```

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env    # then export the variables
cd src
python -m adaptive_tutor ../curricula/fractions.json --learner alice
```

Tests (offline, the chat client is a fake; BKT math is checked against hand calculations): `pytest`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | OpenAI key (any string for Ollama) |
| `OPENAI_MODEL` | `gpt-6-luna` | chat model |
| `OPENAI_BASE_URL` | OpenAI | e.g. `http://localhost:11434/v1` for Ollama (`OPENAI_MODEL=llama3.2`) |

BKT parameters: `BKTParams(p_init, p_learn, p_slip, p_guess)` in `src/adaptive_tutor/bkt.py`
(defaults 0.2, 0.15, 0.1, 0.2). Your own subject: write a curriculum JSON with `subject` and `skills`.

## Limitations

- The default BKT parameters are plausible starting values, not fitted. Fit them per skill on real answer
  logs (for example with the `pyBKT` library) and check the predictions with `p_correct`.
- The LLM can produce a wrong expected answer or an ambiguous question. Small local models do this more
  often. For math, generate questions from templates or verify the answer with code (e.g. `fractions`).
- Only short answers are graded; free-text answers would need a rubric and are not supported.
- "Learning style" is not modelled: there is no good evidence that matching content to learning styles
  helps; the tutor adapts to demonstrated mastery instead.

## For production

- Store learner state in a database, with every answer logged for parameter fitting and review.
- Have teachers review or approve generated questions, or keep a vetted question bank and use the LLM
  only for hints and explanations.
- Add spaced repetition for mastered skills, since BKT has no forgetting.
- Follow data-protection rules for minors (consent, data minimization, no training on learner data
  without agreement).
