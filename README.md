# AI Solutions Camp examples

Runnable reference implementations for the use cases on [aisolutionscamp.io/projects](https://aisolutionscamp.io/projects). Each folder in `examples/` belongs to one solution on the site and contains:

```
examples/<solution>/
  README.md          what it does, how to run it, what to change for production
  solution.json      catalog text for the site (description, steps, tools, ...)
  requirements.txt   pinned dependencies (or package.json for JavaScript examples)
  src/               the code
  tests/             offline tests (external APIs are mocked)
```

`use-cases/<id>.json` holds the text of the use case (problem statement and goals) that the solutions answer.

## Running an example

```bash
cd examples/<solution>
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # if the example calls an API
pytest                 # offline tests
```

Or in Docker, exactly as CI runs it:

```bash
scripts/test-example.sh examples/<solution>
```

## About the numbers

Figures in the examples are **targets or example values**, not measured results from a production system. Measure on your own data before you rely on them.

## License

MIT
