# Shopping list organizer

Turns a shopping note ("2 kg potatoes, milk and 6 eggs, ice cream, soap") into a checklist grouped by
category and sorted in the order you walk through *your* store.

1. **Parse** the note into items with quantity and unit (rule-based, or optionally with an LLM for messy
   text such as "stuff for pancakes" or typos).
2. **Categorize** each item with a keyword table (produce, bakery, meat & fish, dairy, pantry, beverages,
   frozen, household, other). The keyword that ends last wins, so "orange juice" is a beverage and
   "tomato sauce" is pantry; "frozen ..." always goes to frozen.
3. **Merge** duplicates with the same unit ("2 eggs" + "6 eggs" = 8 eggs).
4. **Sort** by a store layout file: a list of categories in the order of the aisles.

There are no shopping, price or store-map APIs involved. The store layout is a small JSON file you write
once per store.

## Architecture

```
note ──► rule parser ─────────────┐
     └─► LLM parser (optional) ───┴─► keyword categorizer ──► merge ──► sort by layout JSON ──► checklist
          OpenAI-compatible API        (validates LLM categories)
```

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m shopping_list "2 kg potatoes, milk and 6 eggs, ice cream, soap" --layout ../layouts/example-store.json
```

Output:

```
PRODUCE
  [ ] 2 kg potatoes
DAIRY
  [ ] 6 eggs
  [ ] 1 milk
FROZEN
  [ ] 1 ice cream
HOUSEHOLD
  [ ] 1 soap
```

With an LLM (copy `.env.example` to `.env` and export the variables, or set them in your shell):

```bash
python -m shopping_list --llm "stuff for pancakes and something for a salad"
```

Tests (offline, the LLM client is a fake): `pytest`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | needed only for `--llm` |
| `OPENAI_MODEL` | `gpt-6-luna` | chat model |
| `OPENAI_BASE_URL` | OpenAI | e.g. `http://localhost:11434/v1` for Ollama (then `OPENAI_MODEL=llama3.2`) |

Store layout: `layouts/example-store.json`. Categories missing from a layout are appended at the end.
Add items or other languages in `KEYWORDS` in `src/shopping_list/categories.py`.

## Limitations

- The keyword table is small and English only. Unknown items land in "other".
- The rule parser splits on commas, new lines and "and", so "mac and cheese" becomes two items (the LLM
  parser handles this).
- Units are not converted (1 l milk and 500 ml milk stay separate lines).
- The LLM may list ingredients you already have; it never sets prices or stores.

## For production

- Let users correct a category once and store it as a personal override (item name to category).
- Store layouts per store, edited by users, instead of a file.
- Add unit conversion and a pantry list to skip items already at home.
- Put the parser behind an API for a mobile app; keep the LLM optional so the app works offline.
