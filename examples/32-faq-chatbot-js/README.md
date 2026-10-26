# FAQ chatbot for a small business website (plain JavaScript)

A chat widget you can drop into any website. It answers common questions from a JSON file (`faq.json`) with keyword matching. There is no AI model, no server and no build step: three static files and your FAQ.

## How it works

```
visitor question
  -> normalize     lowercase, strip accents and punctuation
  -> tokenize      drop stopwords, map synonyms ("delivery" -> "shipping"), light stemming ("prices" -> "price")
  -> score         for each FAQ entry: share of the question's words found in the entry,
                   weighted by IDF (words that appear in few entries count more)
  -> best entry    if score >= minScore (0.35) show its answer, otherwise the fallback with your contact address
  -> suggestions   the next best questions as clickable buttons
```

Files:

- `src/matcher.js`: the matching logic (works in browsers and Node.js)
- `src/chatbot-widget.js`: the chat UI (button, panel, suggestions, history in `localStorage`)
- `src/chatbot.css`: styles; change the colors with the `--bot-*` variables
- `faq.json`: knowledge base (example content for a fictional coffee roaster)
- `index.html`: demo page

## Run the demo

Any static web server works, for example:

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

(Opening `index.html` directly from disk does not work, because browsers block `fetch` for `file://` URLs.)

## Add it to your website

Copy `src/` and `faq.json` to your site, then add before `</body>`:

```html
<link rel="stylesheet" href="/chatbot/src/chatbot.css">
<script type="module">
  import { mountChatbot } from "/chatbot/src/chatbot-widget.js";
  mountChatbot({ knowledgeBaseUrl: "/chatbot/faq.json", title: "Questions? Ask us" });
</script>
```

## Write your FAQ

Each entry in `faq.json`:

```json
{ "id": "shipping-cost", "question": "How much does shipping cost?",
  "alternatives": ["What are the delivery fees?"], "keywords": ["shipping", "price", "free"],
  "answer": "Shipping within Germany costs 4.90 EUR ..." }
```

Add the words your customers really use as `alternatives`, `keywords` or `synonyms`. Look at the questions that got the fallback answer and add them.

## Tests

```bash
npm test          # node:test, no dependencies
```

The tests check the tokenizer, that the knowledge base is valid, that every FAQ question finds its own answer, and that 14 paraphrased questions in `tests/paraphrases.json` find the right entry. The paraphrases were written together with the example FAQ, so they are a regression check, not an accuracy measurement. Write your own test questions for your FAQ.

## Limitations

- English only (stopwords and stemming). For other languages, replace `STOPWORDS` and `stem()`.
- Keyword matching does not understand meaning. "Can I come by on Sunday?" only matches opening hours if "sunday" is in the entry. Questions with words that appear in no entry get the fallback.
- History is stored in the visitor's browser only; you do not see which questions were asked.

## For production

- Log unanswered questions (for example with a small endpoint or your analytics tool) and extend the FAQ from them.
- If keyword matching is not enough, keep this widget as the UI and replace `matcher.ask()` with a call to a small backend that uses embeddings or an LLM restricted to your FAQ.
- Keep answers short and link to the full policy pages.
