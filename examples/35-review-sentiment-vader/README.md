# Restaurant review sentiment with VADER

Reads restaurant reviews from a CSV file and reports how positive or negative they are, overall and per topic
(food, service, ambience, price, waiting time, cleanliness). No training data, no API key, no network.

## How it works

```
reviews.csv ──► VADER compound score per review (-1 .. 1) ──► positive / neutral / negative
           └─► split into clauses (". ! ? ;" and "but", "however", "although")
               ──► clause mentions a topic keyword? ──► VADER score of that clause for the topic
──► report: counts and shares, mean score, per-topic mentions / mean / negative share, most negative reviews
```

- [VADER](https://github.com/cjhutto/vaderSentiment) is a rule-based sentiment model for short English texts. Its
  lexicon ships inside the `vaderSentiment` package. It handles negation, intensifiers ("very"), capitals and
  punctuation.
- `DOMAIN_LEXICON` adds restaurant words that VADER scores as neutral ("overpriced", "bland", "slow", "soggy",
  ...). Plain VADER rates "Overpriced for tiny portions." as neutral; with the lexicon it is negative.
- Labels use VADER's standard cut-offs: compound >= 0.05 positive, <= -0.05 negative, otherwise neutral.

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m reviews ../samples/sample_reviews.csv --out-csv ../results.csv --out-json ../report.json
python -m reviews my_export.csv --text-column comment
```

`report.json` contains `counts`, `shares`, `mean_compound`, `aspects` (per topic: `mentions`, `mean`,
`negative_share`) and `most_negative`. `results.csv` has one row per review with its score, label and topic
scores; charts can be made from it in any spreadsheet tool.

Tests (bundled sample reviews): `pytest`.

## Configuration

- `ASPECTS` in `analyzer.py`: topic keyword lists. Add your dishes and venue terms ("patio", "brunch").
- `DOMAIN_LEXICON`: extra words with scores from -4 to +4, or `ReviewAnalyzer(extra_lexicon={...})`.
- `POSITIVE`, `NEGATIVE`: label cut-offs.

## Limitations

- English only. VADER does not support other languages; translate first or use a multilingual model.
- Mixed reviews: VADER weights the part after "but" more. "The food was amazing ... but quite noisy" (4 stars in
  the sample) comes out negative overall; the topic scores still show positive food and negative ambience.
- Sarcasm and comparisons ("better than last time, which was awful") are often misread.
- Topic detection is keyword matching; a clause can mention a topic without being about it.

## For production

- Compare the labels with star ratings or a hand-labeled sample of your own reviews before relying on the shares.
- If agreement is too low, train a classifier on labeled reviews (for example TF-IDF + logistic regression in
  scikit-learn) or use a transformer sentiment model, and keep the clause/topic logic.
- Run it on new reviews every week and track topic scores over time rather than single reviews.
