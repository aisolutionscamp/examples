# Social media post generator (templates + OpenAI)

Turns a short brief about your business into ready-to-post texts for Instagram, Facebook, LinkedIn and X. The OpenAI model writes the parts of a post (hook, body, call to action, hashtags) as structured output; a Jinja2 template per platform puts them together. Posts that break a platform's length limit are regenerated with feedback. The result is a CSV posting schedule you can review and import into a scheduling tool.

## Architecture

```
brief (JSON: business, audience, tone, post type, details, link)
  -> OpenAI Responses API   responses.parse(..., text_format=PostParts)   per platform, with its style and limit
  -> clean_hashtags         strip "#", spaces and duplicates, limit per platform
  -> Jinja2 template        templates/<platform>.j2 (Instagram: "link in bio", X: one line)
  -> length check           over the limit -> ask again with feedback (max 3 attempts)
  -> schedule               one post per platform per weekday at a fixed time
  -> posts.csv              platform, publish_at, characters, text
```

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env     # OPENAI_API_KEY
PYTHONPATH=src python -m social_posts briefs/coffee_shop.json --platforms instagram,linkedin,x --variants 3
```

The posts are printed and saved to `posts.csv`. Write your own brief by copying `briefs/coffee_shop.json`.

Tests (offline; the OpenAI client is a fake):

```bash
pytest
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `OPENAI_API_KEY` | `.env` | required |
| `OPENAI_MODEL` | `.env` | `gpt-6-luna` |
| Platform limits, hashtag counts, style hints | `PLATFORMS` in `generator.py` | see code |
| Post layout per platform | `src/social_posts/templates/*.j2` | |
| Posting times | `DEFAULT_SLOTS` in `schedule.py` | Instagram 18:00, Facebook 12:00, LinkedIn 8:30, X 9:00 |

## Limitations

- The tool does not publish anything. Import the CSV into your scheduling tool (for example Meta Business Suite, Buffer or Hootsuite, depending on the import format they accept) or copy the texts by hand.
- The length check counts Python characters. X counts links as 23 characters and some characters (for example emoji) as two, so stay well below 280 for X.
- The model is told to use only facts from the brief, but it can still exaggerate. Read every post before it goes out.
- No images. Pair the text with your own photos.
- Engagement effects have not been measured. Compare posts with your platform's analytics.

## For production

- Add a review step (spreadsheet or small web form) where someone approves each post before it is scheduled.
- Publish through the official APIs (Meta Graph API for Instagram and Facebook pages, LinkedIn Posts API, X API). Each needs an app review or paid access.
- Keep a list of past posts and pass recent ones to the prompt to avoid repetition.
