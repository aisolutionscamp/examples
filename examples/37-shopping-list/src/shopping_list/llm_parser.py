"""Optional: let an LLM parse messy free text (typos, recipes, "stuff for pancakes").

Works with OpenAI or any OpenAI-compatible server (Ollama, vLLM). The model only proposes items;
categories it returns are validated, and unknown ones fall back to the keyword categorizer.
"""
import json
import os

from .categories import CATEGORIES, categorize
from .parser import Item, parse_text

SYSTEM_PROMPT = (
    "You turn a shopping note into a JSON object {\"items\": [{\"name\": str, \"quantity\": number, "
    "\"unit\": str or null, \"category\": str}]}. Allowed categories: " + ", ".join(CATEGORIES) + ". "
    "Use singular, lowercase English item names. If the note names a dish, list its usual ingredients. "
    "Do not invent prices, brands or stores. Answer with JSON only."
)


def make_client():
    from openai import OpenAI  # imported lazily so the rule-based path has no API dependency

    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)


def parse_with_llm(text: str, client=None, model: str | None = None) -> list[Item]:
    client = client or make_client()
    model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        response_format={"type": "json_object"},
    )
    try:
        raw_items = json.loads(response.choices[0].message.content)["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return parse_text(text)  # unusable answer: fall back to the rule-based parser

    items = []
    for raw in raw_items:
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            continue
        name = str(raw["name"]).strip().lower()
        try:
            quantity = float(raw.get("quantity") or 1)
        except (TypeError, ValueError):
            quantity = 1.0
        category = raw.get("category")
        if category not in CATEGORIES:
            category = categorize(name)
        items.append(Item(name=name, quantity=quantity, unit=raw.get("unit") or None, category=category))
    return items
