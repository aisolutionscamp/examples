import csv
from datetime import date, datetime
from types import SimpleNamespace as NS

import pytest

from social_posts.generator import PLATFORMS, Brief, PostGenerator, PostParts, clean_hashtags, render
from social_posts.schedule import build_schedule, next_weekdays, write_csv

BRIEF = Brief(business="Example Coffee Roasters", business_type="coffee roastery", audience="coffee lovers",
              post_type="product", details="New Colombian coffee, 250 g for 12.50 EUR.",
              link="https://example.com/shop/huila")


def parts(body="Notes of apple and caramel. 250 g for 12.50 EUR.", tags=("CoffeeLovers", "#coffee lovers", "Huila")):
    return PostParts(hook="Meet our new coffee from Huila.", body=body,
                     call_to_action="Try it this week.", hashtags=list(tags))


class FakeOpenAI:
    def __init__(self, outputs):
        self.outputs, self.requests = list(outputs), []
        self.responses = NS(parse=self._parse)

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        return NS(output_parsed=self.outputs.pop(0))


def test_clean_hashtags():
    assert clean_hashtags(["#Coffee Lovers", "coffeelovers", "", "#", "Café"], 5) == ["CoffeeLovers", "Café"]
    assert clean_hashtags(["a", "b", "c"], 2) == ["a", "b"]


def test_instagram_template():
    post = render(PLATFORMS["instagram"], BRIEF, parts())
    assert post.startswith("Meet our new coffee from Huila.\n\nNotes of apple")
    assert "Try it this week. (link in bio)" in post
    assert post.endswith("#CoffeeLovers #Huila")  # duplicate "coffee lovers" removed
    assert "https://" not in post  # Instagram captions have no clickable links


def test_x_template_is_one_line_with_link():
    post = render(PLATFORMS["x"], BRIEF, parts())
    assert "\n" not in post
    assert post == ("Meet our new coffee from Huila. Notes of apple and caramel. 250 g for 12.50 EUR. "
                    "Try it this week. https://example.com/shop/huila #CoffeeLovers #Huila")


def test_generate_sends_platform_rules_and_brief():
    ai = FakeOpenAI([parts()])
    post = PostGenerator(ai, model="test-model").generate("linkedin", BRIEF)
    req = ai.requests[0]
    assert req["model"] == "test-model" and req["text_format"] is PostParts
    assert "Platform: linkedin" in req["instructions"] and "under 3000 characters" in req["instructions"]
    assert "12.50 EUR" in req["input"]
    assert "https://example.com/shop/huila" in post


def test_too_long_post_is_retried_with_feedback():
    long_body = "Very long text. " * 30
    ai = FakeOpenAI([parts(body=long_body), parts()])
    post = PostGenerator(ai).generate("x", BRIEF)
    assert len(post) <= 280
    assert "the limit is 280" in ai.requests[1]["input"]


def test_gives_up_after_max_attempts():
    ai = FakeOpenAI([parts(body="x" * 300)] * 2)
    with pytest.raises(ValueError, match="under 280"):
        PostGenerator(ai, max_attempts=2).generate("x", BRIEF)


def test_variants_are_unique():
    ai = FakeOpenAI([parts(), parts(), parts(body="Different text.")])
    posts = PostGenerator(ai).generate_variants("facebook", BRIEF, 2)
    assert len(posts) == 2 and posts[0] != posts[1]


def test_schedule_skips_weekends_and_writes_csv(tmp_path):
    assert next_weekdays(date(2026, 10, 30), 3) == [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)]
    schedule = build_schedule({"x": ["a", "b"], "linkedin": ["c"]}, date(2026, 10, 30))
    assert [(p.platform, p.publish_at) for p in schedule] == [
        ("linkedin", datetime(2026, 10, 30, 8, 30)), ("x", datetime(2026, 10, 30, 9, 0)),
        ("x", datetime(2026, 11, 2, 9, 0))]
    out = tmp_path / "posts.csv"
    write_csv(schedule, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0] == {"platform": "linkedin", "publish_at": "2026-10-30T08:30", "characters": "1", "text": "c"}
