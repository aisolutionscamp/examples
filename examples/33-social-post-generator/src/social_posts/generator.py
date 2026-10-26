"""Generate social media posts: OpenAI writes the parts, Jinja2 templates assemble them per platform."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-6-luna"
TEMPLATES = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Platform:
    name: str
    max_chars: int
    max_hashtags: int
    style: str


# Instagram (2,200), LinkedIn (3,000) and X (280) are the platform limits. Facebook allows far more;
# 2,000 and all hashtag counts are editorial choices. X counts every link as 23 characters.
PLATFORMS = {
    "instagram": Platform("instagram", 2200, 10, "visual, warm, emojis allowed but at most two"),
    "facebook": Platform("facebook", 2000, 3, "friendly and conversational, one clear call to action"),
    "linkedin": Platform("linkedin", 3000, 3, "professional, no emojis, short paragraphs"),
    "x": Platform("x", 280, 2, "very short and direct, one sentence plus call to action"),
}


class PostParts(BaseModel):
    hook: str = Field(description="First line that makes people stop scrolling")
    body: str = Field(description="Main text, 1-3 short sentences, only facts from the brief")
    call_to_action: str = Field(description="What the reader should do next")
    hashtags: list[str] = Field(description="Relevant hashtags without the # sign")


class Brief(BaseModel):
    business: str
    business_type: str
    audience: str
    tone: str = "friendly"
    post_type: str = Field(description="product, event, tip or announcement")
    details: str
    link: str | None = None


INSTRUCTIONS = """You write social media posts for a small business.
Platform: {platform} (style: {style}). The finished post must stay under {max_chars} characters.
Use only facts from the brief. Do not invent prices, dates, discounts or claims.
Return at most {max_hashtags} hashtags."""

_env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                   trim_blocks=True, lstrip_blocks=True, autoescape=False)  # plain text, not HTML


def clean_hashtags(tags: list[str], limit: int) -> list[str]:
    """'#Coffee Lovers' -> 'CoffeeLovers'; drop duplicates (case-insensitive) and empty tags."""
    seen, result = set(), []
    for tag in tags:
        cleaned = re.sub(r"[^\w]", "", tag.replace(" ", ""), flags=re.UNICODE)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result[:limit]


def render(platform: Platform, brief: Brief, parts: PostParts) -> str:
    template = _env.get_template(f"{platform.name}.j2")
    text = template.render(brief=brief, parts=parts, hashtags=clean_hashtags(parts.hashtags, platform.max_hashtags))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class PostGenerator:
    def __init__(self, client: OpenAI, model: str | None = None, max_attempts: int = 3):
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.max_attempts = max_attempts

    def _parts(self, platform: Platform, brief: Brief, feedback: str | None) -> PostParts:
        prompt = f"Brief:\n{brief.model_dump_json(indent=2)}"
        if feedback:
            prompt += f"\n\nYour previous version was rejected: {feedback}"
        response = self.client.responses.parse(
            model=self.model,
            instructions=INSTRUCTIONS.format(platform=platform.name, style=platform.style,
                                             max_chars=platform.max_chars, max_hashtags=platform.max_hashtags),
            input=prompt,
            text_format=PostParts,
        )
        if response.output_parsed is None:
            raise RuntimeError("model returned no parsable post")
        return response.output_parsed

    def generate(self, platform_name: str, brief: Brief) -> str:
        """Generate one post; retry with feedback if it breaks the platform's length limit."""
        platform = PLATFORMS[platform_name]
        feedback = None
        for _ in range(self.max_attempts):
            post = render(platform, brief, self._parts(platform, brief, feedback))
            if len(post) <= platform.max_chars:
                return post
            feedback = f"it had {len(post)} characters, the limit is {platform.max_chars}. Make it shorter."
        raise ValueError(f"could not produce a {platform_name} post under {platform.max_chars} characters")

    def generate_variants(self, platform_name: str, brief: Brief, count: int) -> list[str]:
        posts: list[str] = []
        for _ in range(count * 2):  # extra attempts in case the model repeats itself
            post = self.generate(platform_name, brief)
            if post not in posts:
                posts.append(post)
            if len(posts) == count:
                break
        return posts
