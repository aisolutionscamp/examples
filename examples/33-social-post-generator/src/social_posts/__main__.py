"""Usage: python -m social_posts briefs/coffee_shop.json --platforms instagram,x --variants 3 --out posts.csv"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .generator import PLATFORMS, Brief, PostGenerator
from .schedule import build_schedule, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate social media posts from a brief.")
    parser.add_argument("brief", type=Path)
    parser.add_argument("--platforms", default="instagram,facebook,linkedin,x")
    parser.add_argument("--variants", type=int, default=3, help="posts per platform (one per day)")
    parser.add_argument("--start", type=date.fromisoformat, default=date.today() + timedelta(days=1))
    parser.add_argument("--out", type=Path, default=Path("posts.csv"))
    args = parser.parse_args()

    load_dotenv()
    brief = Brief(**json.loads(args.brief.read_text(encoding="utf-8")))
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    unknown = set(platforms) - PLATFORMS.keys()
    if unknown:
        parser.error(f"unknown platform(s): {', '.join(sorted(unknown))}; choose from {', '.join(PLATFORMS)}")

    generator = PostGenerator(OpenAI())
    posts = {p: generator.generate_variants(p, brief, args.variants) for p in platforms}
    schedule = build_schedule(posts, args.start)
    write_csv(schedule, args.out)
    for p in schedule:
        print(f"--- {p.platform} {p.publish_at:%a %Y-%m-%d %H:%M} ({len(p.text)} chars)\n{p.text}\n")
    print(f"Saved {len(schedule)} posts to {args.out}")


if __name__ == "__main__":
    main()
