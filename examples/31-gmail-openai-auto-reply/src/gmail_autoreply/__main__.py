"""Usage: python -m gmail_autoreply [--dry-run] [--max N] [--interval MINUTES]"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from . import gmail
from .autoreply import DEFAULT_QUERY, process_inbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft replies to unread Gmail messages with OpenAI.")
    parser.add_argument("--dry-run", action="store_true", help="classify only; no drafts, no labels")
    parser.add_argument("--max", type=int, default=20, help="messages per run")
    parser.add_argument("--interval", type=int, default=0, help="repeat every N minutes (0 = run once)")
    args = parser.parse_args()

    load_dotenv()
    business_info = Path(os.getenv("BUSINESS_INFO_FILE", "business_info.md")).read_text(encoding="utf-8")
    from_address = os.environ["GMAIL_ADDRESS"]
    service = gmail.get_service(os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json"),
                                os.getenv("GMAIL_TOKEN_FILE", "token.json"))
    client = OpenAI()
    while True:
        for o in process_inbox(service, client, business_info, from_address, os.getenv("GMAIL_QUERY", DEFAULT_QUERY),
                               args.max, args.dry_run):
            print(f"{o.category:13} draft={o.draft_id or '-':20} {o.subject}")
        if not args.interval:
            break
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
