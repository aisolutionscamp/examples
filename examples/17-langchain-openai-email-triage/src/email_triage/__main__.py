"""Usage: python -m email_triage path/to/message.eml"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import from_env
from .preprocess import parse_email
from .zendesk import to_zendesk_ticket

POLICIES = Path(__file__).resolve().parents[2] / "data" / "policies.json"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    load_dotenv()
    raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    triage = from_env(json.loads(POLICIES.read_text(encoding="utf-8")))
    result = triage.process(raw)
    parsed = parse_email(raw)
    print(json.dumps({"triage": result.model_dump(),
                      "zendesk_payload": to_zendesk_ticket(result, parsed.subject, parsed.sender)}, indent=2))


if __name__ == "__main__":
    main()
