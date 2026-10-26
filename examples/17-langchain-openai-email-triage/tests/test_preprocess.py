from email_triage.preprocess import parse_email, strip_quotes_and_signature

RAW = """From: Max <max@example.com>
Subject: Refund for order #AB-123456
Content-Type: text/plain; charset=utf-8

Hello, I want a refund for order AB-123456. The app shows error E-5031.
Account ID: ACC-77812

--
Max

On Mon, 3 Mar 2026 someone wrote:
> old text
"""


def test_parse_rfc822_extracts_fields_and_identifiers():
    p = parse_email(RAW)
    assert p.subject == "Refund for order #AB-123456"
    assert "max@example.com" in p.sender
    assert p.order_numbers == ["AB-123456"]
    assert p.account_ids == ["ACC-77812"]
    assert p.error_codes == ["E-5031"]
    assert "old text" not in p.body and "Max" not in p.body.splitlines()[-1]


def test_plain_text_input():
    p = parse_email("The dashboard returns error 502 since this morning")
    assert p.subject == "" and p.error_codes == ["502"]


def test_strip_quotes_drops_quoted_lines():
    body = "New question\n> quoted line\nsecond line\n-----Original Message-----\nold"
    assert strip_quotes_and_signature(body) == "New question\nsecond line"
