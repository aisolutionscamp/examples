import base64
from email import message_from_bytes, policy

from fakes import FakeGmailService, FakeOpenAI, gmail_message

from gmail_autoreply import gmail
from gmail_autoreply.autoreply import PROCESSED_LABEL, Decision, process_inbox

SHOP = "shop@example.com"
INFO = "Business name: Example Bike Shop. City bikes from 499 EUR."


def decision(category, reply=None):
    return Decision(category=category, needs_reply=reply is not None, reply=reply, reason="test")


def decode_draft(draft_body):
    raw = draft_body["message"]["raw"]
    return message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)


def test_inquiry_gets_threaded_draft_and_labels():
    service = FakeGmailService([gmail_message("m1", "Ann <ann@example.org>", "Price of city bikes?",
                                              "How much is a city bike?", html=True)])
    ai = FakeOpenAI([decision("pricing", "Hi Ann, city bikes start at 499 EUR. Example Bike Shop")])

    outcomes = process_inbox(service, ai, INFO, SHOP)

    assert [(o.category, o.draft_id) for o in outcomes] == [("pricing", "d1")]
    draft = service.created_drafts[0]
    assert draft["message"]["threadId"] == "t-m1"
    msg = decode_draft(draft)
    assert msg["To"] == "Ann <ann@example.org>" and msg["Subject"] == "Re: Price of city bikes?"
    assert msg["In-Reply-To"] == "<m1@mail.example.com>"
    assert "499 EUR" in msg.get_content()
    # The plain-text part (not the HTML one) and the business info went to the model.
    req = ai.requests[0]
    assert "How much is a city bike?" in req["input"] and "<p>" not in req["input"]
    assert INFO in req["instructions"] and req["text_format"] is Decision
    labels = {label["name"]: label["id"] for label in service.labels}
    assert service.modified == [("m1", [labels[PROCESSED_LABEL], labels["AI/pricing"]])]


def test_newsletter_gets_no_draft_even_if_model_wrote_one():
    service = FakeGmailService([gmail_message("m2", "news@example.net", "Weekly deals", "Big sale")])
    ai = FakeOpenAI([Decision(category="newsletter", needs_reply=True, reply="Thanks!", reason="x")])
    outcomes = process_inbox(service, ai, INFO, SHOP)
    assert outcomes[0].draft_id is None and service.created_drafts == []
    assert service.modified[0][0] == "m2"


def test_own_messages_are_skipped_and_dry_run_changes_nothing():
    service = FakeGmailService([gmail_message("m3", f"Shop <{SHOP}>", "Fwd", "internal"),
                                gmail_message("m4", "bob@example.org", "Open on Saturday?", "Are you open?")])
    ai = FakeOpenAI([decision("other", "Yes, 10:00-14:00.")])
    outcomes = process_inbox(service, ai, INFO, SHOP, dry_run=True)
    assert [o.message_id for o in outcomes] == ["m4"] and len(ai.requests) == 1
    assert service.created_drafts == [] and service.modified == []
    assert service.labels == [{"id": "INBOX", "name": "INBOX"}]


def test_query_excludes_processed_mail():
    service = FakeGmailService([])
    process_inbox(service, FakeOpenAI([]), INFO, SHOP)
    assert f'-label:"{PROCESSED_LABEL}"' in service.queries[0]


def test_existing_label_is_reused_and_re_prefix_kept():
    service = FakeGmailService([gmail_message("m5", "c@example.org", "Re: order 1234", "Where is it?")])
    service.labels.append({"id": "Lx", "name": PROCESSED_LABEL})
    process_inbox(service, FakeOpenAI([decision("order_status", "Check https://example.com/orders")]), INFO, SHOP)
    assert sum(label["name"] == PROCESSED_LABEL for label in service.labels) == 1
    assert decode_draft(service.created_drafts[0])["Subject"] == "Re: order 1234"


def test_plain_text_fallback_to_snippet():
    msg = gmail_message("m6", "d@example.org", "Hi", "snippet text")
    msg["payload"] = {"mimeType": "text/html", "body": {"data": ""}, "headers": msg["payload"]["headers"]}
    assert gmail.get_email(FakeGmailService([msg]), "m6").body == "snippet text"
