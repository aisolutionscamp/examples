import hashlib

import httpx

from password_tool.hibp import PwnedPasswordsClient


def make_client(body_for_prefix, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        prefix = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, text=body_for_prefix(prefix))

    return PwnedPasswordsClient(httpx.Client(transport=httpx.MockTransport(handler)))


def sha1(pw):
    return hashlib.sha1(pw.encode()).hexdigest().upper()


def test_only_prefix_is_sent_and_count_is_parsed():
    digest = sha1("password")
    seen = []
    body = f"0018A45C4D1DEF81644B54AB7F969B88D65:3\r\n{digest[5:]}:52256179\r\nFFFFF00000000000000000000000000000:0"
    client = make_client(lambda prefix: body, seen)

    assert client.pwned_count("password") == 52256179
    request = seen[0]
    assert request.url.path == f"/range/{digest[:5]}"
    assert digest[5:] not in str(request.url)
    assert request.headers["Add-Padding"] == "true"


def test_unknown_password_returns_zero():
    client = make_client(lambda prefix: "0018A45C4D1DEF81644B54AB7F969B88D65:3", [])
    assert client.pwned_count("a very unusual passphrase nobody uses") == 0


def test_padding_entry_counts_as_not_pwned():
    pw = "padding-test-value"
    client = make_client(lambda prefix: f"{sha1(pw)[5:]}:0", [])
    assert client.pwned_count(pw) == 0
