import re
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Accepts watch?v=, youtu.be/, /shorts/, /embed/, /live/ URLs or a bare 11-character id."""
    value = url_or_id.strip()
    if _ID.match(value):
        return value
    parsed = urlparse(value if "://" in value else "https://" + value)
    host = (parsed.hostname or "").removeprefix("www.").removeprefix("m.")
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1]
    if not _ID.match(candidate):
        raise ValueError(f"not a YouTube video URL or id: {url_or_id!r}")
    return candidate
