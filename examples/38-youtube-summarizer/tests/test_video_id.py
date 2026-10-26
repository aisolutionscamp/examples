import pytest

from yt_summarizer.video_id import extract_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("value", [
    VID,
    f"https://www.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/watch?feature=share&v={VID}&t=42s",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"youtube.com/live/{VID}",
])
def test_valid(value):
    assert extract_video_id(value) == VID


@pytest.mark.parametrize("value", ["https://example.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/short", "hello"])
def test_invalid(value):
    with pytest.raises(ValueError):
        extract_video_id(value)
