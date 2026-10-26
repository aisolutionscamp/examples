from pathlib import Path
from types import SimpleNamespace

import pytest
from youtube_transcript_api import TranscriptsDisabled

from yt_summarizer import transcript
from yt_summarizer.transcript import Segment, get_transcript


class FakeApi:
    def __init__(self, snippets=None, error=None):
        self.snippets, self.error, self.calls = snippets or [], error, []

    def fetch(self, video_id, languages=("en",)):
        self.calls.append((video_id, languages))
        if self.error:
            raise self.error
        return self.snippets


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, language=None, vad_filter=False):
        self.calls.append((path, language, vad_filter))
        segs = [SimpleNamespace(start=0.0, text=" Hello "), SimpleNamespace(start=3.2, text="  ")]
        return iter(segs), SimpleNamespace(language="en")


def snippet(start, text):
    return SimpleNamespace(start=start, duration=2.0, text=text)


def test_captions_are_cleaned():
    api = FakeApi([snippet(0.0, "first\nline"), snippet(2.5, " "), snippet(4.0, "second")])
    segments, source = get_transcript("abcdefghijk", ("de", "en"), api=api)
    assert source == "captions"
    assert segments == [Segment(0.0, "first line"), Segment(4.0, "second")]
    assert api.calls == [("abcdefghijk", ("de", "en"))]


def test_missing_captions_raise_without_whisper():
    with pytest.raises(TranscriptsDisabled):
        get_transcript("abcdefghijk", api=FakeApi(error=TranscriptsDisabled("abcdefghijk")))


def test_whisper_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript, "download_audio", lambda vid, out: Path(out) / f"{vid}.webm")
    whisper = FakeWhisper()
    segments, source = get_transcript(
        "abcdefghijk", ("en",), use_whisper=True,
        api=FakeApi(error=TranscriptsDisabled("abcdefghijk")), whisper_model=whisper)
    assert source == "whisper"
    assert segments == [Segment(0.0, "Hello")]
    path, language, vad = whisper.calls[0]
    assert path.endswith("abcdefghijk.webm") and language == "en" and vad is True
