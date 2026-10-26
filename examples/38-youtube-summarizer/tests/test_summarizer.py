import pytest

from yt_summarizer.summarizer import Summarizer, chunk_transcript, timestamp
from yt_summarizer.transcript import Segment


def test_timestamp():
    assert timestamp(5) == "00:05"
    assert timestamp(754.9) == "12:34"
    assert timestamp(3725) == "1:02:05"


def test_chunks_respect_size_and_keep_timestamps():
    segments = [Segment(i * 10.0, "word " * 20) for i in range(30)]
    chunks = chunk_transcript(segments, max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0].startswith("[00:00] ")
    assert sum(c.count("[") for c in chunks) == 30


def test_short_transcript_uses_one_request(fake_client):
    client = fake_client()
    result = Summarizer(client, model="m").summarize([Segment(0, "Intro"), Segment(65, "Main idea")], title="T")
    assert result == "summary"
    assert len(client.requests) == 1
    messages = client.requests[0]["messages"]
    assert '"T"' in messages[0]["content"]
    assert messages[1]["content"] == "[00:00] Intro\n[01:05] Main idea"


def test_long_transcript_is_map_reduced(fake_client):
    def reply(messages):
        if messages[0]["content"].startswith("You get part"):
            first_line = messages[1]["content"].splitlines()[0]
            return "note " + first_line.split("]")[0] + "]"
        return "final"

    client = fake_client(reply)
    segments = [Segment(i * 30.0, "x" * 90) for i in range(20)]
    result = Summarizer(client, model="m", max_chars=400).summarize(segments)
    n_chunks = len(chunk_transcript(segments, 400))
    assert result == "final"
    assert len(client.requests) == n_chunks + 1
    final_input = client.requests[-1]["messages"][1]["content"]
    assert final_input.startswith("note [00:00]")
    assert final_input.count("note [") == n_chunks


def test_model_from_env(monkeypatch, fake_client):
    monkeypatch.setenv("OPENAI_MODEL", "llama3.2")
    client = fake_client()
    Summarizer(client).summarize([Segment(0, "a")])
    assert client.requests[0]["model"] == "llama3.2"


def test_empty_transcript(fake_client):
    with pytest.raises(ValueError):
        Summarizer(fake_client(), model="m").summarize([])
