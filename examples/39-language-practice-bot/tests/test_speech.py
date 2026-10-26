import wave
from types import SimpleNamespace

import pytest

from language_bot import speech


class FakeWhisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        words = [SimpleNamespace(word=" Quiero", probability=0.95), SimpleNamespace(word=" una", probability=0.9),
                 SimpleNamespace(word=" paella.", probability=0.31)]
        return iter([SimpleNamespace(words=words)]), SimpleNamespace(language="es")


def test_transcribe_marks_low_confidence_words():
    model = FakeWhisper()
    result = speech.transcribe("in.wav", "es", model=model)
    assert result.text == "Quiero una paella."
    assert result.unclear_words == ["paella"]
    assert model.kwargs == {"language": "es", "word_timestamps": True}


class FakePiperVoice:
    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * len(text))


def test_piper_writes_wav(tmp_path):
    out = speech.speak("Hola", tmp_path / "r.wav", backend="piper", voice=FakePiperVoice())
    with wave.open(str(out)) as wav:
        assert wav.getnframes() == 4


class FakeStream:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream_to_file(self, path):
        self.log.append(path)


def test_openai_tts_request(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_TTS_MODEL", raising=False)
    log, calls = [], []

    def create(**kwargs):
        calls.append(kwargs)
        return FakeStream(log)

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(
        with_streaming_response=SimpleNamespace(create=create))))
    speech.speak("Hola", tmp_path / "r.wav", backend="openai", client=client)
    assert calls == [{"model": "gpt-4o-mini-tts", "voice": "marin", "input": "Hola", "response_format": "wav"}]
    assert log == [tmp_path / "r.wav"]


def test_unknown_backend():
    with pytest.raises(ValueError):
        speech.speak("x", "x.wav", backend="nope")
