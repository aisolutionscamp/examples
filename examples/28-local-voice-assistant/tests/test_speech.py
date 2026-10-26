import sys
import types
import wave

from voice_assistant.speech import Speaker, Transcriber


def test_transcriber_joins_segments_and_uses_vad(monkeypatch):
    calls = {}

    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            calls["init"] = (name, device, compute_type)

        def transcribe(self, path, language, vad_filter):
            calls["transcribe"] = (path, language, vad_filter)
            seg = types.SimpleNamespace
            return iter([seg(text=" Good morning."), seg(text=" How are you? ")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisper))
    assert Transcriber("small").transcribe("in.wav") == "Good morning. How are you?"
    assert calls["init"] == ("small", "cpu", "int8")
    assert calls["transcribe"] == ("in.wav", "en", True)


def test_speaker_writes_wav_with_slower_speech(monkeypatch, tmp_path):
    import piper

    seen = {}

    class FakeVoice:
        def synthesize_wav(self, text, wav_file, syn_config):
            seen["length_scale"] = syn_config.length_scale
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 2205)

    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda path: FakeVoice()))
    out = Speaker("voice.onnx").speak("Hello", str(tmp_path / "o.wav"))
    with wave.open(out) as w:
        assert w.getnframes() == 2205
    assert seen["length_scale"] == 1.15
