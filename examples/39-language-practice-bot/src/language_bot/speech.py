"""Optional speech: faster-whisper for input, Piper (local) or the OpenAI Speech API for output.

Note on "pronunciation feedback": Whisper's per-word probability says how sure the recognizer was, not
how correct the pronunciation was. Low-confidence words are a useful hint ("say this word again"), not a
pronunciation score.
"""
import os
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transcript:
    text: str
    unclear_words: list[str]


def load_whisper(size: str | None = None):
    from faster_whisper import WhisperModel

    return WhisperModel(size or os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def transcribe(audio_path: str | Path, language_code: str, model=None, threshold: float = 0.5) -> Transcript:
    """language_code is ISO 639-1 ("es", "fr"). Forcing it avoids misdetection of learner speech."""
    model = model or load_whisper()
    segments, _info = model.transcribe(str(audio_path), language=language_code, word_timestamps=True)
    words, unclear = [], []
    for segment in segments:
        for word in segment.words or []:
            token = word.word.strip()
            words.append(token)
            if word.probability < threshold:
                unclear.append(token.strip(".,;:!?¿¡"))
    return Transcript(text=" ".join(words).strip(), unclear_words=[w for w in unclear if w])


def speak_piper(text: str, out_path: str | Path, voice_path: str | None = None, voice=None) -> Path:
    if voice is None:
        from piper import PiperVoice

        voice = PiperVoice.load(voice_path or os.environ["PIPER_VOICE"])
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return Path(out_path)


def speak_openai(text: str, out_path: str | Path, client=None, voice: str = "marin") -> Path:
    """OpenAI Speech API. The tts models are deprecated and shut down on 2027-01-06."""
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    with client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=text, response_format="wav"
    ) as response:
        response.stream_to_file(out_path)
    return Path(out_path)


def speak(text: str, out_path: str | Path, backend: str | None = None, **kwargs) -> Path:
    backend = backend or os.getenv("TTS_BACKEND", "piper")
    if backend == "piper":
        return speak_piper(text, out_path, **kwargs)
    if backend == "openai":
        return speak_openai(text, out_path, **kwargs)
    raise ValueError(f"unknown TTS backend {backend!r}")
