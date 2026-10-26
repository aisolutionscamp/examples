"""Speech-to-text with faster-whisper and text-to-speech with Piper. Both run on the device.

The models are loaded lazily, so the rest of the assistant (and the tests) work without them.
"""
from __future__ import annotations

import wave


class Transcriber:
    def __init__(self, model: str = "small", language: str | None = "en", device: str = "cpu"):
        self.model_name, self.language, self.device = model, language, device
        self._model = None

    def transcribe(self, wav_path: str) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel

            # int8 keeps a "small" model usable on a mini PC or Raspberry Pi 5 class device
            self._model = WhisperModel(self.model_name, device=self.device, compute_type="int8")
        segments, _info = self._model.transcribe(wav_path, language=self.language, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


class Speaker:
    def __init__(self, voice_path: str, length_scale: float = 1.15):
        self.voice_path = voice_path
        self.length_scale = length_scale  # > 1.0 speaks a little slower than the voice default
        self._voice = None

    def speak(self, text: str, out_path: str) -> str:
        """Write the spoken reply to a WAV file and return its path (playback is left to the device)."""
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.voice_path)
        from piper import SynthesisConfig

        with wave.open(out_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig(length_scale=self.length_scale))
        return out_path
