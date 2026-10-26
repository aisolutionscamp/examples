"""Get a timed transcript: YouTube captions first, local Whisper transcription as fallback."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException


@dataclass
class Segment:
    start: float  # seconds
    text: str


def fetch_captions(video_id: str, languages: tuple[str, ...] = ("en",), api=None) -> list[Segment]:
    """Manual or auto-generated captions via youtube-transcript-api (no API key, unofficial endpoint)."""
    api = api or YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=languages)
    return [Segment(s.start, s.text.replace("\n", " ")) for s in fetched if s.text.strip()]


def download_audio(video_id: str, out_dir: str) -> Path:
    import yt_dlp  # optional dependency, see requirements-audio.txt

    options = {"format": "bestaudio/best", "outtmpl": f"{out_dir}/%(id)s.%(ext)s", "quiet": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return Path(ydl.prepare_filename(info))


def transcribe_audio(path: Path, language: str | None = None, model=None) -> list[Segment]:
    """Local speech-to-text with faster-whisper (CPU int8 by default)."""
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(path), language=language, vad_filter=True)
    return [Segment(s.start, s.text.strip()) for s in segments if s.text.strip()]


def get_transcript(video_id: str, languages: tuple[str, ...] = ("en",), use_whisper: bool = False,
                   api=None, whisper_model=None) -> tuple[list[Segment], str]:
    """Returns (segments, source) where source is "captions" or "whisper"."""
    try:
        return fetch_captions(video_id, languages, api), "captions"
    except YouTubeTranscriptApiException:
        if not use_whisper:
            raise
    with tempfile.TemporaryDirectory() as tmp:
        audio = download_audio(video_id, tmp)
        return transcribe_audio(audio, languages[0], whisper_model), "whisper"
