"""Map-reduce summarization of long transcripts with an OpenAI-compatible chat model.

1. Chunk the timed transcript (each line keeps its [mm:ss] timestamp).
2. Map: summarize every chunk into notes with timestamps.
3. Reduce: merge the notes into one summary with key points and chapter timestamps.
Short transcripts that fit into one chunk skip the reduce step.
"""
import os

from .transcript import Segment

CHUNK_PROMPT = (
    "You get part {part} of {total} of a video transcript. Every line starts with a [mm:ss] timestamp. "
    "Write concise notes of the main points in this part, each note starting with the timestamp where "
    "it is discussed. Use only information from the transcript."
)
FINAL_PROMPT = (
    "Below are timestamped notes from a video titled \"{title}\". Write in {language}:\n"
    "1. A summary of at most {words} words.\n"
    "2. 5 to 7 key points as bullet points.\n"
    "3. A list of chapters as \"mm:ss - topic\", using only timestamps that appear in the notes.\n"
    "Do not add facts that are not in the notes."
)


def timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


def chunk_transcript(segments: list[Segment], max_chars: int = 12000) -> list[str]:
    """Group timestamped lines into chunks of at most max_chars (a single long line stays whole)."""
    chunks, current, size = [], [], 0
    for seg in segments:
        line = f"[{timestamp(seg.start)}] {seg.text}"
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def make_client():
    from openai import OpenAI

    # OPENAI_BASE_URL is read by the SDK too; passing it explicitly makes the Ollama option obvious
    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)


class Summarizer:
    def __init__(self, client=None, model: str | None = None, max_chars: int = 12000):
        self.client = client or make_client()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        self.max_chars = max_chars

    def _chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content.strip()

    def summarize(self, segments: list[Segment], title: str = "untitled", language: str = "English",
                  words: int = 200) -> str:
        if not segments:
            raise ValueError("empty transcript")
        chunks = chunk_transcript(segments, self.max_chars)
        if len(chunks) == 1:
            notes = chunks[0]  # fits in one request: summarize the transcript directly
        else:
            notes = "\n\n".join(
                self._chat(CHUNK_PROMPT.format(part=i, total=len(chunks)), chunk)
                for i, chunk in enumerate(chunks, start=1)
            )
        final = FINAL_PROMPT.format(title=title, language=language, words=words)
        return self._chat(final, notes)
