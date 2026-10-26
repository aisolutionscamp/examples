"""python -m yt_summarizer <url> [--lang en] [--whisper] [--title "..."]"""
import argparse

from .summarizer import Summarizer
from .transcript import get_transcript
from .video_id import extract_video_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yt_summarizer")
    parser.add_argument("url")
    parser.add_argument("--lang", action="append", help="caption languages in order of preference")
    parser.add_argument("--whisper", action="store_true", help="transcribe audio if there are no captions")
    parser.add_argument("--title", default="untitled")
    parser.add_argument("--summary-language", default="English")
    args = parser.parse_args(argv)

    video_id = extract_video_id(args.url)
    segments, source = get_transcript(video_id, tuple(args.lang or ["en"]), use_whisper=args.whisper)
    print(f"Transcript: {len(segments)} segments from {source}\n")
    print(Summarizer().summarize(segments, title=args.title, language=args.summary_language))


if __name__ == "__main__":
    main()
