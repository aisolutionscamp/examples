"""Text chat in the terminal:  python -m language_bot --scenario restaurant --language Spanish --level A2

Speech: --audio-in learner.wav transcribes one recording as the first message; --speak writes every
reply to reply-N.wav.
"""
import argparse

from .scenarios import SCENARIOS
from .session import PracticeSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="language_bot")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="restaurant")
    parser.add_argument("--language", default="Spanish")
    parser.add_argument("--language-code", default="es", help="ISO 639-1 code for speech recognition")
    parser.add_argument("--level", default="A2", choices=["A1", "A2", "B1", "B2", "C1", "C2"])
    parser.add_argument("--native", default="English")
    parser.add_argument("--audio-in", help="WAV/MP3 file with the learner's first message")
    parser.add_argument("--speak", action="store_true", help="synthesize replies (TTS_BACKEND)")
    args = parser.parse_args(argv)

    session = PracticeSession(SCENARIOS[args.scenario], args.language, args.level, args.native)
    pending = None
    if args.audio_in:
        from .speech import transcribe

        heard = transcribe(args.audio_in, args.language_code)
        print(f"You said: {heard.text}")
        if heard.unclear_words:
            print(f"(not clearly understood: {', '.join(heard.unclear_words)} - try saying these again)")
        pending = heard.text

    print("Type your message, empty line to finish.")
    while True:
        text = pending or input("> ").strip()
        pending = None
        if not text:
            break
        turn = session.send(text)
        for c in turn.corrections:
            print(f"  correction: {c['original']} -> {c['corrected']} ({c['explanation']})")
        print(f"{turn.reply}\n  [{turn.translation}]")
        if args.speak:
            from .speech import speak

            print(f"  audio: {speak(turn.reply, f'reply-{len(session.turns)}.wav')}")

    print(f"\n{len(session.turns)} turns, {len(session.review())} corrections:")
    for c in session.review():
        print(f"- {c['original']} -> {c['corrected']}")


if __name__ == "__main__":
    main()
