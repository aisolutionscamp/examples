"""Run the assistant.

    python -m voice_assistant.main --text          # type instead of speaking (quick check of the LLM)
    python -m voice_assistant.main --wav in.wav    # one spoken turn: writes reply.wav
"""
from __future__ import annotations

import argparse
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from .assistant import Assistant
from .config import Settings
from .emergency import EmergencyEscalation
from .reminders import ReminderStore
from .speech import Speaker, Transcriber


def build(settings: Settings) -> Assistant:
    return Assistant(
        llm=OpenAI(base_url=settings.base_url, api_key=settings.api_key),
        model=settings.llm_model,
        reminders=ReminderStore(settings.reminders_file),
        escalation=EmergencyEscalation(settings.emergency_webhook_url, settings.emergency_contact_name,
                                       settings.local_emergency_number),
        transcriber=Transcriber(settings.whisper_model, settings.whisper_language),
        speaker=Speaker(settings.piper_voice),
    )


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true")
    p.add_argument("--wav")
    p.add_argument("--out", default="reply.wav")
    args = p.parse_args()
    assistant = build(Settings.from_env())

    if args.wav:
        heard, reply = assistant.handle_audio(args.wav, args.out, datetime.now())
        print(f"heard: {heard}\nreply: {reply}\nspoken reply written to {args.out}")
        return
    print("Type a message (empty line to quit).")
    while True:
        for line in assistant.reminder_announcements(datetime.now()):
            print(f"assistant: {line}")
        text = input("you: ").strip()
        if not text:
            break
        print(f"assistant: {assistant.handle_text(text, datetime.now())}")


if __name__ == "__main__":
    main()
