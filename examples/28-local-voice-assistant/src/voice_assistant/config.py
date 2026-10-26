from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    llm_model: str = "llama3.2:3b"
    whisper_model: str = "small"
    whisper_language: str | None = "en"
    piper_voice: str = "voices/en_US-lessac-medium.onnx"
    reminders_file: str = "reminders.json"
    emergency_webhook_url: str = ""
    emergency_contact_name: str = "your emergency contact"
    local_emergency_number: str = "112"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        env = os.environ.get
        return cls(
            base_url=env("OPENAI_BASE_URL", d.base_url),
            api_key=env("OPENAI_API_KEY", d.api_key),
            llm_model=env("LLM_MODEL", d.llm_model),
            whisper_model=env("WHISPER_MODEL", d.whisper_model),
            whisper_language=env("WHISPER_LANGUAGE", d.whisper_language) or None,
            piper_voice=env("PIPER_VOICE", d.piper_voice),
            reminders_file=env("REMINDERS_FILE", d.reminders_file),
            emergency_webhook_url=env("EMERGENCY_WEBHOOK_URL", d.emergency_webhook_url),
            emergency_contact_name=env("EMERGENCY_CONTACT_NAME", d.emergency_contact_name),
            local_emergency_number=env("LOCAL_EMERGENCY_NUMBER", d.local_emergency_number),
        )
