# Language practice bot

Role-play conversations in a foreign language (restaurant, train station, doctor, hotel, small talk) with
an LLM that stays in character, adapts to your CEFR level and corrects the mistakes in your last message.
Speech is optional: speak your message (faster-whisper) and hear the reply (Piper locally, or the OpenAI
Speech API).

Every turn the model answers in JSON:

```json
{"reply": "¡Claro! ¿Algo para beber?",
 "corrections": [{"original": "Yo quiero un mesa", "corrected": "Quiero una mesa",
                  "explanation": "\"mesa\" is feminine: una mesa."}],
 "translation": "Sure. Anything to drink?"}
```

The terminal shows the corrections and the reply; at the end you get all corrections of the session.
Only the in-character replies go back into the chat history, so the corrections do not distract the role
play. The history is trimmed to the last 12 turns.

## About pronunciation feedback

There is no pronunciation score. With speech input, faster-whisper reports a confidence for every word;
words below 0.5 are listed as "not clearly understood, try saying these again". Low confidence can come
from pronunciation, but also from noise, the microphone or a rare word. Real pronunciation assessment
needs a dedicated phoneme-level model or service.

## Architecture

```
mic recording ─► faster-whisper (word confidences) ─┐
typed text ─────────────────────────────────────────┴─► PracticeSession ─► chat model (JSON) ─► reply + corrections
                                                         (scenario, level, history)                │
                                                                                  Piper / OpenAI TTS ◄┘ (optional)
```

## Setup and run

```bash
pip install -r requirements.txt           # text only
pip install -r requirements-speech.txt    # plus speech input and Piper output
cp .env.example .env                       # then export the variables
cd src
python -m language_bot --scenario restaurant --language Spanish --level A2 --native English
python -m language_bot --scenario train --language French --language-code fr --audio-in ../me.wav --speak
```

Piper needs a voice model: `python -m piper.download_voices es_ES-davefx-medium --download-dir voices`
(list of voices: https://huggingface.co/rhasspy/piper-voices), then set `PIPER_VOICE` to the `.onnx`
file.

Tests (offline; chat client, Whisper model, Piper voice and Speech API are fakes): `pytest`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | OpenAI key (any string for Ollama) |
| `OPENAI_MODEL` | `gpt-6-luna` | chat model |
| `OPENAI_BASE_URL` | OpenAI | e.g. `http://localhost:11434/v1` for Ollama |
| `WHISPER_MODEL` | `small` | faster-whisper model size |
| `TTS_BACKEND` | `piper` | `piper` (local) or `openai` |
| `PIPER_VOICE` | none | path to a Piper `.onnx` voice |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | only for `TTS_BACKEND=openai` |

The OpenAI Speech API models (`gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`) are deprecated and shut down on
2027-01-06. OpenAI's replacement is the Realtime API, which needs a different integration; Piper is the
default here for that reason.

## Limitations

- The LLM can miss mistakes or "correct" correct sentences, especially at higher levels or with small
  local models. Treat corrections as suggestions.
- Small local models do not always return valid JSON; the reply is then shown without corrections.
- No progress tracking across sessions; `session.review()` only covers the current session.
- Piper voices differ in quality per language.

## For production

- Store corrections per learner and repeat frequent mistakes in later sessions (spaced repetition).
- Stream the reply and audio for a lower perceived delay; record audio in the browser or app.
- Use `response_format` with a JSON schema where the model supports structured outputs.
- Add content moderation if the bot is used by children or in open-ended chat.
