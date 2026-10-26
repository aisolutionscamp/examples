# Local voice assistant for elderly care: faster-whisper + Ollama + Piper

Use case: [Voice Assistant for Elderly Care](https://aisolutionscamp.io/projects/28)

A voice assistant that runs on a small computer at home. Speech recognition, the language
model and speech output all run locally; nothing is sent to a cloud service. It can:

- talk in short, simple sentences (local LLM through Ollama),
- create and announce medication and appointment reminders, and repeat them until the user says "done",
- send a message to a chosen emergency contact when the user says things like "I fell" or "help me".

> **Not an emergency service and not a medical device.** The escalation only sends a message
> to a person the user chose and tells the user to call the local emergency number. It can
> miss a situation (unusual wording, the user cannot speak, speech not recognised). Use it in
> addition to a certified personal emergency alarm, not instead of one. It gives no medical
> advice and does not monitor health.

## Architecture

```
microphone WAV -> faster-whisper (small, int8, CPU) -> transcript
  -> emergency phrase check (code, before the LLM) -> webhook to contact + "call 112"
  -> "done"? -> acknowledge today's reminders
  -> Ollama (OpenAI-compatible API, llama3.2:3b) with tools add_reminder / list_reminders
       -> reminder store (reminders.json)
  -> Piper TTS (slightly slower speech) -> reply WAV -> speaker

reminder check (every loop): announce due reminders, repeat unconfirmed ones after 30 min
```

- `src/voice_assistant/assistant.py` turn handling, LLM tool loop
- `src/voice_assistant/emergency.py` phrase check and webhook escalation
- `src/voice_assistant/reminders.py` reminder store and scheduling logic
- `src/voice_assistant/speech.py` faster-whisper and Piper wrappers
- `src/voice_assistant/main.py` command line (text mode or one WAV turn)

## Setup

```bash
# 1. Local LLM
ollama pull llama3.2:3b          # https://ollama.com/library/llama3.2

# 2. Python deps and a Piper voice
pip install -r requirements.txt
python -m piper.download_voices --download-dir voices en_US-lessac-medium

# 3. Configuration
cp .env.example .env             # set EMERGENCY_WEBHOOK_URL, EMERGENCY_CONTACT_NAME, LOCAL_EMERGENCY_NUMBER
```

The Whisper model (`small`, about 500 MB) is downloaded from Hugging Face on first use and
then cached.

## Run

```bash
cd src
python -m voice_assistant.main --text            # type messages, see reminders being announced
python -m voice_assistant.main --wav ../in.wav   # one spoken turn, writes reply.wav
```

Microphone capture and playback are left out on purpose to keep the example portable. On the
device, record each utterance to a WAV file (e.g. with `sounddevice` or `arecord` and a wake
word or push-to-talk button), call `Assistant.handle_audio()`, and play the reply WAV.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint (any compatible server works) |
| `OPENAI_API_KEY` | `ollama` | ignored by Ollama, required by the SDK |
| `LLM_MODEL` | `llama3.2:3b` | needs tool-calling support |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` or a local path |
| `WHISPER_LANGUAGE` | `en` | empty = auto-detect |
| `PIPER_VOICE` | `voices/en_US-lessac-medium.onnx` | Piper voice file (the `.onnx.json` must sit next to it) |
| `REMINDERS_FILE` | `reminders.json` | reminder storage |
| `EMERGENCY_WEBHOOK_URL` | empty | receives `{"event": "possible_emergency", "time", "transcript"}` |
| `EMERGENCY_CONTACT_NAME` | | name the assistant says out loud |
| `LOCAL_EMERGENCY_NUMBER` | `112` | e.g. `911` in the US |

## Tests

```bash
pytest     # or: scripts/test-example.sh examples/28-local-voice-assistant
```

No audio, models or network: the LLM is replaced by scripted `openai` `ChatCompletion`
objects, Whisper and Piper by fakes, the webhook by a fake `post`. The tests cover emergency
detection and escalation (including a failed webhook and no contact configured), the tool
loop and invalid tool arguments, reminder timing, repeats and acknowledgement, history
capping and the audio turn.

Checked manually (not part of the tests): a Piper sentence transcribed back correctly with
faster-whisper `small`, and `llama3.2:3b` in Ollama created a 20:00 daily reminder through
the tool call.

## Limitations

- The emergency check is a short English phrase list. It will miss other wordings and other
  languages and trigger on some harmless sentences ("what is the emergency number?"). Extend
  the list with the user and their family, and add a physical help button.
- Small local models sometimes answer instead of calling a tool, or misread times. The tool
  validates the time format but cannot know what the user meant; the assistant repeats the
  time back so the user can correct it.
- Reminders repeat once after 30 minutes and then stop; there is no escalation to the contact
  for missed medication.
- Single user, single device, no authentication on the reminder file.

## For production

- Use a proper wake word or a button, and an echo-cancelling microphone.
- Send emergency messages through a reliable channel with delivery confirmation (e.g. a
  self-hosted ntfy server with the contact's phone app, or SMS through a provider) and test it regularly.
- Let carers manage reminders through a small web UI instead of only by voice.
- Get consent from the user for any message sent to others, and document what is stored.
- Piper (piper1-gpl) is GPL-3.0 licensed; check that this fits your distribution model.
