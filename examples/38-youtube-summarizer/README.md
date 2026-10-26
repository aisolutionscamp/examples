# YouTube video summarizer

Summarizes a YouTube video from its transcript: a short summary, 5 to 7 key points and a chapter list with
timestamps.

1. **Video id** from any common URL form (`watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`).
2. **Transcript** from the video's captions with
   [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) (manual or auto-generated,
   in your preferred languages). If there are none and you pass `--whisper`, the audio is downloaded with
   yt-dlp and transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. **Chunking**: every transcript line keeps its `[mm:ss]` timestamp; lines are grouped into chunks of
   about 12,000 characters.
4. **Map-reduce summarization** with a chat model through the OpenAI Python SDK: notes per chunk, then
   one final summary. Short videos (one chunk) need a single request. Works with OpenAI or a local model
   via Ollama's OpenAI-compatible API.

## Architecture

```
URL ─► video id ─► captions (youtube-transcript-api) ──┐
                   └─ none + --whisper: yt-dlp audio ─► faster-whisper
                                                       ▼
                         timestamped lines ─► chunks ─► per-chunk notes (LLM) ─► final summary (LLM)
```

## Setup and run

```bash
pip install -r requirements.txt            # captions only
pip install -r requirements-audio.txt      # plus Whisper fallback (needs ffmpeg)
cp .env.example .env                        # then export the variables
cd src
python -m yt_summarizer "https://www.youtube.com/watch?v=VIDEO_ID" --title "Talk title"
python -m yt_summarizer https://youtu.be/VIDEO_ID --lang de --lang en --summary-language German
python -m yt_summarizer https://youtu.be/VIDEO_ID --whisper
```

Local and free with Ollama: `ollama pull llama3.2`, then set `OPENAI_BASE_URL=http://localhost:11434/v1`,
`OPENAI_API_KEY=ollama`, `OPENAI_MODEL=llama3.2`.

Tests (offline: caption API, Whisper model, yt-dlp and the LLM client are all fakes): `pytest`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | OpenAI key (any string for Ollama) |
| `OPENAI_MODEL` | `gpt-6-luna` | chat model |
| `OPENAI_BASE_URL` | OpenAI | OpenAI-compatible server, e.g. Ollama |
| `WHISPER_MODEL` | `small` | faster-whisper model size for the fallback |

`Summarizer(max_chars=...)` sets the chunk size; lower it for local models with small context windows.

## Limitations

- youtube-transcript-api uses an undocumented YouTube endpoint, not the official YouTube Data API (which
  only lets video owners download captions). It can break when YouTube changes, and YouTube often blocks
  requests from cloud server IPs (`RequestBlocked` / `IpBlocked`); the library supports proxies for that.
- Downloading audio with yt-dlp may conflict with YouTube's terms of service. Use it only for content you
  have the right to process.
- Summaries can omit or distort points. Timestamps come from the transcript, but the model chooses which
  ones to cite; check important details in the video.
- Auto-generated captions have no punctuation and contain recognition errors.
- Whisper on CPU is slow for long videos (roughly real time or slower with the `small` model).

## For production

- Cache transcripts and summaries by video id and model.
- Run the Whisper fallback as a background job with a GPU (`device="cuda"`).
- Add a quality check, for example ask the model to cite a timestamp for every key point and verify that
  each cited timestamp exists in the transcript.
- Respect rate limits and handle `RequestBlocked` with retries or a proxy.
