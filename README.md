# Voice Agent

A voice assistant. You speak into the browser microphone; the server transcribes the audio (speech-to-text), asks a language model for a reply, and speaks it back (text-to-speech). The reply streams: text appears as the model writes it, and speech starts before the whole answer is synthesized.

It works **with no API key at all**: Whisper, Ollama and local voices on your CPU, private and offline. With a **free Gemini API key** it uses Gemini for transcription, answers and a natural voice in many languages (see [Natural, multilingual voice](#natural-multilingual-voice-with-a-free-gemini-key)). The key can be the server's (`GEMINI_API_KEY`) or one each user enters on the UI's System page, which is then used only for that user's questions ([Using your own key](#using-your-own-gemini-key-in-the-browser)). Local voices exist for English (Piper, or the more natural Kokoro) and Telugu, and are used automatically when the free Gemini voice quota runs out ([Local voices](#local-voices-english-and-telugu)).

I built it to practise the engineering around voice AI rather than the models themselves: streaming, bounded concurrency, timeouts, cancellation, backpressure, and measuring where the time goes. Everything runs on a CPU. The numbers in this README were measured on a modest 2-core laptop and are labelled as such.

## Status: what has and has not been verified

| Area | Status |
|---|---|
| Full voice loop on Windows 10, Python 3.13.14, CPU only (Whisper `base.en`, Ollama `llama3.2:1b`, Piper) | **Verified**: run end to end, and benchmarked (see [Measured results](#measured-results)) |
| Browser UIs: the React HUD at `/` and the classic page at `/classic/`, streaming (WebSocket) and upload (HTTP) modes | **Verified** in headless Chrome using a WAV file as a fake microphone (`scripts/browser_e2e.mjs`): both transports, a phone-sized viewport, reduced motion, and a denied microphone. Not tried with a physical microphone, speakers, Firefox or Safari |
| Automated tests | **Verified**: see [Tests](#tests) for the exact commands and last results |
| Gemini (replies, transcription, natural voice) and local Telugu/English voices | **Verified live** on 2026-10-27 with a free-tier key in the server's `.env`: full turns in the browser in English, Spanish, Hindi and Telugu, and the fallbacks to the local voices when the free voice quota ran out. A handful of turns, so timings are indicative |
| No key; a user's own key entered in the browser | **No key: verified** in the browser (all local, 2026-10-27). **Own key:** the form, the check with Google and the refusal of an invalid key were verified in the browser; a turn with a valid key entered this way was verified only in automated tests with Gemini stubbed out, because no valid key was available for the test |
| Docker | **Not verified.** `Dockerfile` and `docker-compose.yml` are written but were never built or run (no Docker on the dev machine) |
| Linux / macOS | **Not verified.** Commands are given but only Windows was used |
| GPU acceleration | **Not verified.** Documented only (the dev machine has no NVIDIA GPU) |
| Public deployment | **None exists.** There is no hosted demo and no URL |

## How it works

```
  Browser (web/)                                 Server (FastAPI + asyncio, app/)
 +---------------------+                     +--------------------------------------------------+
 | microphone          |   WebSocket         | VoiceSession   one per connection, owns history  |
 |  -> AudioWorklet    |  /ws/voice          |      | "stop"                                    |
 |  -> 16 kHz PCM16    |--- audio frames --->|      v                                           |
 |                     |                     | VoicePipeline  admission, timeouts, metrics      |
 | transcript, reply   |<-- JSON events -----|                                                  |
 | text, diagnostics   |                     |   STT slot        LLM slot         TTS slot      |
 |                     |<-- PCM16 per -------|  +---------+    +----------+  sentences  +-----+ |
 | Web Audio player    |    sentence         |  | Whisper |--->| Ollama / |--(bounded)->|Piper| |
 |  (gapless)          |                     |  | (thread)|    | Gemini   |  queue      |(thr)| |
 +---------------------+                     |  +---------+    +----------+             +-----+ |
        |                                    +--------------------------------------------------+
        |  POST /api/turn  (upload mode: the same stages run one after another)
        +------------------------------------------------------------------------------------>
```

A streaming turn on one connection:

```
 user speaks .......... stop
 server:                 [ STT ]---------->[ LLM tokens ..................................]
                                           sentence 1 | sentence 2 | sentence 3
                                                      [TTS 1]      [TTS 2]    [TTS 3]
 client hears:                                        >>> audio 1 ... audio 2 ... audio 3
                                    first audio ^   (the HTTP path hears nothing until all of it is done)
```

Key ideas, each explained in [docs/design-decisions.md](docs/design-decisions.md):

* **Stages are separated and measured.** Every turn records queue wait, STT time, LLM time-to-first-token and total, TTS time-to-first-chunk and total, and time to first audio, all with `time.perf_counter()`.
* **Bounded everything.** At most `MAX_INFLIGHT_TURNS` turns are admitted (extra ones get an immediate "busy"); each stage has its own concurrency limit; the queue between the LLM and the voice holds at most four sentences, so a slow voice makes the LLM wait instead of buffering without limit.
* **Every await has a deadline**, and the whole turn has one more.
* **Cancellation reaches the work.** A client disconnect or a "cancel" message stops the LLM stream; blocking STT/TTS threads check a flag between segments.

## Features that exist

* Browser UI with two pages. **Assistant**: the visual, the microphone, the language toggle, attached files and the conversation at full height. **System**: the Gemini key, the connection mode (stream or upload), model status, the last turn's pipeline timings and a diagnostics table. A pill in the top bar says where questions are answered (local models, cloud, or cloud with your key). Clear errors throughout (microphone permission, models loading, network, timeouts). Local models are named on the messages they handled and in the System panel; cloud (Gemini) models are shown only as "cloud".
* **The reply's text follows its voice.** A spoken reply appears in the transcript as it is heard, word by word, instead of all at once before the audio. The server sends each piece of audio with the text it speaks; the browser spreads that text's words evenly over the piece's playback (word timings are not known, so it is close, not exact). A reply with no voice is shown in full at once.
* **Ask about files.** Attach PDF, Word (.docx) or text files (the Attach button, or drop them on the page) and ask about them by voice. See [Asking about files](#asking-about-files).
* **Language toggle.** With `STT_LANGUAGE=auto` and `STT_LANGUAGES=te,en`, the UI offers Auto / తెలుగు / English. Picking one tells the recogniser which language you speak and the model to reply in it; Auto detects it per question. The choice is remembered in the browser.
* Two visuals, switchable: the **Core** (rings and a spectrum drawn from the real microphone and reply audio) and the **Graph** (the pipeline, its models, the languages spoken and the conversation as a 3D node graph; the camera travels to the stage that is working and data flows along the active links; drag to look around).
* WebSocket streaming: incremental transcript (optional partials), incremental LLM text, per-sentence audio, `done` event with metrics, `error` and `cancelled` events, per-connection conversation history.
* Plain HTTP turn (`POST /api/turn`): upload audio, get transcript, reply, a playable WAV (base64) and metrics. This path is *not* streaming and is documented as such.
* Providers behind small interfaces: faster-whisper or Gemini (STT), Ollama or Gemini (LLM), Piper, Kokoro or Gemini (TTS). Each stage uses Gemini when a key is available and the local model otherwise, and falls back to the local model when Gemini refuses (quota, a bad key, network).
* Bounded concurrency, admission control (HTTP 429 / `server_busy`), timeouts per stage, graceful shutdown, structured JSON logs with request ids, `/health`, `/ready`, `/api/metrics`, `/api/info`.
* Upload safety: size cap, decoded-duration cap applied *while decoding*, format validation.
* WebSocket origin check (blocks other websites from driving a local instance).
* Tools: CLI client, benchmark script, contention probe, prompt-length comparison, results summariser, browser end-to-end script.

## Technology choices

| Part | Choice | Why |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic, `asyncio` | Native WebSockets, easy streaming responses, validated configuration |
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), `base.en`, int8 | Runs on a CPU, MIT-licensed, accurate on clear speech. Whisper is not a streaming recognizer; see limitations |
| LLM | Ollama (`llama3.2:1b`) or the Gemini REST API | Ollama: free and offline. Gemini: faster on weak hardware and needs a key. Both stream, behind one interface |
| TTS | [Piper](https://github.com/OHF-Voice/piper1-gpl), voice `en_US-ljspeech-medium` | Fast local synthesis; the voice's training data is public domain (see Licensing) |
| Browser audio | AudioWorklet to 16 kHz PCM16, WebSocket binary frames, Web Audio playback | Raw PCM avoids codec latency and server-side decoding; MediaRecorder upload is kept as a fallback |
| Tests | pytest, pytest-asyncio, Node's built-in test runner | No extra test frameworks |

## Prerequisites

* Python **3.11 or newer** (developed and tested on 3.13.14). `ffmpeg` is **not** needed: PyAV bundles the libraries it uses.
* About 200 MB of disk for the speech models and about 1.3 GB for the LLM.
* [Ollama](https://ollama.com/download) for answers without a key (and as the fallback with one). A Gemini API key is optional.
* A browser with microphone access. Microphones work on `http://localhost` and `http://127.0.0.1`, but anywhere else the page must be served over `https://`.
* Optional: Node 20+ (for `node --test` and the browser end-to-end script) and Chrome or Edge.

## Setup

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv            # any Python 3.11+ should work; 3.13 is what was tested
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\download_models.py     # Whisper + Piper voice into .\models (about 200 MB)

cd frontend; npm ci; npm run build; cd ..   # the React UI (Node 20+). Skip it and the classic page is served instead

ollama pull llama3.2:1b               # local replies (after installing Ollama); not needed with Gemini

Copy-Item .env.example .env           # then edit .env (for example to use Gemini, below)
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

**Linux / macOS** (not run by the author)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_models.py
(cd frontend && npm ci && npm run build)

ollama pull llama3.2:1b

cp .env.example .env
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> (the classic page is at <http://127.0.0.1:8000/classic/>). The status pills show "Connected" and "Models ready" once loading has finished (the first start takes a while: models load and warm up). Press the microphone, speak, press it again to send.

If you skip `download_models.py`, the server downloads missing models on first start; `/health` reports `starting` until it is done.

### Local or cloud

Each stage (`STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER`) is `auto` by default: **Gemini when a Gemini key is available, the local model otherwise**. With no key anywhere the app runs fully locally; nothing needs changing. Set a stage to `whisper`, `ollama` or `local` to keep it on your machine even with a key (`gemini` means the same as `auto`).

* **Local (no key).** Answers need Ollama running and the model pulled; `GET /health` tells you if either is missing, with the exact command to fix it. Measured on 2026-10-27 on the 2-core laptop, one browser turn with no key: first audio 15.6 s after stop (Whisper 5.0 s, Llama first token 3.4 s, the Kokoro voice's first audio 6.8 s). A 1B model's answers are short and sometimes odd, and Whisper `base` and the local voices are weaker than Gemini outside English.
* **Gemini.** A key in `.env` (`GEMINI_API_KEY=...`) is used for everyone who has not entered their own; or each user enters theirs in the browser (below). Your speech, the transcript and the conversation are then sent to Google. When Gemini refuses before answering, the local model answers instead (`LLM_FALLBACK_TO_OLLAMA=true`) and the UI says so. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite` (fast; verified live). Model names change often, so check [the list](https://ai.google.dev/gemini-api/docs/models). Newer Gemini models "think" before answering; `GEMINI_THINKING_LEVEL=minimal` keeps replies quick (first token 1.07 s against 1.60 s without it, in one probe each).

### Using your own Gemini key in the browser

On the **System** page, paste a key (free from [Google AI Studio](https://aistudio.google.com/apikey)) and press Save. The server checks it with Google by listing one model, which uses no quota, and shows Google's answer; an invalid key is not saved. A working key is kept in that browser's local storage and sent to the server with that user's questions only (over the WebSocket, or the `X-Gemini-Key` header of `/api/turn`). The server keeps a set of Gemini adapters per key in memory, keyed by a hash of it, and never stores it on disk or logs it. Remove it with "Remove key".

So a public demo needs no key of its own: visitors bring theirs and use their own quota. If the server has a key in `.env`, visitors without one use it, so leave it empty on a public server unless you mean to share your quota. A key typed into a page served over plain `http://` crosses the network unencrypted: serve the app over `https://` anywhere but your own machine.

### Natural, multilingual voice with a free Gemini key

With a Gemini API key the assistant understands and answers in many languages and speaks with a natural voice. Enter the key in the browser (above), or put it in `.env` with the settings that suit it:

```
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=minimal
STT_MODEL=base                            # multilingual Whisper for the fallback (not base.en)
STT_LANGUAGE=auto                         # detect the language of each utterance
GEMINI_TTS_MODEL=gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts,gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts
GEMINI_TTS_VOICE=Kore                     # 30 voices, e.g. Puck, Charon, Aoede, Fenrir, Leda
```

Add the languages you speak, so transcription does not drift into a related language:

```
STT_LANGUAGES=te,en                       # e.g. Telugu and English
```

Measured on 2026-10-27: without this, Gemini wrote the Telugu question "భారతదేశ రాజధాని ఏమిటి?" as a Hindi *translation* (भारत देश की राजधानी क्या है?) in four of four attempts, whatever the instruction said; with it, the same question came back word for word in Telugu script (two of two attempts: one direct, one through the browser), while English and other Telugu sentences stayed correct. The local Whisper fallback uses the same list: it picks the likeliest of those languages instead of any of its ~100.

Then speak: the transcript is shown in its own script with a language tag and the model that heard it, the reply comes back in that language, and the voice detects the language from the text. Verified on 2026-10-27 in English, Spanish and Hindi (a few turns each): first audio arrived 4.4 s (English) and 5.1 s (Spanish) after pressing stop, measured in the browser.

**The free tier is small.** The API's own error messages on 2026-10-27 gave these limits for the voice model: **3 requests per minute and 10 per day**, per project, resetting at midnight Pacific time (check yours at <https://aistudio.google.com/rate-limit>). The text model's limits were not reached in testing. The code is built around that:

* one voice request per reply, not per sentence; the audio still streams back in pieces, so playback starts early;
* after a refusal the adapter waits for the time the API gives (it does not keep retrying);
* when the voice is unavailable, the reply is spoken by a local voice for its script (English and Telugu, see below), sentence by sentence; a reply in a script with no local voice is shown as text only. The UI says which happened and when the quota resets;
* if Gemini transcription is unavailable, local Whisper transcribes instead; partial transcripts are never requested from Gemini;
* several voice models can be listed in `GEMINI_TTS_MODEL`. Each has its own free quota, so when one is used up the next one speaks, and the local voices take over only when all have refused.

Which voice models a free key can use, checked on 2026-10-27 with one short request each (one sample, so the timings are indicative): `gemini-3.8-flash-lite-tts` (first audio 0.80 s), `gemini-3.8-flash-tts` (1.04 s), `gemini-3.1-flash-tts-preview` (1.85 s) and `gemini-2.5-flash-preview-tts` (4.17 s; it sent the audio in one piece) all answered; `gemini-2.5-pro-preview-tts` has a free limit of 0. Only the first model's daily limit (10) was observed; a forum report gives 10 for `gemini-3.1-flash-tts` too. If the others are similar, the list above gives about 40 natural-voice replies a day.

Why Gemini for transcription too: multilingual Whisper `base` recognised a Hindi question as Hindi but wrote it in Urdu script, with errors; Gemini wrote it correctly in Devanagari, in about the same time (2.7–3.4 s against 3.1–3.4 s on this CPU, three samples each).

### Local voices (English and Telugu)

These run on your CPU, cost nothing and need no key. Each sentence goes to the voice for its script, so a reply that mixes Telugu and English uses both.

```
TTS_PROVIDER=local                        # or gemini: then these are its fallback
TTS_LOCAL_ENGLISH=kokoro                  # kokoro (natural) or piper (fast, synthetic-sounding)
KOKORO_VOICE=af_heart                     # af_bella, am_michael, bf_emma, ...
TTS_TELUGU_VOICE=te_IN-padmavathi-medium  # or te_IN-venkatesh-medium (male)
```

`python scripts/download_models.py` fetches what is configured (Kokoro about 350 MB, each Piper voice about 64 MB). Measured on this laptop (Intel i5-6300U, 3 runs each): the Telugu voices synthesized 3.6–4.8 s of speech in 0.4–0.7 s; Kokoro needed about as long as the speech lasts (5.0–5.4 s for 4.9 s of speech, 2.1 s for a short first sentence), so on a 2-core CPU there can be a pause between sentences. Its int8 model was slower here, not faster (medians of 5.6 s and 6.7 s in two runs), so the full-precision one is used. In the browser, a Telugu reply spoken by the local voice started 4.2 s and 6.3 s after pressing stop (two turns).

No Telugu voice of Kokoro's quality exists locally that I could find: Kokoro has no Telugu, and the larger multilingual Indian models (AI4Bharat's Indic Parler-TTS, for example) are far too slow for a laptop CPU. The Piper Telugu voices sound clearer than the old English one but still synthetic; the Gemini voice is the natural option for Telugu.

### Asking about files

Attach a file with the **Attach file** button (or drop it on the page), then ask about it by voice. PDF (with a text layer), Word `.docx`, and text files such as `.txt`, `.md`, `.csv`, `.json` and `.html` are read; images, scanned PDFs and spreadsheets are refused with the reason.

* The text is extracted once, at upload, and kept **in memory only** (lost when the server restarts; the UI then asks you to attach the file again). The server keeps the 16 most recent files.
* For each question the relevant text goes into the language model's prompt: the whole file when it fits `DOCUMENT_CONTEXT_CHARS` (16,000 characters with Gemini, 3,000 with Ollama by default), otherwise the opening of each file plus the passages that share the most words with the question (BM25 ranking, no embedding model). Matching is by words, not meaning: a question in Telugu about an English file finds only the openings.
* The prompt marks the file as data and tells the model not to follow instructions written inside it. That reduces, but cannot rule out, a file steering the answer.
* With a local LLM on a small CPU, a long prompt is slow to process; that is why the Ollama budget is small.
* PDF text comes from `pypdf`. Text in Indic scripts can come out garbled from a PDF that does not map its glyphs back to letters (a known PDF problem; not tested here with a Telugu PDF); a `.docx` or `.txt` of the same document avoids it.

Measured on 2026-10-27 in the browser (Gemini for every stage): with a short brief attached, the spoken question "What is the launch date of Project Falcon?" got "Project Falcon launches on March fourteenth, twenty twenty-seven. The launch will take place at the Warangal test site.", both from the file; first audio 5.4 s after stop. The same question over the HTTP upload path got the same facts.

**Privacy:** with these settings your audio, transcript and replies are sent to Google, and so is the text of attached files that goes into a prompt. Google's Gemini API terms (read 2026-10-27) say content sent on the free (unpaid) tier may be used to improve Google's products and may be read by human reviewers, and ask you not to send sensitive or personal information; in the EEA, Switzerland and the UK the paid-service terms apply instead. Use the local setup for anything private.

### Other settings

Everything is an environment variable or a line in `.env`. [.env.example](.env.example) lists the main settings with their defaults and [app/config.py](app/config.py) is the complete list (including `LLM_SYSTEM_PROMPT`). The ones you are most likely to touch:

| Variable | Default | Effect |
|---|---|---|
| `STT_MODEL` | `base.en` | `tiny.en` is faster and less accurate; `small.en` the reverse |
| `STT_CPU_THREADS` | `0` | CTranslate2 threads; `0` = its default |
| `STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER` | `auto` | Gemini when a key is available, else local; `whisper` / `ollama` / `local` keep a stage local |
| `OLLAMA_MODEL`, `GEMINI_MODEL` | `llama3.2:1b`, `gemini-3.5-flash-lite` | |
| `LLM_FALLBACK_TO_OLLAMA` | `true` | The local model answers when Gemini refuses |
| `OLLAMA_NUM_THREAD` | unset | CPU threads Ollama may use; unset = its default. Fewer leaves cores for STT/TTS but slows generation (measured below) |
| `LLM_MAX_OUTPUT_TOKENS` | `160` | Hard cap on reply length |
| `TTS_VOICE` | `en_US-ljspeech-medium` | Any Piper voice name; check its licence first |
| `STT_CONCURRENCY`, `LLM_CONCURRENCY`, `TTS_CONCURRENCY` | `1` | See "Concurrency" in the design notes before raising these |
| `MAX_INFLIGHT_TURNS` | `4` | Turns admitted at once; beyond this, new turns get "busy" |
| `*_TIMEOUT_S` | 30 / 30 / 60 / 30 / 120 | STT, LLM first token, LLM total, TTS, whole turn |
| `PARTIAL_TRANSCRIPTS` | `false` | Live partial transcripts (costs CPU; see limitations) |
| `TTS_FIRST_CLAUSE_CHARS` | `0` | Let the first spoken chunk end at a comma to start speaking earlier |
| `WARMUP_ON_START` | `true` | `false` lets you measure true cold starts |
| `MAX_DOCUMENT_BYTES` | `10000000` | Largest file that can be attached |
| `DOCUMENT_CONTEXT_CHARS` | 16000 (Gemini) / 3000 (Ollama) | How much of the attached files' text goes into each prompt |

### GPU (untested here)

`STT_DEVICE=cuda STT_COMPUTE_TYPE=float16` runs Whisper on an NVIDIA GPU. It needs an NVIDIA driver plus cuBLAS for CUDA 12 and cuDNN 9, as described in the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu). Piper's `use_cuda` option is not wired up. The LLM runs wherever Ollama puts it. None of this was tested because the development machine has no NVIDIA GPU.

## Try it without a browser

```
python scripts/make_test_audio.py                        # once: creates benchmarks/inputs/*.wav with the local voice
python scripts/voice_client.py benchmarks/inputs/medium.wav                    # streaming
python scripts/voice_client.py benchmarks/inputs/medium.wav --transport http   # one request, nothing until the end
python scripts/voice_client.py question.wav --save-audio reply.wav             # any audio file; saves the spoken reply
```

It prints the transcript, reply, and when each arrived, plus the server's stage timings.

## HTTP and WebSocket API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Always 200 while the process is up; body has `status` (`ok`/`starting`/`degraded`) and per-component state with human-readable detail |
| `GET /ready` | 200 only when STT, TTS and the LLM are all usable, else 503 (for load balancers) |
| `POST /api/turn` | Multipart: `audio` (any format PyAV decodes: WAV, WebM/Opus, Ogg, MP3, MP4), optional `history` (JSON list of `{role, content}`), optional `language` (`auto` or one of the configured codes) and `documents` (JSON list of ids from `/api/documents`). Optional header `X-Gemini-Key`: use that key for this turn. Returns transcript, reply, base64 WAV and metrics. Errors use `{"error": {"code", "message", "request_id"}}` |
| `POST /api/gemini-key/check` | JSON `{"key": "..."}`. Returns `{"valid", "message"}` after asking Google; the key is not stored |
| `POST /api/documents` | Multipart `file`. Extracts its text and returns `{id, name, chars, pages, truncated}`; send the id with each question |
| `DELETE /api/documents/{id}` | Forget an attached file (204) |
| `GET /api/metrics` | Median/p95 per stage over the last 200 turns, outcome counts, current load |
| `GET /api/info` | Versions and non-secret settings (benchmarks save this with their results) |
| `WS /ws/voice` | Streaming turns. The message protocol is documented at the top of [app/voice_socket.py](app/voice_socket.py) |

Error codes include `unsupported_document` (415), `document_missing` (404, with the missing `ids`), `unsupported_audio` (415), `empty_audio` (422), `audio_too_long` / `upload_too_large` (413), `server_busy` (429), `models_not_ready` / `shutting_down` (503), `stt_error` / `llm_error` / `tts_error` (502), `stt_timeout` / `llm_timeout` / `tts_timeout` / `turn_timeout` (504), `cancelled`.

## Tests

```
python -m pytest -m "not integration"          # fast; fake models; no downloads needed
python -m pytest -m integration -v             # real Whisper/Piper/Ollama; skips what is unavailable
python -m pytest                               # both
node --test "web/test/*.test.mjs"              # classic page logic (resampler, WAV builder, messages)
cd frontend && npm test                        # React UI logic (state, protocol parsing, formatting)
node scripts/browser_e2e.mjs --transport stream    # real Chrome/Edge + fake microphone; needs the server running
node scripts/browser_e2e.mjs --transport upload
node scripts/browser_e2e.mjs --url http://127.0.0.1:8000/classic/   # the classic page
node scripts/browser_e2e.mjs --scenario deny-mic                    # microphone permission refused
node scripts/browser_e2e.mjs --audio benchmarks/inputs/es.wav --expect francia   # any language
python -m ruff check .                         # lint
```

What the suites cover: configuration validation, health/readiness states (loading, failed load, LLM down), request ids, upload validation (garbage, empty, too long, too large, malformed history), audio decoding including browser-style WebM/Opus, adapter contracts for the Ollama and Gemini wire formats (mocked HTTP), every stage timeout and the whole-turn timeout, provider errors, cancellation (task cancel, WebSocket close, HTTP disconnect), admission control and per-stage concurrency limits, the bounded sentence queue (checked by mutation), WebSocket protocol, origin checks, shutdown, and the benchmark script's statistics.

Test results are recorded under [Measured results](#test-results). One detail you may trip over: the shutdown test sends a real termination signal to a real server, and uvicorn re-raises that signal after a clean shutdown, so the expected exit code is 3 on Windows and `-SIGTERM` elsewhere (the test also checks the log line "Application shutdown complete").

## Benchmarks

```
python scripts/make_test_audio.py                # fixed prompts (once)

# warm steady state, both transports, 1/2/4 concurrent clients, 3 runs of 6 requests, 2 unmeasured warm-ups
python scripts/benchmark.py --spawn-server --transport both --concurrency 1,2,4 --requests 6 --runs 3 --warmup 2 --label baseline-warm

# cold start: a fresh server (warm-up off) and the LLM evicted from memory, so the first request really is the first
python scripts/benchmark.py --cold --transport ws --requests 6 --runs 3 --label baseline-cold

# try a configuration: any server setting via --env
python scripts/benchmark.py --spawn-server --env STT_MODEL=tiny.en --transport ws --requests 6 --runs 3 --label stt-tiny

# how much do the stages slow each other down?
python scripts/contention_probe.py

python scripts/summarize_results.py benchmarks/results/*-baseline-warm.json    # markdown tables

# compare settings fairly: one short window per setting, repeated in rounds, each setting
# compared with the baseline of the SAME round (see "Why one run per setting is not enough")
python scripts/benchmark.py --label ab-r1-baseline --spawn-server --transport ws --requests 6 --runs 1
python scripts/benchmark.py --label ab-r1-clause40 --spawn-server --env TTS_FIRST_CLAUSE_CHARS=40 --transport ws --requests 6 --runs 1
# ... repeat in a different order for rounds 2 and 3, then:
python scripts/compare_rotation.py ab
```

Method: configurations are interleaved (run 1 of everything, then run 2, ...) so drift hits all of them equally; each result file records the machine, server versions and settings, SHA-256 of every input file, background CPU load before and after, and **every raw request** (JSON and CSV). Latency is measured by the client from the end of user input. "First audio" is when the first synthesized audio reaches the client. Warm and cold results are never mixed. Inputs are spoken by the local TTS voice, so the transcription error rate shown is optimistic compared with real microphones.

## Measured results

All numbers below come from one machine on 2026-10-26: **Intel Core i5-6300U (2 cores / 4 threads), 15.9 GB RAM, Windows 10, no GPU used**, Python 3.13.14, Whisper `base.en` (int8), `llama3.2:1b` (Q8_0) through Ollama, Piper `en_US-ljspeech-medium`. Other applications were running: CPU use with the server idle was between 5% and 65% before individual runs (each result file records it), so read the absolute values as indicative. The three test questions are spoken by the local TTS voice (1.9 s, 4.0 s and 6.9 s long) and are sent as fast as the server accepts them unless a row says "realtime". Latency is measured by the client from the end of the user's audio. "Median [min–max]" is the median of per-run medians and the range across runs. Every figure can be found in `benchmarks/results/` (JSON with every raw request, CSV, console output and server log).

### How big is the noise? (read this before the other tables)

The default configuration, streaming, one client, was measured several times:

| Measurement | First audio, median (s) |
|---|---|
| `baseline-warm` (09:02) | 7.31 [4.36–8.25] |
| `control-ws` (09:41, identical settings) | 5.33 [4.98–5.45] |
| Rotation `ab`, baseline, round 1 / 2 / 3 | 9.92 / 5.59 / 5.02 (round 1 ran with 64.6% background CPU) |

The same settings gave medians from 5.0 s to 9.9 s depending on when they ran. **A difference of about two seconds between two separately measured configurations is therefore not evidence of anything on this machine.** The settings comparison further down avoids that by comparing each setting with the baseline *of the same round*.

First audio also depends a lot on the question, because the reply length differs (`control-ws`, 6 requests per question):

| Question | Reply (median words) | First audio (s) | Turn done (s) |
|---|---|---|---|
| short (1.9 s of audio) | 6 | 2.54 | 2.55 |
| medium (4.0 s) | 47 | 6.18 | 7.16 |
| long (6.9 s) | 53.5 | 5.36 | 8.74 |

### Where the time goes (default settings, warm, one client, streaming)

Server-side stage medians in ms:

| Measurement | STT | LLM first token | LLM total | TTS first chunk | TTS total |
|---|---|---|---|---|---|
| `baseline-warm` | 2082 | 812 | 4577 | 2033 | 3361 |
| `control-ws` | 1569 | 407 | 4218 | 1288 | 2582 |

Before the first sound the user waits for STT, then for the model to write a first sentence, then for that sentence to be synthesized. On this CPU the three stages are of similar size, so no single stage is "the" bottleneck. Piper's first chunk takes 1.3–2.0 s while the LLM is generating, against 0.46–0.66 s when it runs alone (see the contention probe).

### Cold start vs warm

One cold sample (`baseline-cold`: fresh server with warm-up off and the model evicted from Ollama, first request, the short question): first audio **9.58 s**, of which LLM first token 6.37 s (loading the model), STT 2.19 s and TTS first chunk 0.49 s. It is a single sample (n = 1). The same question warm took 2.54 s to first audio (`control-ws`, 6 requests). With `WARMUP_ON_START=true` (the default) the server pays this cost at startup instead of on the first user turn.

### Concurrency

`baseline-warm`, 3 runs of 6 requests per level, all stages limited to one at a time:

| Config | First audio (s) | Turn done (s) | Throughput (req/s) | STT queue wait (ms) |
|---|---|---|---|---|
| streaming, 1 client | 7.31 [4.36–8.25] | 8.39 | 0.12 | 0 |
| streaming, 2 clients | 12.55 [9.37–13.67] | 14.02 | 0.14 | 765 |
| streaming, 4 clients | 20.06 [13.34–20.37] | 24.15 | 0.13 | 6049 |
| HTTP, 1 client | 8.23 [7.53–8.84] | 8.23 | 0.14 | 0 |
| HTTP, 2 clients | 11.35 [11.28–14.98] | 11.35 | 0.17 | 513 |
| HTTP, 4 clients | 18.49 [18.03–20.61] | 18.49 | 0.16 | 4524 |

Throughput stays at roughly 0.12–0.17 turns per second while latency grows about linearly with the number of clients: on two cores extra clients wait in line; the machine does not do more work. That is why every stage defaults to one at a time and why `MAX_INFLIGHT_TURNS` refuses extra turns instead of queueing them without bound. This says nothing about machines with more cores. Retune with `scripts/benchmark.py` on the target host.

### Streaming vs the plain HTTP path

* **Short replies (default prompt):** streaming 7.31 [4.36–8.25] s vs HTTP 8.23 [7.53–8.84] s to first audio at one client. The ranges overlap, so no win is claimed: a two-sentence reply leaves little to overlap.
* **Long replies:** with the older, looser system prompt (`verbose-ws`, `verbose-http`; replies of 64 and 108 words for the medium and long questions, 3 requests per question per run) streaming first audio was 4.57 [3.53–7.77] s, while the HTTP path returned nothing until 11.09 [10.87–11.46] s. For the long question alone the medians were 4.57 s (streaming) vs 29.72 s (HTTP, 122-word reply), with only 3 samples each. These two were run one after the other, not interleaved, so treat the size of the gap as indicative. The direction follows from the design: HTTP waits for the whole reply.

### Do the stages slow each other down? (`contention_probe.py`, 5 repetitions)

| Operation (unit) | While this runs | Alone | Contended | Slowdown |
|---|---|---|---|---|
| TTS, one sentence (s) | STT | 0.46 | 0.69 | 1.49× |
| TTS, one sentence (s) | LLM generating | 0.46 | 1.43 | 3.08× |
| STT, one utterance (s) | TTS | 1.79 | 2.06 | 1.15× |
| STT, one utterance (s) | LLM generating | 1.79 | 8.73 (range 5.49–15.14) | 4.89× |
| LLM generation (tokens/s, higher is better) | TTS | 13.46 | 11.39 | 1.18× |
| LLM generation (tokens/s, higher is better) | STT | 13.46 | 12.37 | 1.09× |

Medians of 5 repetitions. The LLM loses only 9–18% of its speed when the others run, while STT and TTS become several times slower when the LLM is generating. The STT-while-LLM figure is very spread out (5.5 to 15.1 s across the five repetitions), so read it as "several times slower", not as 4.89×. That is exactly the situation a streaming turn creates (TTS of sentence 1 while the LLM writes sentence 2), which is why overlapping stages helps less on a two-core CPU than the diagram suggests.

### Speech-to-text

* Whisper's time barely depends on utterance length (`baseline-warm` plus `control-ws`, 12 requests per question): `base.en` 1585 ms for 1.9 s of audio, 1670 ms for 4.0 s, 1790 ms for 6.9 s. `tiny.en` (`stt-tiny`, 6 requests per question): 760, 762 and 900 ms. This matches Whisper's fixed 30-second window.
* The transcription error rate was **0.0% in every run of every configuration**. That only shows that clean, synthetic, same-voice audio is transcribed correctly; it says nothing about real microphones, accents or noise, and nothing about how much accuracy `tiny.en` gives up there. Not measured.

### Comparing settings fairly (rotation `ab`: 3 rounds, 6 variants, order rotated each round)

Each window starts a fresh server, runs 2 unmeasured warm-ups and 6 measured requests (two per question, streaming, one client). Round 1 ran with a disturbed baseline (64.6% background CPU) and therefore overstates every improvement; compare the rounds 2 and 3 deltas for a fairer view. "Reading" is mechanical: a change counts only if its paired difference has the same sign in all three rounds. Three rounds is a modest bar, not statistical proof.

First audio (s), each cell a window median; Δ = variant minus the same round's baseline:

| Variant | Round 1 | Round 2 | Round 3 | Δ per round | Reading |
|---|---|---|---|---|---|
| baseline | 9.92 | 5.59 | 5.02 | – | reference |
| `TTS_FIRST_CLAUSE_CHARS=40` | 6.38 | 3.66 | 4.26 | −3.53, −1.93, −0.76 | lower in every round |
| `STT_CPU_THREADS=2` | 6.72 | 4.28 | 6.11 | −3.19, −1.31, +1.09 | inconclusive |
| `OLLAMA_NUM_THREAD=1` | 7.99 | 5.78 | 6.80 | −1.93, +0.20, +1.78 | inconclusive |
| `STT_MODEL=tiny.en` | 4.66 | 3.99 | 4.71 | −5.26, −1.60, −0.31 | lower in every round |
| tiny.en + 2 STT threads + clause 40 ("tuned") | 2.49 | 3.15 | 2.57 | −7.43, −2.44, −2.45 | lower in every round |

What the other stages did, same method:

* **`tuned`** had the lowest first audio of all six variants in all three rounds. Its whole-turn time was **not** consistently better (Δ −6.26, −1.82, +0.10 s), because the LLM and TTS tail dominates the end of a turn.
* **`tiny.en`**: STT time lower in every round (816–829 ms against 1.7–3.3 s for the baseline); turn-done time also lower in every round (−5.66, −1.64, −1.37 s).
* **`TTS_FIRST_CLAUSE_CHARS=40`** shortened Piper's first chunk in every round (−2160, −1423, −427 ms) because the first chunk is shorter; turn-done time was not consistently better.
* **`STT_CPU_THREADS=2`** reduced STT time in every round (−1809, −583, −218 ms) but the first-audio effect is inconclusive.
* **`OLLAMA_NUM_THREAD=1`** made LLM generation slower in every round (+1671, +2174, +3900 ms; median of rounds 6.66 s against 4.42 s). It did free the cores for Piper (first chunk lower in every round) but the net effect on first audio is inconclusive, so it is not recommended.

**These are not defaults.** `tiny.en` trades accuracy that this benchmark cannot see (see above), and a first chunk that ends at a comma can sound less natural; I could not judge prosody by ear. Both are one-line opt-ins. A low-latency profile that was measured above:

```
STT_MODEL=tiny.en
STT_CPU_THREADS=2
TTS_FIRST_CLAUSE_CHARS=40
```

### Partial transcripts (realtime pacing)

With the audio paced at real time, `PARTIAL_TRANSCRIPTS=false` gave 5.92 [5.00–7.08] s to first audio and a 1691 ms STT stage; `true` gave 6.16 [5.70–6.81] s and 2045 ms. The STT stage was about 0.35 s longer with partials, while the end-to-end difference is inside the noise described above. The two were measured one after the other.

### What these numbers do not show

One machine, one LLM, short replies; synthetic speech; local loopback (no network latency, no jitter, no TLS); 3 runs of 3–6 requests (the rotation: 3 windows of 6 requests per variant); sequential rather than interleaved for the streaming-vs-HTTP and partial-transcript comparisons; a noisy background load. They show where the time goes and which direction changes move it on this class of hardware, not what a server or a GPU would deliver.

### Test results

Last full run (2026-10-27, Windows 10, Python 3.13.14, Node 24). It was made before the no-key mode and the per-user Gemini key were added; the note under the table says what was re-run since:

| Command | Result |
|---|---|
| `python -m pytest` with `GEMINI_API_KEY` set (unit, WebSocket, shutdown and integration tests; the integration tests used the real Whisper, Piper (English and Telugu), Kokoro, Ollama and Gemini) | **271 passed** in 150 s, none skipped (the live Gemini voice test skips itself when the free daily voice quota is used up; it was not) |
| `node --test "web/test/*.test.mjs"` | 9 passed |
| `cd frontend && npm test` | 44 passed (re-run after the later changes) |
| `python -m ruff check .` | All checks passed |
| `node scripts/browser_e2e.mjs` against the React UI and the classic page, both transports, phone viewport, reduced motion, denied microphone (local models, 2026-10-26) | All passed |
| The same script with Gemini, in English, Spanish, Hindi and Telugu (2026-10-27) | English and Spanish passed all checks with the natural voice. Hindi: transcript and reply correct, text only (the voice's daily quota was used up by then). Telugu, after the fixes: transcript and reply in Telugu, spoken by the local Telugu voice; English spoken by Kokoro. Their only failed check is the deliberate notice that the local voice stood in |
| The same script after adding files, the language toggle and text that follows the voice (2026-10-27, Gemini for every stage) | All checks passed in three runs: a question about an attached text file (streaming and HTTP upload), and the Telugu question with తెలుగు selected. While the voice spoke, the reply on screen grew with it (sampled 0 → 7 → 15 of 27 characters for Telugu; 0 → 81 of 87 over HTTP) |

Since that run, after the no-key mode, the per-user key and the two-page UI were added: the unit and WebSocket tests (everything except the integration tests) were run once, with 273 passed and 1 failed (a test that expected the old `/api/info` output; it was updated, because with a server key the default model is now Gemini with the local one as its fallback, and the file then passed with 11 tests). The whole suite including the integration tests has **not** been re-run in one go since those changes. Browser end-to-end runs on 2026-10-27 with no key (local models, English; the same run on a 390x844 phone viewport) and with an invalid key typed into the System page (refused by Google's check, the turn still ran locally, and the key did not appear in the server log) passed.

Not covered by any test: a physical microphone and speakers, Firefox and Safari, Docker, Linux and macOS.

## Known limitations

* **Whisper is not a streaming recognizer.** The transcript appears only after you stop speaking. The optional partial transcripts re-run Whisper on everything recorded so far, which costs a full inference each time and, on a small CPU, can delay the final transcript (measured above: the STT stage was about 0.35 s longer with partials on, while the end-to-end difference was inside the noise).
* **No automatic end-of-speech detection.** You press stop. (Whisper's built-in voice-activity filter only trims silence.)
* **CPU contention.** STT, LLM and TTS share the same cores. See the measurements above and the design notes; adding concurrency does not add speed on a 2-core machine.
* **A 1B model gives short, sometimes odd answers**, and ignores soft length instructions (the default prompt is deliberately blunt about this).
* **Languages.** The local-only setup (Whisper `base.en`, Llama 3.2 1B, Piper) is English. Other languages need the multilingual settings above; local voices exist only for English and Telugu.
* **No echo cancellation beyond what the browser provides**; use headphones if the assistant hears itself.
* **No authentication or per-user limits.** Run it on localhost, or put it behind something that authenticates (see the deployment guide). Limits such as `MAX_INFLIGHT_TURNS` are global.
* **Barge-in** (interrupting the assistant by speaking) is not implemented; there is a Stop button.
* **Conversation history is kept in memory per WebSocket connection** and lost on reload.
* **HTTP uploads are parsed before admission control runs.** A body whose `Content-Length` exceeds the limit is refused (413) before any of it is read, but a chunked upload that declares no length is only stopped once it has been received, costing memory up to the proxy's limit. For a public deployment, set a body-size limit on the reverse proxy.
* **A thread cannot be killed.** Cancelling a turn stops it at the next Whisper segment or Piper sentence boundary; the segment already running finishes first.

### Ideas for next steps

Smaller or quantised LLMs and a smaller Whisper to cut the biggest stages; streaming recognition (for example a chunked or transducer model); voice-activity-based end-of-speech and barge-in; a GPU path for Whisper and Piper; per-client rate limiting and authentication; CI; an Opus transport to cut upload size; speculative first-clause synthesis tuned by measurement.

## Security and privacy

* **Audio is processed in memory and never written to disk or logged.** Logs hold request ids, timings and error codes, not transcripts or replies (every log statement was reviewed).
* With **Ollama**, nothing leaves your machine after the models are downloaded. With **Gemini**, your transcript and conversation are sent to Google's API, and (unless `STT_PROVIDER=whisper`) your recorded audio too, plus the parts of attached files that go into a prompt. On the free tier Google may use that content to improve its products (see [Natural, multilingual voice](#natural-multilingual-voice-with-a-free-gemini-key)).
* **Secrets** live in environment variables or an untracked `.env` (`GEMINI_API_KEY` is read as a secret value, never logged or returned by `/api/info`, and is sent as a header, not in a URL). A key a user enters in the browser is kept in that browser's local storage, sent to this server only with that user's requests, held in server memory while in use, and never logged; the browser end-to-end test checked that a key it typed in did not appear in the server log. Any script running on the page could read local storage; the page loads no third-party scripts and renders model output as text, never HTML.
* The server has **no authentication**. It listens on `127.0.0.1` by default; compose publishes it on loopback only. Do not expose it publicly as is.
* The WebSocket rejects browser connections from other origins, so a website you visit cannot drive an instance on your machine. A cross-site plain `POST /api/turn` is not blocked (the response is unreadable to that site, but it could still make your server do work).
* Uploads are size-limited, decoded with a duration cap, and validated; model output is rendered with `textContent` (never as HTML).
* `/api/info` and `/api/metrics` are unauthenticated and reveal versions and timings.

## Licensing

This project's own code is licensed under the **GNU General Public License, version 3 or (at your option) any later version** (see [LICENSE](LICENSE), the unmodified text from <https://www.gnu.org/licenses/gpl-3.0.txt>). It was chosen because the code imports `piper-tts`, which is GPL-3.0-or-later, so GPL-3.0 is the simplest licence that is consistent with it; that is a plain reading of the dependencies, not legal advice. The GPL covers this code, not the models you download: those keep the licences below.

Third-party pieces, read from each project's own metadata or model card:

| Component | Licence |
|---|---|
| `pypdf` (reads attached PDFs) | BSD-3-Clause |
| `piper-tts` (Python package) | **GPL-3.0-or-later**. Distributing this project, or an image containing it, brings GPL obligations for that component |
| Piper Telugu voices `te_IN-padmavathi-medium`, `te_IN-venkatesh-medium` | Model cards: trained on AI4Bharat's IndicVoices-R, **CC-BY-4.0** (credit AI4Bharat when you use them). The third Telugu voice, `maya`, uses IIT Madras's own IndicTTS licence and is not used here |
| Kokoro-82M (`kokoro-v1.0.onnx`, voices) | Apache-2.0 (model card). `kokoro-onnx` is MIT; it uses `phonemizer` (GPL-3.0) and espeak-ng through `espeakng-loader` |
| Piper voice `en_US-ljspeech-medium` | Model card: LJ Speech dataset, "public domain". The `en_US-lessac-medium` voice (used in many Piper examples) was **not** chosen because its training-data licence is research-only |
| faster-whisper, CTranslate2 | MIT |
| Whisper `base.en` (Systran CTranslate2 conversion) | MIT (model card) |
| Llama 3.2 (`llama3.2:1b` via Ollama) | Llama 3.2 Community License: redistributing the weights requires the licence text and "Built with Llama", and there is an Acceptable Use Policy. This project does not redistribute them; do not bake them into a public image without reading the licence (`ollama show llama3.2:1b --license`) |
| PyAV | BSD-3-Clause for PyAV; its wheels bundle FFmpeg libraries with their own terms |
| FastAPI, Uvicorn, websockets, httpx, NumPy, pydantic-settings and the rest | MIT / BSD / Apache-2.0 per their package metadata |
| Gemini API | Subject to Google's API terms |

## Docker

```
docker compose up --build        # not run by the author: see docs/deployment.md
```

The `Dockerfile` and `docker-compose.yml` follow the setup above (non-root user, models in a volume, port bound to loopback, Ollama on the host or as an optional compose service). They have never been built here, so expect to fix something the first time.

## Deployment

**There is no public deployment.** [docs/deployment.md](docs/deployment.md) explains what hosting this needs (HTTPS for the microphone, WebSocket proxying, authentication, CPU sizing), with vendor prices and limits read from their pages on 2026-10-26.

## Project layout

```
app/
  main.py              FastAPI app: routes, middleware, lifespan, shutdown
  voice_socket.py      WebSocket session and protocol
  config.py            validated settings (env / .env)
  errors.py            error types with stable codes
  schemas.py           response models
  readiness.py         component state for /health and /ready
  startup.py           adapter construction and background model loading
  webui.py             which UI is served at / and /classic/
  logging_setup.py     JSON logs and request ids
  services/
    voice_pipeline.py  turn orchestration: admission, slots, timeouts, streaming
    stt.py llm.py tts.py   adapters: Whisper/Gemini, Ollama/Gemini, Piper/Gemini, and fallbacks
    speech_text.py     markdown cleanup and sentence chunking
    audio.py           decoding with a length cap, WAV helpers
    metrics.py         timers, per-turn metrics, rolling summary
frontend/              React HUD (Vite, TypeScript, Motion) and its Node tests
web/                   classic browser client (no build step) and its Node tests
scripts/               download_models, make_test_audio, voice_client, benchmark,
                       contention_probe, compare_prompts, summarize_results,
                       compare_rotation, browser_e2e
tests/                 unit, integration, WebSocket, shutdown, benchmark tests
benchmarks/results/    raw benchmark output (JSON, CSV, console logs)
docs/                  design-decisions, deployment, github-checklist
LICENSE                GNU GPL v3 (see Licensing)
```
