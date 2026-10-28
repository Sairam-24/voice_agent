# Design decisions

Why the code looks the way it does, what each choice costs, what the measurements say, and what I would do next. Where a statement rests on a measurement, the file in `benchmarks/results/` is named; where it rests on reasoning only, it says so.

## 1. Shape of the problem

A voice turn is three heavy stages in a row: speech-to-text (STT), a language model (LLM), text-to-speech (TTS). The delay a person feels is *time to first sound after they stop talking*, so the design question is how early the first audio can start, and what keeps the system predictable when it is slow, overloaded or the user walks away.

This was built and measured on one machine: an Intel i5-6300U (2 cores, 4 threads), 16 GB RAM, no usable GPU, Windows 10, with other applications running. That is a deliberately weak target. It makes bottlenecks obvious, and it means the absolute numbers are not what you would see on a server or a GPU.

## 2. Decisions and trade-offs

### Small adapters behind `Protocol`s, with fakes only in tests

`SpeechToText`, `LanguageModel` and `TextToSpeech` are each a few methods. The pipeline never imports a model library; tests pass fakes (`tests/fakes.py`) that can delay, fail, and count how many calls overlapped. There is no "mock mode" in the running app, so nothing a user can start could be mistaken for real inference.
*Cost:* a little indirection. *Benefit:* every failure mode (timeouts, provider errors, cancellation) is testable deterministically in milliseconds.

### Blocking inference runs in worker threads; the pool size equals the concurrency limit

faster-whisper and Piper are blocking calls. Each adapter owns a `ThreadPoolExecutor(max_workers=N)` with the same `N` as its semaphore. If a cancelled coroutine gives up its slot while its thread is still finishing, the next job waits in the executor queue, so more than `N` inferences can never run at once.

### Per-stage semaphores, so queueing is visible

The pipeline wraps each stage in its own semaphore and records `*_queue_ms` separately from the stage's work time. One global lock would hide the difference between "the model is slow" and "the box is busy".

### Admission control instead of a queue

`MAX_INFLIGHT_TURNS` (default 4) bounds running plus waiting turns. The next one is refused immediately (`429 server_busy` / `server_busy` event). A queue would turn overload into latency for everybody and unbounded memory; refusing is honest and cheap. The benchmark counts rejections as failures instead of hiding them.
*Limit:* the cap is global, not per client, and HTTP uploads are parsed before admission is checked. A small middleware refuses a body whose `Content-Length` is over the limit before reading it; a chunked upload with no declared length is not caught there, so a public deployment needs a body-size limit on the reverse proxy.

### Deadlines per `await`, never around a `yield`

Every stage has a timeout and the turn has one more. The LLM stage is an async generator, and a `timeout()` scope that stays open across a `yield` can fire while the consumer is suspended elsewhere and cancel the wrong task. So each `anext()` is awaited under its own deadline computed from a start time, and the generator never holds a timeout scope open between yields. There is a separate first-token deadline because model loading makes the first token much slower than the rest.

### Streaming at sentence granularity, through a bounded queue

LLM tokens go into a sentence chunker; each sentence is synthesized and sent while the LLM keeps generating. A `TaskGroup` runs the generate and speak halves, so an error or a disconnect on either side cancels the other. The queue between them holds four sentences: if the voice falls behind, the LLM blocks instead of piling up text. A unit test replaces the bound with "unlimited" and checks the test fails (mutation check), because a bound nobody tests is a bound that quietly disappears.
*Why sentences:* a TTS engine needs enough text for natural intonation, and Piper itself works sentence by sentence. Cutting finer would sound choppy.
*Option, off by default:* `TTS_FIRST_CLAUSE_CHARS` lets only the *first* chunk end at a comma, because a long opening sentence delays the first sound by the time the LLM takes to write it. Measured effect: see section 4.

### The HTTP path is kept, and honest about being sequential

`POST /api/turn` runs STT, then the whole LLM reply, then TTS, and returns everything at once. It is the simplest reliable path, a fallback when AudioWorklet or WebSocket is unavailable, and the baseline that shows what streaming buys. Its response says what it is; the diagnostics label "First audio ready" is the time the audio existed on the server, which for HTTP is also when the client gets it.

### WebSocket with raw PCM rather than WebRTC

The browser downsamples the microphone to 16 kHz PCM16 in an AudioWorklet and sends 100 ms binary frames. That needs no audio codec on the server, and the protocol is a dozen readable JSON messages.
*Not as good as WebRTC at:* jitter buffering and loss concealment on bad networks, Opus's bandwidth saving (PCM16 at 16 kHz is 256 kbit/s, fine on a LAN or localhost), and NAT traversal. WebRTC would be the next step for a deployed service; it adds a signalling layer and a media server dependency that the goals did not need.

### STT: Whisper, with its limits stated

faster-whisper gives good accuracy on a CPU. It is an offline model: it transcribes a window, not a stream. Consequences, each visible in the code:

* The final transcript can only start after the user stops. There is no automatic end-of-speech detection; the user presses stop.
* Whisper's encoder always processes a 30-second window, so a short utterance costs nearly as much as a long one (confirmed in the measurements below).
* "Partial transcripts" are implemented the only way Whisper allows: periodically re-decoding everything recorded so far. It is opt-in, never queues behind or in front of a real turn (if the STT slot is busy it skips), and its cost is measured rather than assumed.

### Cancellation: what it can and cannot stop

Cancelling a turn cancels the LLM HTTP stream immediately and stops TTS between sentences. Worker threads receive a `threading.Event` and check it between Whisper segments and between Piper sentences. A single Whisper window or a single Piper sentence that is already executing cannot be interrupted; it finishes, and its result is discarded. Tests cover task cancellation, WebSocket close and HTTP client disconnect (the last needs a real server because test clients do not send a real disconnect).

### Observability

Every turn produces one `TurnMetrics` record: queue wait, STT time, LLM first-token and total, TTS first-chunk and total, time to first audio, total, and the outcome. Timers are `time.perf_counter()`: monotonic and high resolution, unaffected by clock adjustments. Records feed a bounded rolling window (200 turns) behind `/api/metrics`, the UI's diagnostics table and the JSON logs, which carry a request id from a `ContextVar`. Nothing in the diagnostics UI is generated: a missing value is shown as a dash.

### Security choices

No secrets in code or logs; `SecretStr` for the API key; key sent as a header. The WebSocket checks `Origin`. Uploads are capped in bytes (checked from `Content-Length` before the body is read, and again after) and, more importantly, in *decoded* duration while decoding, because a few megabytes of low-bitrate audio can expand to gigabytes of PCM. Model output is shown with `textContent`. There is no authentication, and the docs say so wherever exposure is discussed.

### The default voice was chosen by licence

The first working voice was the one in most Piper examples. Reading its model card led to a dataset licence that allows research use only and excludes commercial use. I switched the default to a voice whose model card says its data is public domain and re-ran every measurement with it. This is a reminder that a model name is a dependency with terms, not just a file.

## 3. Things that went wrong while building (and what they changed)

* **A dependency API changed under a broad `except`.** PyAV 19 removed a keyword my first decoder passed; my catch-all turned that programming error into a "415 unsupported audio" response. The decoder now catches only PyAV's error types, so the next API change fails loudly as a 500. `faster-whisper` itself pins only `av>=11`, so `pip` installs a PyAV that breaks faster-whisper's own file decoder; this project passes arrays instead and pins `av`.
* **A test that passed for the wrong reason.** My browser end-to-end script reported success in "upload mode" while still using streaming, because it flipped the dropdown before the page's module script had attached its listener, and it never checked which path ran. It now waits for the app to initialise and asserts a transport-specific label.
* **A weak first prompt.** The 1B model ignored "one to three short sentences" and wrote about 120 words for a broad question. A blunter prompt cut the median to about 37 (5 samples per cell; raw data in `benchmarks/results/prompt-length-2026-10-26.txt`). Prompt wording is a latency control as much as a quality one.
* **First measurements were cold.** The first request after start showed an LLM first token around 2 s versus under 0.1 s for the next, which is why cold and warm results are measured and reported separately.

## 4. What the measurements show

The tables are in the README under "Measured results"; the raw files are in `benchmarks/results/`. This section is what the numbers changed in the design, and what they do not support.

**The measurement method had to change mid-way.** The first comparisons ran one configuration after another. Then the identical default configuration, measured twice, came out at 7.31 s and 5.33 s to first audio, and a third and fourth measurement gave 9.92 s (while other programs used 65% of the CPU) and about 5 s. On this laptop the drift between runs is larger than most of the effects being compared. Two changes followed: configurations are interleaved within a run, and settings are compared in rotating rounds with each setting subtracted from the baseline *of the same round*. A change is only reported when its paired difference has the same sign in all three rounds. That is a modest bar and the README says so. It also means several earlier sequential results (for example "first-clause-40 was 3 s faster") were weaker than they looked, and the README quotes the rotation instead.

**What the data supports**

* *Concurrency does not buy throughput on two cores.* Throughput stayed near 0.12–0.17 turns/s from one to four clients while first-audio latency rose from about 7 s to about 20 s, with the STT stage queueing for up to 6 s. This justifies the defaults (every stage limited to one at a time) and admission control that refuses extra turns, rather than queueing them, at `MAX_INFLIGHT_TURNS=4`.
* *The stages interfere.* STT and TTS run several times slower while the LLM generates; the LLM loses 9–18% of its speed when they run. Overlapping stages is the point of streaming, but on this CPU the overlap is paid for in slower STT/TTS, so the benefit is smaller than the pipeline diagram suggests. A machine with more cores, a GPU, or a hosted LLM would change this.
* *Streaming pays off when replies are long.* With ~110-word replies the first sound arrived after 4.6 s through the WebSocket and 29.7 s through the HTTP path (3 samples each, run sequentially). With the default two-sentence replies the two paths were not distinguishable. Reply length is the lever behind this: the stricter default prompt cut the median reply for a broad question from 120 to 37 words (section 3), and a shorter reply shortens the turn on both paths.
* *Whisper's time is nearly independent of utterance length* (about 1.6–1.8 s for 2–7 s of audio on `base.en`; about 0.8–0.9 s on `tiny.en`), consistent with its fixed 30-second window. A faster STT means a smaller model, not shorter input.
* *The best measured combination was `tiny.en` + 2 STT threads + a first chunk ending at a comma.* Its first audio was the lowest of six variants in all three rounds (2.49, 3.15 and 2.57 s; about 2.4 s lower than the baseline in the two rounds with a clean baseline). Its whole-turn time was not consistently better.
* *Limiting Ollama to one thread was a net loss.* Generation became 1.5× slower; the cores it freed made Piper faster but not the first audio.

**What it does not support**

* Whether `tiny.en` is acceptable. The transcription error rate was 0.0% for every configuration, which only shows that clean synthetic speech is easy. It does not measure real microphones, accents or noise. So the lower-latency profile is documented as opt-in, not made the default.
* Whether a first chunk that ends at a comma sounds worse. I could not judge prosody by ear. It is off by default for the same reason.
* Anything about other hardware, other models, a network, or more than one user at a time beyond the small concurrency test.
* Partial transcripts: the STT stage was about 0.35 s longer with them on, and the end-to-end difference was inside the noise, so neither "free" nor "harmful" is established.

**Choices I made about defaults.** I changed no default because of the rotation. Each of the improvements trades something this benchmark cannot see (accuracy on real speech, naturalness of speech), and that trade is the user's to make. They are one-line `.env` settings.

## 5. What I would do next

In rough order of expected value:

1. **Shrink the stages that precede the first sound.** On this hardware STT (1.6–2.1 s), the LLM's first sentence and Piper's first chunk (1.3–2.0 s while the LLM runs) are similar in size, so no single stage dominates. Candidates: a smaller Whisper (measured: about 0.8 s instead of 1.7 s, with an accuracy cost that still has to be measured on real speech), a smaller or more aggressively quantised LLM, or moving one stage to a GPU or a hosted API. Evaluate any accuracy trade with recorded real speech and a word-error-rate script, not with synthetic audio.
2. **Real end-of-speech detection** (voice-activity detection on the client or server), so the user does not press stop, and **barge-in** (interrupt the assistant by speaking).
3. **A streaming recogniser** so STT overlaps with speech instead of following it, and so partial transcripts are genuinely incremental.
4. **WebRTC with Opus** for networks that are not a LAN.
5. **Per-client limits and authentication** before any shared deployment.
6. **CI** running the unit and JS tests on Linux, and a Linux/Docker run of everything that has so far only run on Windows.
7. **Tune with data, per host:** `STT_CPU_THREADS`, the concurrency limits, `TTS_FIRST_CLAUSE_CHARS` and the LLM size should be chosen from `scripts/benchmark.py` runs on the machine that will serve users.
