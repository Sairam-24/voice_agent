"""Environment-driven configuration.

Every setting can be set as an environment variable (case-insensitive) or in a
local `.env` file. Names match the field names, e.g. `STT_MODEL`, `OLLAMA_MODEL`.
Secrets are `SecretStr` so they never show up in logs or `repr()`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Small local models ignore soft limits like "one to three sentences" (llama3.2:1b wrote ~120 words
# for a broad question). This stricter wording cut the median to ~37; see benchmarks/results/.
DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Your reply is spoken aloud. Reply in at most two short sentences, "
    "under 40 words in total. Use plain spoken language with no lists, markdown, or emoji. "
    "Never give more than two sentences, even if the question is broad. "
    "Always reply in the same language the user speaks."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- app ---
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    # Unset = serve the React build (frontend/dist) if it exists, else the plain page in web/.
    # Set it to serve exactly one folder. See app/webui.py.
    web_dir: Path | None = None
    # Load models and send one tiny request through each at startup, so the first
    # real turn does not pay one-time initialisation costs. Turn off to measure cold starts.
    warmup_on_start: bool = True

    # --- local or cloud: which model each stage uses ---
    # "auto" (the default) = Gemini when a Gemini API key is available, the local model otherwise, so
    # the app works with no key at all. The key is GEMINI_API_KEY below, or one a user enters in the
    # browser, which is then used for that user's questions only. "gemini" means the same as "auto".
    # whisper / ollama / local keep a stage on this machine even when there is a key.

    # --- speech-to-text ---
    # Local: faster-whisper (audio never leaves the machine). Gemini: far better than a small Whisper
    # for many languages (Hindi in Devanagari, for example) and no local CPU; the audio is sent to Google.
    stt_provider: Literal["auto", "whisper", "gemini"] = "auto"
    gemini_stt_model: str = "gemini-3.5-flash-lite"
    # With STT_PROVIDER=gemini: transcribe locally with Whisper when Gemini refuses (rate limit, network).
    stt_fallback_to_whisper: bool = True

    stt_model: str = "base.en"  # the Whisper model
    stt_model_dir: Path = Path("models/whisper")
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_cpu_threads: int = Field(0, ge=0, description="0 = CTranslate2 default")
    stt_beam_size: int = Field(1, ge=1, le=10)
    stt_vad_filter: bool = True
    # "en" (default) or any Whisper language code. Empty or "auto" = detect the language of each
    # utterance, which needs a multilingual model (a name without ".en", e.g. "base" or "small").
    stt_language: str | None = "en"
    # With STT_LANGUAGE=auto: the languages the speaker actually uses, comma-separated (e.g. "te,en").
    # Gemini is told about them (without this it wrote some Telugu sentences as Hindi translations),
    # and Whisper picks the most likely of them instead of any of its ~100 languages.
    stt_languages: str = ""

    # --- language model ---
    llm_provider: Literal["auto", "ollama", "gemini"] = "auto"
    # With Gemini: let the local Ollama model answer when Gemini refuses (quota, an invalid key).
    llm_fallback_to_ollama: bool = True
    llm_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_max_output_tokens: int = Field(160, ge=1, le=4096)
    llm_temperature: float = Field(0.6, ge=0.0, le=2.0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_keep_alive: str = "30m"
    # CPU threads Ollama may use for generation. None = Ollama's default (all physical cores). Lowering
    # it leaves cores free for STT and TTS, at the cost of slower token generation.
    ollama_num_thread: int | None = Field(None, ge=1, le=64)

    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Optional. Newer Gemini models "think" before answering, which adds latency and
    # consumes max_output_tokens. Unset = use the model's default.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    # --- text-to-speech ---
    # Local: voices on this machine, one per script, chosen per sentence. Gemini: a natural,
    # multilingual cloud voice; the free tier allows very few TTS requests (10 a day per model when
    # tested), so one request is made per reply.
    tts_provider: Literal["auto", "local", "gemini"] = "auto"
    # One model, or several separated by commas: each has its own free daily quota, so when one is
    # used up the next is tried before falling back to the local voices.
    gemini_tts_model: str = "gemini-3.8-flash-lite-tts"
    gemini_tts_voice: str = "Kore"
    # With TTS_PROVIDER=gemini: speak with the local voices when Gemini refuses (free-tier quota,
    # network error) instead of leaving the reply silent.
    tts_fallback_to_local: bool = True

    # Local English voice: piper (fast, synthetic-sounding) or kokoro (natural; about real time on a
    # 2-core laptop CPU, so there can be a short pause between sentences).
    tts_local_english: Literal["piper", "kokoro"] = "piper"
    tts_voice: str = "en_US-ljspeech-medium"  # the Piper English voice
    kokoro_voice: str = "af_heart"
    kokoro_model_dir: Path = Path("models/kokoro")
    # Local Telugu voice (a Piper voice name), e.g. te_IN-padmavathi-medium. Empty = none: Telugu
    # replies are then shown as text unless the Gemini voice is used.
    tts_telugu_voice: str = ""
    tts_model_dir: Path = Path("models/piper")
    # Streaming only. 0 = speak whole sentences. N > 0 = the first chunk may end at a comma once
    # N characters have been generated, trading a little prosody for an earlier first sound.
    tts_first_clause_chars: int = Field(0, ge=0, le=200)

    # --- concurrency and backpressure ---
    # How many requests may run each stage at once. On a small CPU box more than 1
    # usually makes every request slower, not faster; scripts/benchmark.py measures it.
    stt_concurrency: int = Field(1, ge=1, le=16)
    llm_concurrency: int = Field(1, ge=1, le=16)
    tts_concurrency: int = Field(1, ge=1, le=16)
    # Turns admitted at once (running + waiting for a stage slot). Beyond this we
    # answer "busy" immediately instead of building an unbounded queue.
    max_inflight_turns: int = Field(4, ge=1, le=256)

    # --- timeouts (seconds) ---
    stt_timeout_s: float = Field(30.0, gt=0)
    llm_first_token_timeout_s: float = Field(30.0, gt=0)
    llm_timeout_s: float = Field(60.0, gt=0)
    tts_timeout_s: float = Field(30.0, gt=0)
    turn_timeout_s: float = Field(120.0, gt=0)

    # --- input limits ---
    max_audio_seconds: float = Field(30.0, gt=0)
    max_upload_bytes: int = Field(10_000_000, gt=0)
    max_history_turns: int = Field(6, ge=0, le=50)

    # --- attached documents (ask questions about a file) ---
    max_document_bytes: int = Field(10_000_000, gt=0)
    # How much of the attached files' text goes into each prompt, in characters. Unset = 16000 with
    # Gemini; 3000 with Ollama, whose prompt processing on a laptop CPU takes seconds per thousand.
    document_context_chars: int | None = Field(None, ge=500, le=1_000_000)

    # --- WebSocket ---
    # Whisper is not a streaming recognizer. When enabled, the server re-transcribes the audio
    # recorded so far every partial_interval_s and sends it as a non-final transcript. Each
    # partial is a full inference, so on a small CPU it can delay the final transcript.
    partial_transcripts: bool = False
    partial_interval_s: float = Field(2.0, gt=0)
    # Browsers send an Origin header on WebSocket handshakes. Same-origin is always allowed;
    # list extra origins here (comma-separated) only if the page is served from elsewhere.
    allowed_origins: str = ""

    @field_validator(
        "gemini_api_key", "gemini_thinking_level", "ollama_num_thread", "web_dir", "document_context_chars", mode="before"
    )
    @classmethod
    def blank_means_unset(cls, value):
        # `GEMINI_API_KEY=` copied from .env.example arrives as "", not as a missing value.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("tts_provider", mode="before")
    @classmethod
    def piper_means_local(cls, value):
        return "local" if value == "piper" else value  # the name used before there were several local voices

    @property
    def cloud_stages(self) -> frozenset[str]:
        """The stages that use Gemini when a key is available."""
        local = {"stt": "whisper", "llm": "ollama", "tts": "local"}
        chosen = {"stt": self.stt_provider, "llm": self.llm_provider, "tts": self.tts_provider}
        return frozenset(stage for stage, provider in chosen.items() if provider != local[stage])

    @property
    def gemini_tts_models(self) -> list[str]:
        return [name.strip() for name in self.gemini_tts_model.split(",") if name.strip()]

    @property
    def document_budget(self) -> int:
        if self.document_context_chars:
            return self.document_context_chars
        return 16_000 if self.llm_provider == "gemini" else 3_000

    @property
    def expected_languages(self) -> list[str]:
        return [code.strip().lower() for code in self.stt_languages.split(",") if code.strip()]

    @field_validator("stt_language", mode="before")
    @classmethod
    def auto_language(cls, value):
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
