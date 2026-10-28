import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert (s.stt_provider, s.llm_provider, s.tts_provider) == ("auto", "auto", "auto")
    assert s.cloud_stages == {"stt", "llm", "tts"}, "every stage uses Gemini when there is a key"
    assert s.stt_concurrency == 1
    assert s.gemini_api_key is None


def test_local_provider_names_keep_a_stage_local():
    s = Settings(_env_file=None, stt_provider="whisper", llm_provider="gemini", tts_provider="local")
    assert s.cloud_stages == {"llm"}


def test_voice_and_language_settings():
    assert Settings(_env_file=None, tts_provider="piper").tts_provider == "local", "the old name still works"
    assert Settings(_env_file=None, stt_languages=" te, EN ,").expected_languages == ["te", "en"]
    assert Settings(_env_file=None).expected_languages == []


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("STT_MODEL", "tiny.en")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MAX_INFLIGHT_TURNS", "9")
    s = Settings(_env_file=None)
    assert (s.stt_model, s.llm_provider, s.max_inflight_turns) == ("tiny.en", "gemini", 9)


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_PROVIDER", "carrier-pigeon"),
        ("STT_CONCURRENCY", "0"),
        ("TURN_TIMEOUT_S", "0"),
        ("LLM_TEMPERATURE", "9"),
        ("MAX_AUDIO_SECONDS", "-1"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_is_never_shown_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    s = Settings(_env_file=None)
    assert "super-secret-key" not in repr(s)
    assert s.gemini_api_key.get_secret_value() == "super-secret-key"


def test_env_example_parses_and_matches_defaults(tmp_path, monkeypatch):
    """.env.example must stay loadable (inline comments, empty secret) and not drift from defaults."""
    for key in list(__import__("os").environ):
        if key.isupper() and key.startswith(("STT_", "LLM_", "OLLAMA_", "GEMINI_", "TTS_", "MAX_", "TURN_", "LOG_")):
            monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    shutil.copy(ENV_EXAMPLE, env_file)
    from_example = Settings(_env_file=env_file)
    defaults = Settings(_env_file=None)
    assert from_example == defaults
