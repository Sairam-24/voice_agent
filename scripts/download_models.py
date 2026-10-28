"""Download the Whisper model and the local voices configured in .env (or the defaults).

    python scripts/download_models.py

The server also downloads missing models on first start; running this ahead of time
just moves the wait out of the first launch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.services.stt import ensure_whisper_model  # noqa: E402
from app.services.tts import ensure_kokoro_files, ensure_piper_voice  # noqa: E402


def main() -> None:
    s = get_settings()
    print(f"Whisper model '{s.stt_model}' -> {s.stt_model_dir}")
    whisper_dir = ensure_whisper_model(s.stt_model, s.stt_model_dir)
    print(f"  ready: {whisper_dir}")

    for voice in filter(None, [s.tts_voice, s.tts_telugu_voice]):
        print(f"Piper voice '{voice}' -> {s.tts_model_dir}")
        print(f"  ready: {ensure_piper_voice(voice, s.tts_model_dir)}")

    if s.tts_local_english == "kokoro":
        print(f"Kokoro English voice -> {s.kokoro_model_dir} (about 350 MB)")
        print(f"  ready: {', '.join(str(p) for p in ensure_kokoro_files(s.kokoro_model_dir))}")

    print("\nCheck each model's license before redistributing anything (see README, 'Licensing').")


if __name__ == "__main__":
    main()
