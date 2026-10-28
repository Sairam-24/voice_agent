import io
import wave

import av
import numpy as np
import pytest

from app.errors import AudioTooLongError, EmptyAudioError, UnsupportedAudioError
from app.services.audio import decode_audio_capped, pcm16_seconds, pcm16_to_float32, pcm16_to_wav
from tests.conftest import sine_wav


def stereo_wav(seconds: float, sample_rate: int) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    left = (np.sin(2 * np.pi * 300 * t) * 10000).astype("<i2")
    right = (np.sin(2 * np.pi * 600 * t) * 10000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.column_stack([left, right]).tobytes())
    return buffer.getvalue()


def webm_opus(seconds: float = 1.0, sample_rate: int = 48000) -> bytes:
    """What a browser's MediaRecorder typically uploads, encoded here with PyAV."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    samples = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2").reshape(1, -1)
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=sample_rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def test_decodes_mono_wav_to_float32_16k():
    audio = decode_audio_capped(sine_wav(1.0, 16000), max_seconds=30)
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(16000, abs=50)
    assert 0.2 < np.abs(audio).max() < 0.5  # a 12000/32768 sine; proves scaling to [-1, 1]


def test_resamples_and_downmixes_stereo_44k_wav():
    audio = decode_audio_capped(stereo_wav(2.0, 44100), max_seconds=30)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(2 * 16000, abs=100)


def test_decodes_browser_style_webm_opus():
    audio = decode_audio_capped(webm_opus(1.0), max_seconds=30)
    assert len(audio) == pytest.approx(16000, rel=0.1)
    assert np.abs(audio).max() > 0.1


@pytest.mark.parametrize("junk", [b"\x00\x01\x02" * 2000, b"this is definitely not audio" * 40])
def test_undecodable_bytes_raise_unsupported_audio(junk):
    with pytest.raises(UnsupportedAudioError):
        decode_audio_capped(junk, max_seconds=30)


def test_empty_upload_raises_empty_audio():
    with pytest.raises(EmptyAudioError):
        decode_audio_capped(b"", max_seconds=30)


def test_valid_wav_with_zero_samples_raises_empty_audio():
    with pytest.raises(EmptyAudioError):
        decode_audio_capped(pcm16_to_wav(b"", 16000), max_seconds=30)


def test_audio_over_the_limit_is_rejected_while_decoding():
    with pytest.raises(AudioTooLongError):
        decode_audio_capped(sine_wav(3.0), max_seconds=1.0)


def test_pcm16_to_float32_scales_and_ignores_a_dangling_byte():
    pcm = np.array([0, 16384, -32768], dtype="<i2").tobytes() + b"\x7f"
    np.testing.assert_allclose(pcm16_to_float32(pcm), [0.0, 0.5, -1.0])


def test_wav_round_trip_keeps_format_and_length():
    pcm = np.arange(2205, dtype="<i2").tobytes()  # 0.1 s at 22050 Hz
    wav_bytes = pcm16_to_wav(pcm, 22050)
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 22050)
        assert wav.getnframes() == 2205
    assert pcm16_seconds(pcm, 22050) == pytest.approx(0.1)
