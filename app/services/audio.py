"""Audio decoding and encoding helpers.

Browsers upload compressed audio (MediaRecorder produces WebM/Opus or MP4), while
the speech model wants 16 kHz mono float32. PyAV, which faster-whisper already
depends on, bundles FFmpeg's libraries, so no system ffmpeg install is needed.
"""

import io
import wave

import av
import numpy as np

from app.errors import AudioTooLongError, EmptyAudioError, UnsupportedAudioError

SAMPLE_RATE = 16_000  # what Whisper expects


def decode_audio_capped(data: bytes, max_seconds: float) -> np.ndarray:
    """Decode an uploaded audio file to float32 mono at 16 kHz.

    faster_whisper.decode_audio does the same job but decodes the whole file before
    we can look at its length. A few MB of low-bitrate Opus can expand to gigabytes
    of PCM, so this version stops as soon as the cap is exceeded.
    """
    if not data:
        raise EmptyAudioError("The audio upload was empty.")

    max_samples = int(max_seconds * SAMPLE_RATE)
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pieces: list[np.ndarray] = []
    total = 0

    def collect(frames) -> None:
        nonlocal total
        for frame in frames:
            samples = frame.to_ndarray().reshape(-1)  # packed mono s16 -> 1-D
            total += samples.size
            if total > max_samples:
                raise AudioTooLongError(f"Audio is longer than the {max_seconds:g}s limit.")
            pieces.append(samples)

    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                raise UnsupportedAudioError("The file contains no audio track.")
            decoder = container.decode(audio=0)
            while True:
                try:
                    frame = next(decoder)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    continue  # skip a corrupt packet, keep what decodes
                frame.pts = None  # the resampler rejects timestamp gaps; we only need the samples
                collect(resampler.resample(frame))
            collect(resampler.resample(None))  # flush
    except (av.error.FFmpegError, ValueError) as exc:
        # PyAV reports undecodable input as FFmpegError subclasses (InvalidDataError is also
        # a ValueError). Anything else, e.g. a TypeError from a PyAV API change, is our bug
        # and must surface as a 500, not be blamed on the client's upload.
        raise UnsupportedAudioError("Could not decode the audio. Send WAV, WebM/Opus, Ogg, MP3 or MP4.") from exc

    if total == 0:
        raise EmptyAudioError("The audio contained no samples.")
    return np.concatenate(pieces).astype(np.float32) / 32768.0


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Raw little-endian signed 16-bit mono PCM -> float32 in [-1, 1]."""
    usable = len(pcm) - (len(pcm) % 2)  # drop a dangling half-sample if a frame was cut
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def pcm16_seconds(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / 2 / sample_rate
