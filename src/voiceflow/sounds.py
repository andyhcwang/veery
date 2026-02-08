"""Subtle audio feedback for recording start/stop.

Generates short, warm tones at import time and plays them via NSSound.
Tones are custom (not macOS system sounds) to avoid confusion with
system alerts. Designed to be soft, non-jarring, and under 100ms.
"""

from __future__ import annotations

import io
import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded NSSound instances
_start_sound = None
_stop_sound = None
_loaded = False


def _generate_tone(
    freq_start: float,
    freq_end: float,
    duration_ms: int,
    amplitude: float = 0.15,
) -> np.ndarray:
    """Generate a smooth, warm tone with frequency sweep.

    Uses sine wave + 2nd harmonic (warmth) with a sine-shaped envelope
    to avoid any clicks or harsh transients.
    """
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)

    # Frequency sweep
    freqs = np.linspace(freq_start, freq_end, n_samples)
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate

    # Sine + touch of 2nd harmonic for warmth
    wave = np.sin(phase) * 0.85 + np.sin(2 * phase) * 0.15

    # Smooth sine envelope — zero at both ends, no clicks
    envelope = np.sin(np.linspace(0, np.pi, n_samples)) ** 0.7

    return (wave * envelope * amplitude).astype(np.float32)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Pack float32 mono audio into a WAV byte string (16-bit PCM)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    data_size = len(pcm) * 2
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


def _load_sounds() -> None:
    """Generate tones and wrap as NSSound objects."""
    global _start_sound, _stop_sound, _loaded
    if _loaded:
        return

    try:
        import AppKit

        # Start: soft rising warm tone (480→580 Hz, 90ms)
        start_wav = _to_wav_bytes(_generate_tone(480, 580, 90, amplitude=0.15))
        start_data = AppKit.NSData.dataWithBytes_length_(start_wav, len(start_wav))
        _start_sound = AppKit.NSSound.alloc().initWithData_(start_data)

        # Stop: softer descending tone (540→420 Hz, 70ms)
        stop_wav = _to_wav_bytes(_generate_tone(540, 420, 70, amplitude=0.12))
        stop_data = AppKit.NSData.dataWithBytes_length_(stop_wav, len(stop_wav))
        _stop_sound = AppKit.NSSound.alloc().initWithData_(stop_data)

        _loaded = True
        logger.debug("Audio feedback sounds loaded")
    except Exception:
        logger.debug("Could not load audio feedback sounds")


def play_start() -> None:
    """Play the recording-start sound (non-blocking)."""
    _load_sounds()
    if _start_sound is not None:
        _start_sound.stop()
        _start_sound.play()


def play_stop() -> None:
    """Play the recording-stop sound (non-blocking)."""
    _load_sounds()
    if _stop_sound is not None:
        _stop_sound.stop()
        _stop_sound.play()
