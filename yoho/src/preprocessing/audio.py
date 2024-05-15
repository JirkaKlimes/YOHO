import ffmpeg
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from yoho.src.preprocessing.mel_spectogram import generate_mel_filters


def load_audio(path: Path, sample_rate: int):
    raw_bytes = (
        ffmpeg.input(path)
        .output("pipe:", format="s16le", ac=1, acodec="pcm_s16le", ar=sample_rate)
        .run(capture_stdout=True, quiet=True)[0]
    )
    audio = np.frombuffer(raw_bytes, np.int16).flatten() / 32768.0
    return audio


def mel_spectogram(audio, n_fft: int, hop_len: int, sample_rate: int, n_mels: int):
    stft = jax.scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop_len)[-1]
    magnitudes = jnp.abs(stft[..., :-1]) ** 2
    filters = generate_mel_filters(sample_rate, n_fft, n_mels)
    spectogram = filters @ magnitudes
    return spectogram.T


def normalize_spectogram(spectogram):
    normalized = jnp.log10(jnp.clip(spectogram, a_min=1e-10))
    normalized = jnp.maximum(normalized, normalized.max() - 8.0)
    normalized = (normalized + 4.0) / 4.0
    return normalized
