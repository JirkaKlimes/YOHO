import ffmpeg
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from yoho.src.preprocessing.mel_filterbanks import mel_filter_banks
from yoho.src.config import YOHOConfig


def load_audio(path: Path, sample_rate: int):
    raw_bytes = (
        ffmpeg.input(path)
        .output("pipe:", format="s16le", ac=1, acodec="pcm_s16le", ar=sample_rate)
        .run(capture_stdout=True, quiet=True)[0]
    )
    audio = np.frombuffer(raw_bytes, np.int16).flatten()
    return audio


def save_audio(audio: np.ndarray, path: Path, sample_rate: int):
    process = (
        ffmpeg.input("pipe:0", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
        .output(str(path.with_suffix(".mp4")), format="mp4", audio_bitrate="16k")
        .run_async(pipe_stdin=True, quiet=True)
    )
    process.stdin.write(audio.tobytes())
    process.stdin.close()
    process.wait()


def mel_spectogram(audio, n_fft: int, hop_len: int, sample_rate: int, n_mels: int):
    stft = jax.scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop_len, boundary=None)[-1]
    magnitudes = jnp.abs(stft) ** 2
    filters = mel_filter_banks(sample_rate, n_fft, n_mels)
    spectogram = jnp.einsum("ij,ajk->aki", filters, magnitudes)
    return spectogram


def get_batched_spectogram(config: YOHOConfig):
    def func(audio):
        return mel_spectogram(
            audio,
            config.n_fft,
            config.stft_hop,
            config.sample_rate,
            config.n_mel_bands,
        )

    return func


def normalize_spectogram(spectogram):
    spec = jnp.log10(jnp.clip(spectogram, a_min=1e-10))
    spec = jnp.maximum(spec, spec.max() - 8.0)
    spec = (spec + 4.0) / 4.0
    return spec
