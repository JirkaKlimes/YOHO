import ffmpeg
from pathlib import Path
import numpy as np
import scipy
import scipy.signal

from yoho.src.preprocessing.mel_spectogram import generate_mel_filters


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
        .output(str(path.with_suffix(".opus")), format="opus", audio_bitrate="16k")
        .run_async(pipe_stdin=True, quiet=True)
    )
    process.stdin.write(audio.tobytes())
    process.stdin.close()
    process.wait()


def mel_spectogram(
    audio, n_fft: int, hop_len: int, sample_rate: int, n_mels: int, htk: bool = False
):
    stft = scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop_len, boundary=None)[-1]
    magnitudes = np.abs(stft) ** 2
    filters = generate_mel_filters(sample_rate, n_fft, n_mels, htk)
    spectogram = filters @ magnitudes
    return spectogram.T


def normalize_spectogram(spectogram):
    normalized = np.log10(np.clip(spectogram, a_min=1e-10))
    normalized = np.maximum(normalized, normalized.max() - 8.0)
    normalized = (normalized + 4.0) / 4.0
    return normalized
