import numpy as np
from functools import cache


def hz_to_mel(frequencies):
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_frequencies(n_mels: int, fmin: float, fmax: float):
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)
    hz = mel_to_hz(mels)
    return hz


@cache
def mel_filter_banks(
    sr: float,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    fmax = sr * 0.5

    weights = np.zeros((n_mels, 1 + n_fft // 2), np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_frequencies(n_mels + 2, fmin=0.0, fmax=fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights
