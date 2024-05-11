import warnings
import numpy as np


def hz_to_mel(frequencies, htk=False):
    frequencies = np.asarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = frequencies >= min_log_hz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore divide by zero in log
        mels = np.where(log_t, min_log_mel + np.log(frequencies / min_log_hz) / logstep, mels)
    return mels


def mel_to_hz(mels, htk=False):
    mels = np.asarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs = np.where(log_t, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def mel_frequencies(
    n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> np.ndarray:
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    hz = mel_to_hz(mels, htk=htk)
    return hz


def generate_mel_filters(
    sr: float,
    n_fft: int,
    n_mels: int,
    htk: bool = False,
) -> np.ndarray:
    """Create a Mel filter-bank.

    Produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.
    """
    fmax = sr * 0.5

    weights = np.zeros((n_mels, 1 + n_fft // 2), np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_frequencies(n_mels + 2, fmin=0.0, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


if __name__ == "__main__":
    import librosa

    SR = 16000
    N_FFT = 400
    N_MELS = 80

    filters = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    filters_jax = generate_mel_filters(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    print(np.all(filters == filters_jax))
