from pathlib import Path
import os
import time
import numpy as np
import pickle
from tabulate import tabulate
from librosa import load

from yoho.src.preprocessing.audio import load_audio

results = []
pickle_duration = None

for path in Path("./data/audio").iterdir():
    suffix = path.suffix
    file_size = os.stat(path).st_size / (1024 * 1024)

    st = time.monotonic()
    if suffix == ".npy":
        np.load(path)
    elif suffix == ".pkl":
        with open(path, "rb") as f:
            pickle.load(f)
    elif suffix in [".mp3", ".wav"]:
        load(path, sr=16000)
    else:
        load_audio(path, 16000)
    et = time.monotonic()

    duration = et - st
    if suffix == ".pkl":
        pickle_duration = duration

    results.append([path.name, suffix, file_size, duration])

if pickle_duration:
    for result in results:
        result.append(f"{result[3] / pickle_duration:.02f}x slower")
    headers = [
        "File Name",
        "File Suffix",
        "File Size (MB)",
        "Load Duration (seconds)",
        "Relative Performance",
    ]
else:
    headers = ["File Name", "File Suffix", "File Size (MB)", "Load Duration (seconds)"]

print(tabulate(results, headers=headers, tablefmt="grid"))
# +-------------+---------------+------------------+---------------------------+------------------------+
# | File Name   | File Suffix   |   File Size (MB) |   Load Duration (seconds) | Relative Performance   |
# +=============+===============+==================+===========================+========================+
# | sample.opus | .opus         |         0.318685 |               0.326616    | 502.31x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.spx  | .spx          |         0.287523 |               0.269639    | 414.69x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.wav  | .wav          |         5.48256  |               0.213095    | 327.72x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.pkl  | .pkl          |         5.48264  |               0.000650225 | 1.00x slower           |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.mp4  | .mp4          |         0.357733 |               0.0965566   | 148.50x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.mp3  | .mp3          |         0.343056 |               0.0282004   | 43.37x slower          |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.m4a  | .m4a          |         0.357714 |               0.0901698   | 138.67x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.npy  | .npy          |         5.4826   |               0.000877127 | 1.35x slower           |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.tta  | .tta          |         5.06743  |               0.060562    | 93.14x slower          |
# +-------------+---------------+------------------+---------------------------+------------------------+
