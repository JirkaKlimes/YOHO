from pathlib import Path
import os
import time
import numpy as np
import pickle
from yoho.src.preprocessing.audio import load_audio
from tabulate import tabulate

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
# | sample.opus | .opus         |         0.318685 |               0.249073    | 346.25x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.wav  | .wav          |         5.48256  |               0.0574982   | 79.93x slower          |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.pkl  | .pkl          |         5.48264  |               0.000719351 | 1.00x slower           |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.mp3  | .mp3          |         0.343056 |               0.104334    | 145.04x slower         |
# +-------------+---------------+------------------+---------------------------+------------------------+
# | sample.npy  | .npy          |         5.4826   |               0.00094827  | 1.32x slower           |
# +-------------+---------------+------------------+---------------------------+------------------------+
