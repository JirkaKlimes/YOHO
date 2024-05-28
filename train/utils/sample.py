import os
from mutagen.mp4 import MP4
import numpy as np
from typing import Optional
from pathlib import Path
from yoho.src.preprocessing.audio import save_audio, load_audio


class AudioSample:
    def __init__(
        self,
        pcmdata: np.ndarray,
        sample_rate: int,
        transcript: Optional[str] = None,
        path: Optional[Path] = None,
    ):
        self._pcmdata = pcmdata
        self.sample_rate = sample_rate
        self.transcript = transcript
        self._path = path

    def save(self, path: Path):
        assert path.suffix == ".mp4", "Use `.mp4` since it's the fastest format"
        if path.exists():
            os.remove(path)
        save_audio(self.pcmdata, path, self.sample_rate)
        if self.transcript:
            audio_file = MP4(path)
            audio_file.tags["\xa9cmt"] = self.transcript
            audio_file.save()

    @classmethod
    def load(cls, path: Path, sample_rate: int):
        assert path.suffix == ".mp4", "Can only load `.mp4` files"
        audio_file = MP4(path)
        transcript = audio_file.tags.get("\xa9cmt", [None])[0]
        return cls(None, sample_rate, transcript, path)

    @property
    def pcmdata(self) -> np.ndarray:
        if self._pcmdata is None:
            self._pcmdata = load_audio(path, sample_rate)
        return self._pcmdata


if __name__ == "__main__":
    path = Path("./data/sample.mp4")

    sample_rate = 16000
    pcm_audio = np.random.uniform(-32768, 32768, size=5 * sample_rate).astype(np.int16)
    transcript = "Hello, World!"

    sample = AudioSample(pcm_audio, sample_rate, transcript)
    sample.save(path)

    loaded_sample = AudioSample.load(path, sample_rate)
    print(loaded_sample.transcript)
    print(loaded_sample.pcmdata)
