from typing import Any
import srt
from eld import LanguageDetector
import numpy as np
import bisect

from yoho.src.config import YOHOConfig

from train.utils.base_dataloader import Dataloader
from train.utils.config import CONFIG


class TranscriptionDataloader(Dataloader):
    def __init__(
        self,
        yoho_config: YOHOConfig,
        batch_size: int,
        shuffle: bool = True,
        max_queued_batches: int = 8,
        num_workers: int = 4,
        warmup_queue: bool = True,
        use_multiprocessing: bool = True,
    ):
        self.yoho_config = yoho_config
        self.shuffle = shuffle

        language_detector = LanguageDetector()

        all_paths = CONFIG.dataset.noisy.joinpath("transcripts").iterdir()
        sizes = []
        paths = []
        for path in all_paths:
            with open(path, encoding="utf-8") as f:
                data = f.read()
            transcript = list(srt.parse(data))
            content = "\n".join([utterance.content for utterance in transcript])
            lang = language_detector.detect(content).language
            if lang not in CONFIG.language_whitelist:
                continue
            sizes.append(len(transcript))
            paths.append(path)

        self.sizes = np.array(sizes, dtype=np.uint64).cumsum()
        self.paths = paths
        self.index_table = np.arange(self.sizes[-1], dtype=np.uint64)
        if self.shuffle:
            np.random.shuffle(self.index_table)

        super().__init__(
            batch_size, max_queued_batches, num_workers, warmup_queue, use_multiprocessing
        )

    def get_num_batches(self) -> int:
        return self.sizes[-1] // self.batch_size

    def get_batch(self, idx: int) -> Any:
        i = idx * self.batch_size
        j = (idx + 1) * self.batch_size

        sample_indices = self.index_table[i:j]
        path_indices = [bisect.bisect_right(self.sizes, i) for i in sample_indices]

        transcripts = {}
        for i in set(path_indices):
            with open(self.paths[i], encoding="utf-8") as f:
                data = f.read()
            transcripts[i] = list(srt.parse(data))

        for si, pi in zip(sample_indices, path_indices):
            si = int(si - (0 if pi == 0 else self.sizes[pi - 1]))
            transcript = transcripts[pi]
            start_time = None if si == 0 else transcript[si - 1].end
            start_speech_time = transcript[si].start
            while (
                transcript[si + 1].end - start_speech_time
            ).total_seconds() < self.yoho_config.max_input_seconds:
                si += 1
            end_speech_time = transcript[si].end
            end_time = None if si == len(transcript) - 2 else transcript[si + 1].start

            print(f"--> {self.paths[pi].name}")
            print(start_time)
            print(start_speech_time)
            print(end_speech_time)
            print(end_time)


dataloader = TranscriptionDataloader(YOHOConfig(), 16, use_multiprocessing=False, shuffle=False)
dataloader.get_prepared_batch()
