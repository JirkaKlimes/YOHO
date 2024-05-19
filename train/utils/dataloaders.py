from typing import Any
import srt
from eld import LanguageDetector
import numpy as np
import bisect
import datetime as dt

from yoho.src.config import YOHOConfig
from yoho.src.preprocessing.audio import load_audio
from yoho.src.preprocessing.tokenizer import BPEasyTokenizer

from train.utils.base_dataloader import Dataloader
from train.utils.standardize_text import standardize
from train.utils.config import CONFIG


class TranscriptionDataloader(Dataloader):
    def __init__(
        self,
        yoho_config: YOHOConfig,
        tokenizer: BPEasyTokenizer,
        batch_size: int,
        shuffle: bool = True,
        max_queued_batches: int = 8,
        num_workers: int = 4,
        warmup_queue: bool = True,
        use_multiprocessing: bool = True,
    ):
        self.yoho_config = yoho_config
        self.tokenizer = tokenizer
        self.shuffle = shuffle

        language_detector = LanguageDetector()

        self.root_path = CONFIG.dataset.noisy
        all_paths = self.root_path.joinpath("transcripts").iterdir()
        sizes = []
        paths = []
        langs = []
        for path in all_paths:
            with open(path, encoding="utf-8") as f:
                data = f.read()
            transcript = list(srt.parse(data))
            content = "\n".join([utterance.content for utterance in transcript])
            lang = language_detector.detect(content).language
            if lang not in CONFIG.language_whitelist:
                continue
            sizes.append(len(transcript))
            paths.append((path, self.root_path.joinpath("audio", path.with_suffix(".opus").name)))
            langs.append(lang)

        self.sizes = np.array(sizes, dtype=np.uint64).cumsum()
        self.paths = paths
        self.langs = langs
        self.index_table = np.arange(self.sizes[-1], dtype=np.uint64)
        if self.shuffle:
            np.random.shuffle(self.index_table)

        super().__init__(
            batch_size, max_queued_batches, num_workers, warmup_queue, use_multiprocessing
        )

    def randomize_padding(self, start_time, end_time, start_speech_time, end_speech_time):
        duration = (end_speech_time - start_speech_time).total_seconds()
        padding_left = (start_speech_time - start_time).total_seconds()
        padding_left = np.random.uniform(
            0, min(padding_left, self.yoho_config.max_input_seconds - duration)
        )
        start_time = start_speech_time - dt.timedelta(seconds=padding_left)
        duration = (end_speech_time - start_time).total_seconds()
        padding_right = (end_time - end_speech_time).total_seconds()
        padding_right = np.random.uniform(
            0, min(padding_right, self.yoho_config.max_input_seconds - duration)
        )
        end_time = end_speech_time + dt.timedelta(seconds=padding_right)
        return start_time, end_time

    def load_sample_utterances(self, sample_index, transcript, audio, lang):
        si = sample_index

        utterances = []
        utterances.append(transcript[si])
        start_time = dt.timedelta() if si == 0 else transcript[si - 1].end
        start_speech_time = transcript[si].start
        while True:
            if si >= len(transcript) - 2:
                break
            if (
                not (transcript[si + 1].end - start_speech_time).total_seconds()
                < self.yoho_config.max_input_seconds
            ):
                break
            si += 1
            utterances.append(transcript[si])
        end_speech_time = transcript[si].end
        end_time = (
            dt.timedelta(seconds=len(audio) / self.yoho_config.sample_rate)
            if si >= len(transcript) - 2
            else transcript[si + 1].start
        )

        start_time, end_time = self.randomize_padding(
            start_time, end_time, start_speech_time, end_speech_time
        )
        from_sample = int(np.ceil(start_time.total_seconds() * self.yoho_config.sample_rate))
        to_sample = int(np.floor(end_time.total_seconds() * self.yoho_config.sample_rate))
        audio = audio[from_sample:to_sample]

        if len(audio) > self.yoho_config.n_samples:
            return None, None

        audio = np.pad(audio, (0, self.yoho_config.n_samples - len(audio)))

        utterances_relative = [
            (
                int(
                    np.floor((ut.start - start_time).total_seconds() * self.yoho_config.sample_rate)
                ),
                int(np.ceil((ut.end - start_time).total_seconds() * self.yoho_config.sample_rate)),
                standardize(ut.content, lang=lang),
            )
            for ut in utterances
        ]

        return audio, utterances_relative

    def get_num_batches(self) -> int:
        return self.sizes[-1] // self.batch_size

    def get_batch(self, idx: int) -> Any:
        i = idx * self.batch_size
        j = (idx + 1) * self.batch_size

        samples = []
        for sample_idx in self.index_table[i:j]:
            while True:
                asset_index = bisect.bisect_right(self.sizes, sample_idx)
                transcript_path, audio_path = self.paths[asset_index]
                lang = self.langs[asset_index]

                with open(transcript_path, encoding="utf-8") as f:
                    data = f.read()
                transcript = list(srt.parse(data))
                audio = load_audio(audio_path, self.yoho_config.sample_rate)

                relative_sample_idx = int(
                    sample_idx - (0 if asset_index == 0 else self.sizes[asset_index - 1])
                )
                audio, utterances = self.load_sample_utterances(
                    relative_sample_idx, transcript, audio, lang
                )
                if audio is not None:
                    break

                # TODO: this is ugly (if sample is too long, we discard and take next one)
                sample_idx = int((sample_idx + 1) % self.sizes[-1])
            samples.append((audio, utterances))

        audio_batch = []
        tokens_batch = []

        for audio, utterances in samples:
            transcript = "<|startoftranscript|>"
            for start, end, content in utterances:
                start = int(
                    min(
                        np.floor(start / self.yoho_config.stft_hop),
                        self.yoho_config.max_audio_len - 1,
                    )
                )
                end = int(
                    min(
                        np.floor(end / self.yoho_config.stft_hop),
                        self.yoho_config.max_audio_len - 1,
                    )
                )
                content = f"<|t-{start}|>{content}<|t-{end}|><|voiceprint|>"
                transcript += content
            transcript += "<|endoftranscript|>"
            tokens = self.tokenizer.encode(transcript, allowed_special="all")
            audio_batch.append(audio)
            tokens_batch.append(tokens)

        audio_batch = np.array(audio_batch, dtype=np.float32)
        seq_lengths = np.array([len(s) for s in tokens_batch])
        tokens_batch = np.array(
            [np.pad(t, (0, self.yoho_config.max_text_len - len(t))) for t in tokens_batch],
            dtype=np.uint64,
        )
        return audio_batch, tokens_batch, seq_lengths


if __name__ == "__main__":
    import time
    import os

    from yoho.src.preprocessing.tokenizer import load_tokenizer

    tokenizer = load_tokenizer("./weights/vocab.txt", YOHOConfig())

    BATCH_SIZE = 16
    NUM_WORKERS = os.cpu_count()
    MODEL_DELAY = 2

    st = time.monotonic()
    dataloader = TranscriptionDataloader(
        YOHOConfig(),
        tokenizer,
        BATCH_SIZE,
        use_multiprocessing=True,
        shuffle=True,
        max_queued_batches=NUM_WORKERS,
        num_workers=NUM_WORKERS,
    )
    et = time.monotonic()
    print(f"{NUM_WORKERS} threads prepared {NUM_WORKERS} batches in {et-st:.02f} seconds")

    while True:
        st = time.monotonic()
        spectograms, tokens, lengths = dataloader.get_prepared_batch()
        et = time.monotonic()
        print(
            f"Batch was loaded in {et-st:.02f} seconds. Queue {dataloader.num_prepared_batches}/{dataloader.max_queued_batches}"
        )
        time.sleep(MODEL_DELAY)
