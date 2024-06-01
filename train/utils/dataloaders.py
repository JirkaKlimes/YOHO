from typing import Any
import srt
from eld import LanguageDetector
import numpy as np
import bisect
import datetime as dt
import sentencepiece as spm
from librosa import load
from pathlib import Path

from train.utils.base_dataloader import Dataloader
from train.utils.standardize_text import standardize_text
from train.utils.config import SessionConfig


class TranscriptionDataloader(Dataloader):
    def __init__(
        self,
        config: SessionConfig,
        path: Path,
        tokenizer: spm.SentencePieceProcessor,
        batch_size: int,
        shuffle: bool = True,
        max_queued_batches: int = 8,
        num_workers: int = 4,
        warmup_queue: bool = True,
        use_multiprocessing: bool = True,
        disable_warnings: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.shuffle = shuffle

        language_detector = LanguageDetector()

        self.path = path
        all_paths = self.path.joinpath("transcripts").iterdir()
        sizes = []
        paths = []
        langs = []
        for path in all_paths:
            with open(path, encoding="utf-8") as f:
                data = f.read()
            transcript = list(srt.parse(data))
            content = "\n".join([utterance.content for utterance in transcript])
            lang = language_detector.detect(content).language
            if lang not in self.config.language_whitelist:
                continue
            sizes.append(len(transcript))
            paths.append((path, self.path.joinpath("audio", path.with_suffix(".mp3").name)))
            langs.append(lang)

        self.sizes = np.array(sizes, dtype=np.uint64).cumsum()
        self.paths = paths
        self.langs = langs
        self.index_table = np.arange(self.sizes[-1], dtype=np.uint64)
        if self.shuffle:
            np.random.shuffle(self.index_table)

        super().__init__(
            batch_size,
            max_queued_batches,
            num_workers,
            warmup_queue,
            use_multiprocessing,
            disable_warnings,
        )

    def on_epoch(self):
        if self.shuffle:
            np.random.shuffle(self.index_table)

    def randomize_padding(self, start_time, end_time, start_speech_time, end_speech_time):
        duration = (end_speech_time - start_speech_time).total_seconds()
        padding_left = (start_speech_time - start_time).total_seconds()
        padding_left = np.random.uniform(
            0, min(padding_left, self.config.yoho.max_input_seconds - duration)
        )
        start_time = start_speech_time - dt.timedelta(seconds=padding_left)
        duration = (end_speech_time - start_time).total_seconds()
        padding_right = (end_time - end_speech_time).total_seconds()
        padding_right = np.random.uniform(
            0, min(padding_right, self.config.yoho.max_input_seconds - duration)
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
                < self.config.yoho.max_input_seconds
            ):
                break
            si += 1
            utterances.append(transcript[si])
        end_speech_time = transcript[si].end
        end_time = (
            dt.timedelta(seconds=len(audio) / self.config.yoho.sample_rate)
            if si >= len(transcript) - 2
            else transcript[si + 1].start
        )

        start_time, end_time = self.randomize_padding(
            start_time, end_time, start_speech_time, end_speech_time
        )
        from_sample = int(np.ceil(start_time.total_seconds() * self.config.yoho.sample_rate))
        to_sample = int(np.floor(end_time.total_seconds() * self.config.yoho.sample_rate))
        audio = audio[from_sample:to_sample]

        if len(audio) > self.config.yoho.n_samples:
            return None, None

        audio = np.pad(audio, (0, self.config.yoho.n_samples - len(audio)))

        utterances_relative = [
            (
                int(
                    np.floor((ut.start - start_time).total_seconds() * self.config.yoho.sample_rate)
                ),
                int(np.ceil((ut.end - start_time).total_seconds() * self.config.yoho.sample_rate)),
                standardize_text(ut.content, lang=lang),
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
                audio = load(audio_path, sr=self.config.yoho.sample_rate)[0]

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
                        np.floor(start / self.config.yoho.stft_hop),
                        self.config.yoho.max_audio_len - 1,
                    )
                )
                end = int(
                    min(
                        np.floor(end / self.config.yoho.stft_hop),
                        self.config.yoho.max_audio_len - 1,
                    )
                )
                content = f"<|t-{start}|>{content}<|t-{end}|><|voiceprint|>"
                transcript += content
            transcript += "<|endoftranscript|>"
            tokens = self.tokenizer.encode(transcript)
            audio_batch.append(audio)
            tokens_batch.append(tokens)

        audio_batch = np.array(audio_batch)
        seq_lengths = np.array([len(s) for s in tokens_batch])
        tokens_batch = np.array(
            [
                t[: self.config.yoho.max_text_len]
                if len(t) > self.config.yoho.max_text_len
                else np.pad(t, (0, self.config.yoho.max_text_len - len(t)))
                for t in tokens_batch
            ],
            dtype=np.uint64,
        )
        loss_mask = np.zeros((self.batch_size, self.config.yoho.max_text_len), np.uint8)
        for i, (length, tokens) in enumerate(zip(seq_lengths, tokens_batch)):
            loss_mask[i, :length] = 1
            for j, tok in enumerate(tokens):
                if tok == self.tokenizer.encode("<|voiceprint|>")[0]:
                    loss_mask[i, j] = 0

        return audio_batch, tokens_batch, loss_mask
