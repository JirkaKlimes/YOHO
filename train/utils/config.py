from typing import List
from pydantic import BaseModel
from pathlib import Path

from yoho.src.config import YOHOConfig


class Dataset(BaseModel):
    noisy: Path
    clean: Path
    finetune: Path
    ambient: Path
    speakers: Path


class Tokenizer(BaseModel):
    max_token_length: int
    vocab_size: int


class Training(BaseModel):
    learning_rate: float
    batch_size: int
    accumulated_batches: int
    updates: int


class TrainingASR(Training):
    speechless_sample_ratio: float


class TrainingReconstruction(Training): ...


class TrainingVoicePrints(Training): ...


class Hyperparameters(BaseModel):
    tokenizer: Tokenizer
    transcribe_pretrain: TrainingASR
    reconstruct_pretrain: TrainingReconstruction
    voiceprint_finetune: TrainingVoicePrints
    transcribe_finetune: TrainingASR


class Weights(BaseModel):
    tokenizer: Path
    asr: Path
    voice_reconstruction: Path
    voiceprint: Path
    yoho: Path


class SessionConfig(BaseModel):
    name: str
    yoho: YOHOConfig
    dataset: Dataset
    hyperparameters: Hyperparameters
    weights: Weights
    language_whitelist: List[str]

    @property
    def path(self):
        return Path("./sessions").joinpath(self.name)
