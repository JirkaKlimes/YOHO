from bpeasy.tokenizer import BPEasyTokenizer
from pathlib import Path
import pickle

from yoho.src.config import YOHOConfig


def load_tokenizer(vocab_path: Path, yoho_config: YOHOConfig):
    with open(vocab_path, "rb") as f:
        tokens = pickle.load(f)

    vocab = {t: i for i, t in enumerate(tokens)}

    special_tokens = [
        "<|startoftranscript|>",
        "<|resumeoftranscript|>",
        "<|nospeech|>",
        *[f"<|{i}|>" for i in range(yoho_config.max_audio_len)],
    ]

    tokenizer = BPEasyTokenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=True,
    )

    return tokenizer
