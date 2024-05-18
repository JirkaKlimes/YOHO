import base64
from bpeasy.tokenizer import BPEasyTokenizer
from pathlib import Path

from yoho.src.config import YOHOConfig


def load_tokenizer(vocab_path: Path, yoho_config: YOHOConfig):
    with open(vocab_path, encoding="ascii") as f:
        tokens = [base64.b64decode(t.rstrip()) for t in f.readlines()]

    vocab = {t: i for i, t in enumerate(tokens)}

    special_tokens = [
        "<|startoftranscript|>",
        "<|endoftranscript|>",
        "<|voiceprint|>",
        *[f"<|t-{i}|>" for i in range(yoho_config.max_audio_len)],
    ]

    tokenizer = BPEasyTokenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=True,
    )

    return tokenizer
