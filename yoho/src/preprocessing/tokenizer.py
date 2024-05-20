import sentencepiece as spm
from pathlib import Path


def load_tokenizer(vocab_path: Path):
    tokenizer = spm.SentencePieceProcessor(model_file=str(vocab_path))
    return tokenizer
