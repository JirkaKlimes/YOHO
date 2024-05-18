from bpeasy.tokenizer import train_bpe, _DEFAULT_REGEX_PATTERN
import pickle

from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.config import YOHOConfig

# TODO: read actual data
data = iter("some string")

vocab = train_bpe(
    data,
    # TODO: research this regex pretokenizer and find optimal split pattern
    python_regex=_DEFAULT_REGEX_PATTERN,
    max_token_length=64,
    vocab_size=32000,
)

mappings = sorted(vocab.items(), key=lambda p: p[1])
assert len(mappings) == mappings[-1][1] + 1, "number of tokens doesn't match the ids"

tokens = [m[0] for m in mappings]

VOCAB_PATH = "./weights/vocab.pkl"
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(tokens, f)


config = YOHOConfig()
tokenizer = load_tokenizer(VOCAB_PATH, config)

encoded = tokenizer.encode("Hello, World!")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
