import base64
from bpeasy.tokenizer import train_bpe, _DEFAULT_REGEX_PATTERN

from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.config import YOHOConfig
from train.utils.config import CONFIG

# TODO: read actual data
data = iter(["Hello, World!"])

vocab = train_bpe(
    data,
    python_regex=_DEFAULT_REGEX_PATTERN,
    max_token_length=CONFIG.hyperparameters.tokenizer.max_token_length,
    vocab_size=CONFIG.hyperparameters.tokenizer.vocab_size,
)

mappings = sorted(vocab.items(), key=lambda p: p[1])
assert len(mappings) == mappings[-1][1] + 1, "number of tokens doesn't match the ids"

tokens = [base64.b64encode(m[0]).decode() + "\n" for m in mappings]

VOCAB_PATH = CONFIG.weights.vocab
with open(VOCAB_PATH, "w", encoding="ascii") as f:
    f.writelines(tokens)

config = YOHOConfig()
tokenizer = load_tokenizer(VOCAB_PATH, config)

encoded = tokenizer.encode("Hello, World!")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
