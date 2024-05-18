import base64
from bpeasy.tokenizer import train_bpe, _DEFAULT_REGEX_PATTERN
import srt
from eld import LanguageDetector

from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.config import YOHOConfig
from train.utils.config import CONFIG
from train.utils.standardize_text import standardize


def load_transcripts():
    paths = [
        *CONFIG.dataset.noisy.joinpath("./transcripts").iterdir(),
        *CONFIG.dataset.clean.joinpath("./transcripts").iterdir(),
    ]

    for p in paths:
        with open(p) as f:
            data = f.read()
        utterances = [sub.content for sub in srt.parse(data)]
        lang = LanguageDetector().detect("\n".join(utterances)).language
        if lang not in CONFIG.language_whitelist:
            continue
        for utterance in utterances:
            yield standardize(utterance)


vocab = train_bpe(
    load_transcripts(),
    python_regex=_DEFAULT_REGEX_PATTERN,
    max_token_length=CONFIG.hyperparameters.tokenizer.max_token_length,
    vocab_size=CONFIG.hyperparameters.tokenizer.vocab_size,
)

mappings = sorted(vocab.items(), key=lambda p: p[1])
tokens = [base64.b64encode(m[0]).decode() + "\n" for m in mappings]

VOCAB_PATH = CONFIG.weights.vocab
with open(VOCAB_PATH, "w", encoding="ascii") as f:
    f.writelines(tokens)

config = YOHOConfig()
tokenizer = load_tokenizer(VOCAB_PATH, config)

encoded = tokenizer.encode("Ahoj, světe!")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
