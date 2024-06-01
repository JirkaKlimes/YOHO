import srt
import sentencepiece as spm
import io
from eld import LanguageDetector

from yoho.src.preprocessing.tokenizer import load_tokenizer

from train.utils.config import SessionConfig
from train.utils.standardize_text import standardize_text


def load_transcripts(config: SessionConfig):
    paths = [
        *config.dataset.noisy.joinpath("./train", "./transcripts").iterdir(),
        *config.dataset.clean.joinpath("./train", "./transcripts").iterdir(),
        *config.dataset.finetune.joinpath("./train", "./transcripts").iterdir(),
        *config.dataset.noisy.joinpath("./val", "./transcripts").iterdir(),
        *config.dataset.clean.joinpath("./val", "./transcripts").iterdir(),
        *config.dataset.finetune.joinpath("./val", "./transcripts").iterdir(),
    ]

    for p in paths:
        with open(p) as f:
            data = f.read()
        utterances = [sub.content for sub in srt.parse(data)]
        lang = LanguageDetector().detect("\n".join(utterances)).language
        if lang not in config.language_whitelist:
            continue
        for utterance in utterances:
            yield standardize_text(utterance, lang)


def generate_special_tokens(config: SessionConfig):
    special_tokens = [
        "<|startoftranscript|>",
        "<|endoftranscript|>",
        "<|voiceprint|>",
        *[f"<|t-{i}|>" for i in range(config.yoho.max_audio_len)],
    ]
    return special_tokens


def train_model(config: SessionConfig):
    model = io.BytesIO()

    data = load_transcripts(config)
    special_tokens = generate_special_tokens(config)

    spm.SentencePieceTrainer.Train(
        sentence_iterator=data,
        model_writer=model,
        vocab_size=config.hyperparameters.tokenizer.vocab_size,
        user_defined_symbols=special_tokens,
    )
    with open(config.weights.tokenizer, "wb") as f:
        f.write(model.getvalue())


def main(config: SessionConfig):
    train_model(config)

    tokenizer = load_tokenizer(config.weights.tokenizer)

    encoded = tokenizer.encode("Ahoj, světe!")

    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
