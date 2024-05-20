import srt
import sentencepiece as spm
import io
from eld import LanguageDetector

from yoho.src.preprocessing.tokenizer import load_tokenizer

from train.utils.config import CONFIG
from train.utils.standardize_text import standardize_text


def load_transcripts():
    paths = [
        *CONFIG.dataset.noisy.joinpath("./transcripts").iterdir(),
        *CONFIG.dataset.clean.joinpath("./transcripts").iterdir(),
        *CONFIG.dataset.finetune.joinpath("./transcripts").iterdir(),
    ]

    for p in paths:
        with open(p) as f:
            data = f.read()
        utterances = [sub.content for sub in srt.parse(data)]
        lang = LanguageDetector().detect("\n".join(utterances)).language
        if lang not in CONFIG.language_whitelist:
            continue
        for utterance in utterances:
            yield standardize_text(utterance, lang)


special_tokens = [
    "<|startoftranscript|>",
    "<|endoftranscript|>",
    "<|voiceprint|>",
    *[f"<|t-{i}|>" for i in range(CONFIG.yoho.max_audio_len)],
]

model = io.BytesIO()
spm.SentencePieceTrainer.Train(
    sentence_iterator=load_transcripts(),
    model_writer=model,
    vocab_size=CONFIG.hyperparameters.tokenizer.vocab_size,
    user_defined_symbols=special_tokens,
)
with open(CONFIG.weights.tokenizer, "wb") as f:
    f.write(model.getvalue())


tokenizer = load_tokenizer(CONFIG.weights.tokenizer)

encoded = tokenizer.encode("Ahoj, světe!")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
