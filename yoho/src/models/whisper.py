import base64
import itertools
import pickle
from subprocess import run
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from yoho.src.models.layers import AudioEncoder, TextDecoder


class Whisper(nn.Module):
    audio_dims: int
    audio_seq_len: int
    audio_heads: int
    audio_layers: int
    text_vocab_size: int
    text_seq_len: int
    text_dims: int
    text_heads: int
    text_layers: int

    def setup(self):
        self.audio_encoder = AudioEncoder(
            seq_len=self.audio_seq_len,
            dims=self.audio_dims,
            n_heads=self.audio_heads,
            n_layers=self.audio_layers,
        )
        self.text_decoder = TextDecoder(
            vocab_size=self.text_vocab_size,
            seq_len=self.text_seq_len,
            dims=self.text_dims,
            n_heads=self.text_heads,
            n_layers=self.text_layers,
        )

    def encode_audio(self, mel: jnp.ndarray) -> jnp.ndarray:
        return self.audio_encoder(mel)

    def decode_text(self, encoded_audio: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        return self.text_decoder(tokens, encoded_audio)

    def __call__(self, mel: jnp.ndarray, tokens: jnp.ndarray):
        audio_features = self.audio_encoder(mel)
        logits = self.text_decoder(tokens, audio_features)
        return logits


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


def get_encoding():
    with open("./weights/multilingual.tiktoken") as f:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in f if line)
        }
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = dict(zip(specials, itertools.count(n_vocab)))
    n_vocab += len(specials)
    import tiktoken

    return tiktoken.Encoding(
        name="multilingual",
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


if __name__ == "__main__":
    import jax

    from yoho.src.preprocessing.mel_spectogram import generate_mel_filters

    N_MELS = 80
    N_VOCAB = 51865
    AUDIO_SEQ_LEN = 1500
    AUDIO_DIMS = 512
    AUDIO_HEADS = 8
    AUDIO_LAYERS = 6
    TEXT_SEQ_LEN = 448
    TEXT_DIMS = 512
    TEXT_HEADS = 8
    TEXT_LAYERS = 6

    model = Whisper(
        audio_dims=AUDIO_DIMS,
        audio_seq_len=AUDIO_SEQ_LEN,
        audio_heads=AUDIO_HEADS,
        audio_layers=AUDIO_LAYERS,
        text_vocab_size=N_VOCAB,
        text_seq_len=TEXT_SEQ_LEN,
        text_dims=TEXT_DIMS,
        text_heads=TEXT_HEADS,
        text_layers=TEXT_LAYERS,
    )

    with open("./weights/params.bin", "rb") as f:
        params = pickle.load(f)

    variables = {"params": params}

    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "/home/jirka/Downloads/ted_interview.mp3",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    out = run(cmd, capture_output=True, check=True).stdout
    buffer = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    buffer = np.pad(buffer, (0, N_SAMPLES - buffer.shape[0]))
    stft = jax.scipy.signal.stft(buffer, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)[-1]
    magnitudes = jnp.abs(stft)[..., :-1] ** 2
    filters = generate_mel_filters(SAMPLE_RATE, N_FFT, N_MELS)
    mel_spec = filters @ magnitudes
    log_spec = jnp.log10(jnp.clip(mel_spec, a_min=1e-10))
    log_spec = jnp.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    mel = log_spec.T[None]
    encoded_audio = model.apply(variables, mel, method=Whisper.encode_audio)

    enc = get_encoding()

    @jax.jit
    def fn(mel, tok):
        out = model.apply(variables, mel, tok, method=Whisper.decode_text)
        return out

    decoded_text = np.zeros(TEXT_SEQ_LEN, dtype=np.uint64)
    decoded_text[0] = enc._special_tokens["<|startoftranscript|>"]
    decoded_text[2] = enc._special_tokens["<|transcribe|>"]

    pos = 1

    while True:
        if pos == 2:
            pos += 1
            continue
        out = fn(encoded_audio, jnp.array([decoded_text]))[0, pos - 1]
        idx = int(jnp.argmax(out.at[enc._special_tokens["<|notimestamps|>"]].set(0), axis=-1))
        decoded_text[pos] = idx
        pos += 1
        dec = enc.decode(decoded_text[:pos])
        print(dec)
        if idx == enc._special_tokens["<|endoftext|>"]:
            break
