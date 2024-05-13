import base64
import itertools
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

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


def get_whisper():
    from yoho.train.convert_weights import parse_weights

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

    dummy_mel = jnp.ones((1, 2 * AUDIO_SEQ_LEN, N_MELS))
    dummy_tokens = jnp.ones((1, TEXT_SEQ_LEN), dtype=jnp.uint32)
    # print(model.tabulate(jax.random.key(0), dummy_mel, dummy_tokens))
    variables = model.init(jax.random.key(0), dummy_mel, dummy_tokens)

    parsed_weights = parse_weights("./weights/model_base_multi.safetensors")
    flat_params = flatten_dict(variables["params"])
    for name in flat_params:
        flat_params[name] = parsed_weights[name]
    params = unflatten_dict(flat_params)
    variables["params"] = params

    return model, variables


if __name__ == "__main__":
    import jax
    from flax.traverse_util import flatten_dict, unflatten_dict

    from yoho.train.convert_weights import parse_weights

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

    dummy_mel = jnp.ones((1, 2 * AUDIO_SEQ_LEN, N_MELS))
    dummy_tokens = jnp.ones((1, TEXT_SEQ_LEN), dtype=jnp.uint32)
    # print(model.tabulate(jax.random.key(0), dummy_mel, dummy_tokens))
    variables = model.init(jax.random.key(0), dummy_mel, dummy_tokens)

    parsed_weights = parse_weights("./weights/model_base_multi.safetensors")
    flat_params = flatten_dict(variables["params"])
    for name in flat_params:
        flat_params[name] = parsed_weights[name]
    params = unflatten_dict(flat_params)
    variables["params"] = params

    with open("model.bin", "wb") as f:
        pickle.dump((variables, model), f)

    print("weights loaded")
    # out = model.apply(variables, dummy_mel, dummy_tokens)

    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

    from original_whisper.whisper.audio import log_mel_spectrogram

    mel = log_mel_spectrogram(
        "/home/jirka/Downloads/ted_interview.mp3", N_MELS, padding=N_SAMPLES, device="cpu"
    )
    mel = jnp.asarray(mel, jnp.float32)[None, :, :3000].transpose((0, 2, 1))
    encoded_audio = model.apply(variables, mel, method=Whisper.encode_audio)
    print(encoded_audio)

    enc = get_encoding()

    lst = [
        enc._special_tokens["<|startoftranscript|>"],
        enc._special_tokens["<|en|>"],
        enc._special_tokens["<|transcribe|>"],
    ]

    while True:
        out = model.apply(variables, encoded_audio, jnp.array([lst]), method=Whisper.decode_text)
        print(out.shape)
        out = jnp.argmax(out[0], axis=-1)
        print("model out:", enc.decode(out))
        lst.append(out[-1])
        dec = enc.decode(lst)
        print(dec)
        if dec.endswith("<|endoftext|>"):
            quit()
