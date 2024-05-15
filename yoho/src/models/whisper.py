import jax.numpy as jnp
import flax.linen as nn

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


if __name__ == "__main__":
    import jax
    from pathlib import Path
    import pickle
    import numpy as np

    from yoho.src.preprocessing.audio import load_audio, mel_spectogram, normalize_spectogram
    from yoho.src.tokenizer import get_tokenizer

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

    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE

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

    audio = load_audio(Path("/home/jirka/Downloads/ted_interview.mp3"), SAMPLE_RATE)

    if audio.shape[0] > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    audio = np.pad(audio, (0, N_SAMPLES - audio.shape[0]))

    spectogram = mel_spectogram(audio, N_FFT, HOP_LENGTH, SAMPLE_RATE, N_MELS)
    norm_spectogram = normalize_spectogram(spectogram)

    encoded_audio = model.apply(variables, norm_spectogram[None], method=Whisper.encode_audio)

    tokenizer = get_tokenizer()

    @jax.jit
    def fn(mel, tok):
        out = model.apply(variables, mel, tok, method=Whisper.decode_text)
        return out

    decoded_text = np.zeros(TEXT_SEQ_LEN, dtype=np.uint64)
    decoded_text[0] = tokenizer._special_tokens["<|startoftranscript|>"]
    decoded_text[2] = tokenizer._special_tokens["<|transcribe|>"]
    decoded_text[3] = tokenizer._special_tokens["<|notimestamps|>"]

    pos = 1

    while True:
        if pos in [2, 3]:
            pos += 1
            continue
        out = fn(encoded_audio, jnp.array([decoded_text]))[0, pos - 1]
        idx = int(jnp.argmax(out, axis=-1))
        decoded_text[pos] = idx
        pos += 1
        dec = tokenizer.decode(decoded_text[:pos])
        print(dec)
        if idx == tokenizer._special_tokens["<|endoftext|>"]:
            break
