import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class SinPositionalEncoding(nn.Module):
    length: int
    dim: int
    max_timescale: int = 10000

    def sin_positional_encoding(self) -> jnp.ndarray:
        """Returns sinusoids for positional embedding"""
        assert self.dim & 1 == 0, "Number of dimensions must be even"
        log_timescale_increment = jnp.log(self.max_timescale) / (self.dim // 2 - 1)
        inv_timescales = jnp.e ** (-log_timescale_increment * jnp.arange(self.dim // 2))
        scaled_time = jnp.arange(self.length)[:, None] * inv_timescales[None, :]
        return jnp.concat([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)

    def setup(self):
        self.encoding_matrix = self.sin_positional_encoding()

    def __call__(self, x):
        return x + self.encoding_matrix


class MultiHeadAttention(nn.Module):
    # TODO: add KV cache
    dims: int
    n_head: int

    def setup(self):
        self.query_proj = nn.Dense(self.dims)
        self.key_proj = nn.Dense(self.dims, use_bias=False)
        self.value_proj = nn.Dense(self.dims)
        self.out_proj = nn.Dense(self.dims)

    def __call__(
        self,
        q: jnp.ndarray,
        kv: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        q = self.query_proj(q)

        k = self.key_proj(q if kv is None else kv)
        v = self.value_proj(q if kv is None else kv)

        wv = self.qkv_attention(q, k, v, mask=mask)

        return self.out_proj(wv)

    def qkv_attention(self, q, k, v, mask=None):
        batch_size, seq_len, dims = q.shape
        scale = (dims // self.n_head) ** -0.25

        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk += mask[:seq_len, :seq_len]

        w = nn.softmax(qk)
        output = (w @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return output


class EncoderBlock(nn.Module):
    dims: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(self.dims, self.n_heads)(x, x)
        x += res

        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dims * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dims)(x)
        x += res

        return x


class DecoderBlock(nn.Module):
    dims: int
    n_heads: int

    @nn.compact
    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        res = q
        q = nn.LayerNorm()(q)
        q = MultiHeadAttention(self.dims, self.n_heads)(q, q, mask=mask)
        q += res

        res = q
        q = nn.LayerNorm()(q)
        x = MultiHeadAttention(self.dims, self.n_heads)(q, kv)
        x += res

        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dims * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dims)(x)
        x += res

        return x


class AudioEncoder(nn.Module):
    seq_len: int
    dims: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.dims, 3, padding=1, dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.dims, 3, padding=1, strides=2, dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = SinPositionalEncoding(self.seq_len, self.dims)(x)

        for _ in range(self.n_layers):
            x = EncoderBlock(self.dims, self.n_heads)(x)

        x = nn.LayerNorm()(x)
        return x


class TextDecoder(nn.Module):
    vocab_size: int
    seq_len: int
    dims: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray) -> jnp.ndarray:
        seq_len = q.shape[1]
        pos = self.param(
            "positional_embedding",
            nn.initializers.glorot_uniform(),
            (self.seq_len, self.dims),
        )
        embed_layer = nn.Embed(self.vocab_size, self.dims)
        q = embed_layer(q)
        q += pos[:seq_len]

        mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf), 1)

        for _ in range(self.n_layers):
            q = DecoderBlock(self.dims, self.n_heads)(q, kv, mask=mask)

        x = nn.LayerNorm()(q)
        logits = x @ embed_layer.embedding.T

        return logits


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
