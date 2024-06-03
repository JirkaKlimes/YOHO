import jax
import jax.numpy as jnp
import flax.linen as nn

from yoho.src.nn.layers import EncoderBlock, DecoderBlock, SinPositionalEncoding
from yoho.src.config import YOHOConfig


class AudioEncoder(nn.Module):
    q_heads: int
    kv_heads: int
    dims: int
    seq_len: int
    blocks: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(self.dims, 3)(x)
        x = nn.RMSNorm()(x)
        x = nn.silu(x)
        x = nn.max_pool(x, (1, 2), (1, 2))
        x = nn.Conv(self.dims, 3)(x)
        x = nn.RMSNorm()(x)
        x = nn.silu(x)
        x = SinPositionalEncoding(self.seq_len, self.dims)(x)

        for _ in range(self.blocks):
            x = EncoderBlock(self.q_heads, self.kv_heads, self.dims)(x)

        return x


class TextDecoder(nn.Module):
    q_heads: int
    kv_heads: int
    dims: int
    seq_len: int
    blocks: int
    vocab_size: int

    @nn.compact
    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray) -> jnp.ndarray:
        embed = nn.Embed(self.vocab_size, self.dims)
        q = embed(q)
        seq_len = q.shape[1]

        pos = self.param(
            "positional_embedding",
            nn.initializers.glorot_uniform(),
            (self.seq_len, self.dims),
        )
        q += pos[:seq_len]
        mask = jnp.triu(jnp.full((seq_len, seq_len), 1)).T

        for _ in range(self.blocks):
            q = DecoderBlock(self.q_heads, self.kv_heads, self.dims)(q, kv, mask=mask)

        x = nn.RMSNorm()(q)
        logits = x @ embed.embedding.T
        return logits


class Model(nn.Module):
    config: YOHOConfig
    vocab_size: int

    def setup(self):
        self.encoder = AudioEncoder(
            self.config.n_audio_heads,
            self.config.n_audio_heads // 2,
            self.config.dims,
            self.config.max_audio_len,
            self.config.n_audio_blocks,
        )
        self.decoder = TextDecoder(
            self.config.n_text_heads,
            self.config.n_text_heads // 2,
            self.config.dims,
            self.config.max_text_len,
            self.config.n_text_blocks,
            self.vocab_size,
        )

    def __call__(self, text: jnp.ndarray, audio: jnp.ndarray):
        audio_features = self.encode_audio(audio)
        decoded_text = self.decode_text(text, audio_features)
        return decoded_text

    def encode_audio(self, audio: jnp.ndarray):
        return self.encoder(audio)

    def decode_text(self, text: jnp.ndarray, audio: jnp.ndarray):
        return self.decoder(text, audio)


if __name__ == "__main__":
    config = YOHOConfig()
    vocab_size = 32000

    model = Model(config, vocab_size)

    dummy_audio = jnp.ones((1, config.max_audio_len, config.n_mel_bands), jnp.float32)
    dummy_text = jnp.ones((1, config.max_text_len), jnp.uint32)

    print(model.tabulate(jax.random.PRNGKey(0), dummy_text, dummy_audio))
