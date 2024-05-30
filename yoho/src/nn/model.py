import jax
import jax.numpy as jnp
import flax.linen as nn

from yoho.src.nn.layers import EncoderBlock, DecoderBlock
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

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.dims)
        self.decoders = [
            DecoderBlock(self.q_heads, self.kv_heads, self.dims) for _ in range(self.blocks)
        ]
        self.norm = nn.RMSNorm()

    def emebed_tokens(self, tokens: jnp.ndarray):
        return self.embed(tokens)

    def decode_embeddings(self, embeddings: jnp.ndarray, audio: jnp.ndarray):
        seq_len = embeddings.shape[1]
        mask = jnp.triu(jnp.full((seq_len, seq_len), 1)).T

        for decoder in self.decoders:
            embeddings = decoder(embeddings, audio, mask=mask)

        embeddings = self.norm(embeddings)
        logits = embeddings @ self.embed.embedding.T
        return logits

    def __call__(self, embeddings: jnp.ndarray, audio: jnp.ndarray) -> jnp.ndarray:
        embeddings = self.emebed_tokens(embeddings)
        logits = self.decode_embeddings(embeddings, audio)
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

    def __call__(self, tokens: jnp.ndarray, audio: jnp.ndarray):
        audio_features = self.encode_audio(audio)
        embeddings = self.embedd_tokens(tokens)
        decoded_text = self.decode_text(embeddings, audio_features)
        return decoded_text

    def encode_audio(self, audio: jnp.ndarray):
        return self.encoder(audio)

    def embedd_tokens(self, tokens: jnp.ndarray):
        return self.decoder.emebed_tokens(tokens)

    def decode_text(self, embeddings: jnp.ndarray, audio: jnp.ndarray):
        return self.decoder.decode_embeddings(embeddings, audio)


if __name__ == "__main__":
    config = YOHOConfig()
    vocab_size = 32000

    model = Model(config, vocab_size)

    dummy_audio = jnp.ones((1, config.max_audio_len, config.n_mel_bands), jnp.float32)
    dummy_text = jnp.ones((1, config.max_text_len), jnp.uint32)

    print(model.tabulate(jax.random.PRNGKey(0), dummy_text, dummy_audio))
