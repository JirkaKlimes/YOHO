import flax.linen as nn
import jax.numpy as jnp
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
