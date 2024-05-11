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
        kv_cache: Optional[dict] = None,
    ) -> jnp.ndarray:
        q = self.query_proj(q)

        if kv_cache is None or kv is None or "key" not in kv_cache:
            k = self.key_proj(q if kv is None else kv)
            v = self.value_proj(q if kv is None else kv)
        else:
            k = kv_cache["key"]
            v = kv_cache["value"]

        qk = q @ k.transpose((0, 2, 1))
        scale = (self.dims // self.n_head) ** -0.25
        qk = qk * scale

        if mask is not None:
            qk += mask

        attn_weights = jnp.softmax(qk, axis=-1)
        wv = attn_weights @ v

        return self.out_proj(wv)


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
    def __call__(
        self, q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, kv_cache: Optional[dict] = None
    ) -> jnp.ndarray:
        res = q
        q = nn.LayerNorm()(q)
        q = MultiHeadAttention(self.dims, self.n_heads)(q, q, mask=mask, kv_cache=kv_cache)
        q += res

        res = q
        x = nn.LayerNorm()(q)
        x = MultiHeadAttention(self.dims, self.n_heads)(q, kv, kv_cache=kv_cache)
        x += res
        res = x

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
        x = nn.Conv(self.dims, 3)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.dims, 3, strides=2)(x)
        x = nn.gelu(x)
        x = jnp.transpose(x, (0, 2, 1))
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
        pos = self.param(
            "positional_embedding",
            nn.initializers.glorot_uniform,
            (self.seq_len, self.dims),
        )
        q = nn.Embed(self.vocab_size, self.dims)(q)
        q += pos

        for _ in range(self.n_layers):
            q = DecoderBlock(self.dims, self.n_heads)(q, kv)

        x = nn.LayerNorm()(q)
        logits = x @ jnp.transpose(pos)

        return logits
