from typing import Optional
import jax.numpy as jnp
import flax.linen as nn
from einops import einsum, reduce, rearrange


class SwiGLU(nn.Module):
    """https://arxiv.org/pdf/2002.05202"""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        input_dim = x.shape[-1]

        w1 = nn.Dense(self.hidden_dim, use_bias=False)
        w2 = nn.Dense(self.hidden_dim, use_bias=False)
        w3 = nn.Dense(input_dim, use_bias=False)

        return w3(nn.silu(w1(x)) * w2(x))


class RoPE(nn.Module):
    """https://arxiv.org/pdf/2104.09864"""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        seq_len, dim = x.shape[-2:]
        assert not dim & 1, "number of input dims mut be even"

        thetas = 10 ** -jnp.arange(0, dim // 2, dtype=jnp.float32)
        angles = jnp.outer(jnp.arange(1, seq_len + 1), thetas)
        angles = jnp.e ** (1j * angles)

        x_complex = x[..., ::2] + 1j * x[..., 1::2]
        x_rotated = x_complex * angles

        x_out = jnp.empty_like(x)
        x_out = x_out.at[..., ::2].set(x_rotated.real)
        x_out = x_out.at[..., 1::2].set(x_rotated.imag)

        return x_out


class GroupedQueryAttention(nn.Module):
    """https://arxiv.org/pdf/2305.13245"""

    q_heads: int
    dims: int
    kv_heads: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        kv: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ):
        kv_heads = self.kv_heads or self.q_heads
        kv = q if kv is None else kv

        assert self.q_heads >= kv_heads, "Number of Q heads must be greater or equal to number of KV heads."
        assert self.q_heads % kv_heads == 0, "Number of KV heads must be divisible by number of Q heads."
        assert self.dims % self.q_heads == 0, "Number of dimensions must be divisible by number of Q heads."

        in_dim = q.shape[-1]
        head_dim = self.dims // self.q_heads
        groups = self.q_heads // kv_heads

        q = nn.DenseGeneral((self.q_heads, head_dim), use_bias=False)(q)
        k = nn.DenseGeneral((kv_heads, head_dim), use_bias=False)(kv)
        v = nn.DenseGeneral((kv_heads, head_dim), use_bias=False)(kv)

        q = rearrange(q, "... s (h g) d -> ... g h s d", g=groups)
        k = rearrange(k, "... s h d -> ... h s d")
        v = rearrange(v, "... s h d -> ... h s d")

        rope = RoPE()
        q = rope(q)
        k = rope(k)

        scores = einsum(q, k, "... g h s d, ... h a d -> ... h s a")

        if mask is not None:
            mask = rearrange(mask, "s a -> 1 1 s a")
            scores -= 1 / mask - 1

        scale = head_dim**0.5
        attention = nn.softmax(scores / scale)

        out = einsum(attention, v, "... h s a, ... h a d -> ... h s d")
        out = rearrange(out, "... h s d -> ... s (h d)")
        out = nn.Dense(in_dim, use_bias=False)(out)
        return out


class EncoderBlock(nn.Module):
    """https://arxiv.org/pdf/1706.03762"""

    q_heads: int
    kv_heads: int
    dims: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.RMSNorm()(x)
        x = GroupedQueryAttention(self.q_heads, self.dims, self.kv_heads)(x, x)
        x += res

        res = x
        x = nn.RMSNorm()(x)
        x = SwiGLU(int(self.dims * 3))(x)
        x += res

        return x


class DecoderBlock(nn.Module):
    """https://arxiv.org/pdf/1706.03762"""

    q_heads: int
    kv_heads: int
    dims: int

    @nn.compact
    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        res = q
        q = nn.RMSNorm()(q)
        q = GroupedQueryAttention(self.q_heads, self.dims, self.kv_heads)(q, q, mask=mask)
        q += res

        res = q
        q = nn.RMSNorm()(q)
        kv = nn.RMSNorm()(kv)
        x = GroupedQueryAttention(self.q_heads, self.dims, self.kv_heads)(q, kv)
        x += res

        res = x
        x = nn.RMSNorm()(x)
        x = SwiGLU(int(self.dims * 1.5))(x)
        x += res

        return x
