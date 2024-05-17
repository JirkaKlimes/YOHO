from typing import Optional
import jax.numpy as jnp
import flax.linen as nn


class SwiGLU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        input_dim = x.shape[-1]

        w1 = nn.Dense(self.hidden_dim, use_bias=False)
        w2 = nn.Dense(self.hidden_dim, use_bias=False)
        w3 = nn.Dense(input_dim, use_bias=False)

        return w3(nn.silu(w1(x)) * w2(x))


class RoPE(nn.Module):
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
        # TODO: add kv cache
        batch_size, q_seq_len, in_dims = q.shape
        batch_size, kv_seq_len, in_dims = kv.shape

        q_heads = self.q_heads
        kv_heads = self.kv_heads or self.q_heads
        kv = q if kv is None else kv

        assert q_heads % kv_heads == 0, "num of kv heads must be divisible by num of q heads"
        assert q_heads >= kv_heads, "num of q heads must be greater or equal to num of kv heads"

        xq = nn.Dense(q_heads * self.dims, use_bias=False)(q)
        xk = nn.Dense(kv_heads * self.dims, use_bias=False)(kv)
        xv = nn.Dense(kv_heads * self.dims, use_bias=False)(kv)
        xk = xk.repeat(2, axis=-1)
        xv = xv.repeat(2, axis=-1)

        xq = xq.reshape(batch_size, q_seq_len, q_heads, self.dims)
        xk = xk.reshape(batch_size, kv_seq_len, q_heads, self.dims)
        xv = xv.reshape(batch_size, kv_seq_len, q_heads, self.dims)
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        xq = RoPE()(xq)
        xk = RoPE()(xk)

        xk = xk.transpose(0, 1, 3, 2)
        attention_scores = (xq @ xk) * (self.dims / q_heads) ** -0.5
        if mask is not None:
            attention_scores -= 1 / mask - 1
        attention_scores = nn.softmax(attention_scores)
        out = (attention_scores @ xv).transpose(0, 2, 1, 3).reshape(batch_size, q_seq_len, -1)
        out = nn.Dense(in_dims, use_bias=False)(out)

        return out


class EncoderBlock(nn.Module):
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
        x = SwiGLU(self.dims * 2)(x)
        x += res

        return x


class DecoderBlock(nn.Module):
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
        x = SwiGLU(self.dims * 2)(x)
        x += res

        return x
