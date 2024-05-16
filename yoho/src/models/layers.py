from typing import Optional
import jax.numpy as jnp
import flax.linen as nn


class SwiGLU(nn.Module):
    hidden_dim_factor: float = 3.0
    hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        hidden_dim = self.hidden_dim or int(input_dim * self.hidden_dim_factor)

        w1 = nn.Dense(hidden_dim, use_bias=False)
        w2 = nn.Dense(hidden_dim, use_bias=False)
        w3 = nn.Dense(input_dim, use_bias=False)

        return w3(nn.silu(w1(x)) * w2(x))


class RoPE(nn.Module):
    @nn.compact
    def __call__(self, x):
        _, seq_len, dim = x.shape
        assert not dim & 1, "number of input dims mut be even"

        theta = jnp.arange(0, dim, 2)
        inv_freq = 1.0 / (10000 ** (theta / dim))
        freqs = jnp.outer(jnp.arange(seq_len), inv_freq)
        freq_complex = jnp.e ** (1j * freqs)

        x_complex = x[:, :, ::2] + 1j * x[:, :, 1::2]

        x_rotated = x_complex * freq_complex[None]

        x_out = jnp.empty_like(x)
        x_out = x_out.at[:, :, ::2].set(x_rotated.real)
        x_out = x_out.at[:, :, 1::2].set(x_rotated.imag)

        return x_out
