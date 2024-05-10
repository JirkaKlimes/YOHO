import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sin_positional_embedding(length, dim, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert dim & 1 == 0
    log_timescale_increment = jnp.log(max_timescale) / (dim // 2 - 1)
    inv_timescales = jnp.e ** (-log_timescale_increment * jnp.arange(dim // 2))
    scaled_time = jnp.arange(length)[:, None] * inv_timescales[None, :]
    return jnp.concat([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)


class ResidualAttentionBlock(nn.Module): ...


class AudioEncoder(nn.Module): ...


class TextDecoder(nn.Module): ...


class Whisper(nn.Module): ...
