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

    def __call__(self, mel: jnp.ndarray, tokens: jnp.ndarray):
        audio_features = self.audio_encoder(mel)
        logits = self.text_decoder(tokens, audio_features)
        return logits


if __name__ == "__main__":
    import jax
    from flax.traverse_util import flatten_dict, unflatten_dict

    from yoho.train.convert_weights import parse_weights

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

    dummy_mel = jnp.ones((1, 2 * AUDIO_SEQ_LEN, N_MELS))
    dummy_tokens = jnp.ones((1, TEXT_SEQ_LEN), dtype=jnp.uint32)
    print(model.tabulate(jax.random.key(0), dummy_mel, dummy_tokens))
    variables = model.init(jax.random.key(0), dummy_mel, dummy_tokens)

    parsed_weights = parse_weights("./weights/model_base_multi.safetensors")
    flat_params = flatten_dict(variables["params"])
    for name in flat_params:
        flat_params[name] = parsed_weights[name]
    params = unflatten_dict(flat_params)
    variables["params"] = params

    out = model.apply(variables, dummy_mel, dummy_tokens)

    print(out)
    print(out.shape)
