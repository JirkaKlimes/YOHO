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

    model_dims = {
        "audio_dims": 512,
        "audio_seq_len": 1024,
        "audio_heads": 8,
        "audio_layers": 6,
        "text_vocab_size": 30522,
        "text_seq_len": 512,
        "text_dims": 512,
        "text_heads": 8,
        "text_layers": 6,
    }
    model = Whisper(**model_dims)
    dummy_mel = jnp.ones((1, model_dims["audio_seq_len"], model_dims["audio_dims"]))
    dummy_tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, model_dims["text_seq_len"]), 0, model_dims["text_vocab_size"]
    )
    variables = model.init(jax.random.key(0), dummy_mel, dummy_tokens)

    logits = model.apply(variables, dummy_mel, dummy_tokens)
    print("Logits shape:", logits.shape)
