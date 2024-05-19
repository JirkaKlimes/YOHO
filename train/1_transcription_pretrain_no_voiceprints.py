import time
import jax
import jax.numpy as jnp

from yoho.src.config import YOHOConfig
from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import mel_spectogram

from train.utils.config import CONFIG
from train.utils.dataloaders import TranscriptionDataloader


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    HYPERPARAMETERS = CONFIG.hyperparameters.transcribe_pretrain

    config = YOHOConfig()
    tokenizer = load_tokenizer(CONFIG.weights.vocab, config)

    model = Model(config, len(tokenizer.vocab))

    @jax.jit
    @jax.vmap
    def mel_spectogram_partial(audio):
        spectogram = mel_spectogram(
            audio,
            config.n_fft,
            config.stft_hop,
            config.sample_rate,
            config.n_mel_bands,
            htk=True,
        )
        return spectogram

    dataloader = TranscriptionDataloader(
        config,
        tokenizer,
        HYPERPARAMETERS.batch_size,
        shuffle=False,
        max_queued_batches=1,
        num_workers=1,
    )

    audio, tokens, lengths = dataloader.get_prepared_batch()
    spectogram = mel_spectogram_partial(audio)
    spectogram = jnp.astype(spectogram, jnp.float32)
    tokens = jnp.astype(tokens, jnp.uint32)

    variables = model.init(jax.random.PRNGKey(0), tokens, spectogram)

    @jax.jit
    def apply_fn(text, audio):
        return model.apply(variables, text, audio)

    for _ in range(4):
        audio, tokens, lengths = dataloader.get_prepared_batch()
        spectogram = mel_spectogram_partial(audio)
        spectogram = jnp.astype(spectogram, jnp.float32)
        tokens = jnp.astype(tokens, jnp.uint32)

        st = time.monotonic()
        out = apply_fn(tokens, spectogram)
        out = out.max().max()
        print(out)
        et = time.monotonic()
        print(f"Inference took: {et-st:.02f} seconds")
