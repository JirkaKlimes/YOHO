import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import mel_spectogram

from train.utils.config import CONFIG
from train.utils.dataloaders import TranscriptionDataloader


if __name__ == "__main__":
    import sounddevice  # noqa: F401

    HYPERPARAMETERS = CONFIG.hyperparameters.transcribe_pretrain

    tokenizer = load_tokenizer(CONFIG.weights.tokenizer, CONFIG.yoho)

    model = Model(CONFIG.yoho, len(tokenizer.vocab) + len(tokenizer.special_tokens))

    @jax.jit
    @jax.vmap
    def batched_mel_spectogram(audio):
        spectogram = mel_spectogram(
            audio,
            CONFIG.yoho.n_fft,
            CONFIG.yoho.stft_hop,
            CONFIG.yoho.sample_rate,
            CONFIG.yoho.n_mel_bands,
            htk=True,
        )
        return spectogram

    dataloader = TranscriptionDataloader(
        CONFIG.yoho,
        tokenizer,
        HYPERPARAMETERS.batch_size,
        shuffle=True,
        max_queued_batches=8,
        num_workers=8,
        disable_warnings=True,
    )

    # for audio_track, toks in zip(audio, tokens):
    #     print(tokenizer.decode(toks))
    #     sounddevice.play(audio_track, config.sample_rate, blocking=True)

    def get_batch():
        audio, tokens, loss_mask = dataloader.get_prepared_batch()
        spectogram = batched_mel_spectogram(audio / 32768.0)
        spectogram = jnp.astype(spectogram, jnp.float32)
        tokens = jnp.astype(tokens, jnp.uint32)
        loss_mask = jnp.astype(loss_mask, jnp.uint8)
        return spectogram, tokens, loss_mask

    dummy_tokens = jnp.empty(
        (HYPERPARAMETERS.batch_size, CONFIG.yoho.max_text_len), dtype=jnp.uint32
    )
    dummy_spectogram = jnp.empty(
        (HYPERPARAMETERS.batch_size, CONFIG.yoho.max_audio_len, CONFIG.yoho.n_mel_bands),
        dtype=jnp.uint32,
    )
    variables = model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_spectogram)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adamw(HYPERPARAMETERS.learning_rate),
    )

    def loss_fn(params, spectogram, tokens, loss_mask):
        logits = state.apply_fn({"params": params}, tokens, spectogram)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:])
        loss *= loss_mask[:, :-1]
        loss = jnp.mean(loss)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(state, batch):
        loss, grads = grad_fn(state.params, *batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    while True:
        batch = get_batch()
        state, loss = train_step(state, batch)
        print(f"Loss: {loss}")
