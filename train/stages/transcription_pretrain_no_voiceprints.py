import pickle
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import mel_spectogram, normalize_spectogram

from train.utils.dataloaders import TranscriptionDataloader

# TODO: save loss progress
# TODO: reject very wrong samples, save to bitmap image for visualization
# TODO: write webui and launch server to monitor training progress
# TODO: generate validation samples every few steps


def main(config):
    HYPERPARAMETERS = config.hyperparameters.transcribe_pretrain

    tokenizer = load_tokenizer(config.weights.tokenizer)

    model = Model(config.yoho, tokenizer.vocab_size())

    @jax.jit
    @jax.vmap
    def batched_mel_spectogram(audio):
        spectogram = mel_spectogram(
            audio,
            config.yoho.n_fft,
            config.yoho.stft_hop,
            config.yoho.sample_rate,
            config.yoho.n_mel_bands,
        )
        return spectogram

    dataloader = TranscriptionDataloader(
        config,
        tokenizer,
        HYPERPARAMETERS.batch_size,
        shuffle=True,
        max_queued_batches=8,
        num_workers=8,
        disable_warnings=True,
    )

    def get_batch():
        audio, tokens, loss_mask = dataloader.get_prepared_batch()
        spectogram = batched_mel_spectogram(audio / 32768.0)
        spectogram = jnp.astype(spectogram, jnp.float32)
        spectogram = normalize_spectogram(spectogram)
        tokens = jnp.astype(tokens, jnp.uint32)
        loss_mask = jnp.astype(loss_mask, jnp.uint8)
        return spectogram, tokens, loss_mask

    if config.weights.asr.exists():
        with open(config.weights.asr, "rb") as f:
            params = pickle.load(f)
        variables = {"params": params}

    else:
        dummy_tokens = jnp.empty(
            (HYPERPARAMETERS.batch_size, config.yoho.max_text_len), dtype=jnp.uint32
        )
        dummy_spectogram = jnp.empty(
            (HYPERPARAMETERS.batch_size, config.yoho.max_audio_len, config.yoho.n_mel_bands),
            dtype=jnp.uint32,
        )
        # print(model.tabulate(jax.random.key(0), dummy_tokens, dummy_spectogram))
        variables = model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_spectogram)

        with open(config.weights.asr, "wb") as f:
            pickle.dump(variables["params"], f)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.MultiSteps(
            optax.adamw(HYPERPARAMETERS.learning_rate),
            HYPERPARAMETERS.accumulated_batches,
        ),
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

    pbar = tqdm(range(HYPERPARAMETERS.updates))
    for update in pbar:
        losses = []
        for accumulation_step in range(HYPERPARAMETERS.accumulated_batches):
            batch = get_batch()
            state, loss = train_step(state, batch)
            losses.append(loss)
        pbar.set_description_str(f"Loss: {sum(losses) / len(losses):.05f}")

        with open(config.weights.asr, "wb") as f:
            pickle.dump(state.params, f)
