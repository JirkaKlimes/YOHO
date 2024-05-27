import pickle
import threading
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import mel_spectogram, normalize_spectogram

from train.utils.dataloaders import TranscriptionDataloader
from train.utils.config import SessionConfig

# TODO: save loss progress
# TODO: reject very wrong samples, save to bitmap image for visualization
# TODO: write webui and launch server to monitor training progress
# TODO: generate validation samples every few steps


def main(config: SessionConfig):
    HYPERPARAMETERS = config.hyperparameters.transcribe_pretrain
    STAGE_PATH = config.path.joinpath("stages", "1")
    STAGE_PATH.mkdir(exist_ok=True)
    CHECHPOINT_PATH = STAGE_PATH.joinpath("checkpoint.pkl")

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

    if CHECHPOINT_PATH.exists():
        with open(CHECHPOINT_PATH, "rb") as f:
            step, params, opt_state = pickle.load(f)

        state = TrainState(
            step,
            apply_fn=model.apply,
            params=params,
            tx=optax.MultiSteps(
                optax.adamw(HYPERPARAMETERS.learning_rate),
                HYPERPARAMETERS.accumulated_batches,
            ),
            opt_state=opt_state,
        )

    else:
        dummy_tokens = jnp.empty(
            (HYPERPARAMETERS.batch_size, config.yoho.max_text_len), dtype=jnp.uint32
        )
        dummy_spectogram = jnp.empty(
            (HYPERPARAMETERS.batch_size, config.yoho.max_audio_len, config.yoho.n_mel_bands),
            dtype=jnp.uint32,
        )
        print(model.tabulate(jax.random.key(0), dummy_tokens, dummy_spectogram, compute_flops=True))
        variables = model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_spectogram)

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optax.MultiSteps(
                optax.adamw(HYPERPARAMETERS.learning_rate),
                HYPERPARAMETERS.accumulated_batches,
            ),
        )

        with open(CHECHPOINT_PATH, "wb") as f:
            pickle.dump((state.step, state.params, state.opt_state), f)

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

    state, loss = train_step(state, *get_batch())

    losses = []
    pbar = tqdm(
        initial=int(state.step // HYPERPARAMETERS.accumulated_batches),
        total=HYPERPARAMETERS.updates,
    )
    while state.step < HYPERPARAMETERS.updates * HYPERPARAMETERS.accumulated_batches:
        accumulation_step = state.step % HYPERPARAMETERS.accumulated_batches

        spectogram, tokens, loss_mask = get_batch()
        state, loss = train_step(state, spectogram, tokens, loss_mask)
        losses.append(loss)
        pbar.set_description_str(
            f"Acc: {accumulation_step+1}/{HYPERPARAMETERS.accumulated_batches}"
        )

        if accumulation_step == HYPERPARAMETERS.accumulated_batches - 1:
            pbar.update()
            pbar.set_postfix_str(f"Loss: {sum(losses) / len(losses):.05f}")
            losses = []

            def _save():
                with open(CHECHPOINT_PATH, "wb") as f:
                    pickle.dump((state.step, state.params, state.opt_state), f)

            threading.Thread(target=_save).start()
