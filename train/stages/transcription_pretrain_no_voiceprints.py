import pickle
import threading
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import common_utils
from flax import jax_utils
import flax.linen as nn
import optax
from tqdm import tqdm
import os
import pandas as pd

from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import get_batched_spectogram

from train.utils.dataloaders import TranscriptionDataloader
from train.utils.config import SessionConfig

# TODO: reject very wrong samples, save to bitmap image for visualization
# TODO: write webui and launch server to monitor training progress
# TODO: generate validation samples every few steps


def fully_flatten_tree(tree):
    flat_leaves = jax.tree.map(jnp.ravel, tree)
    flat_tree, _ = jax.tree.flatten(flat_leaves)
    flat_tree = jnp.concat(flat_tree)
    return flat_tree


class Trainer:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.hyperparameters = config.hyperparameters.transcribe_pretrain
        self.stage_path = self.config.path.joinpath("stages", "1")
        self.stage_path.mkdir(exist_ok=True)
        self.checkpoint_path = self.stage_path.joinpath("checkpoint.pkl")
        self.metrics_path = self.stage_path.joinpath("metrics.csv")

        self.tokenizer = load_tokenizer(config.weights.tokenizer)
        self.model = Model(self.config.yoho, self.tokenizer.vocab_size())
        self.dataloader = TranscriptionDataloader(
            self.config,
            self.tokenizer,
            self.hyperparameters.batch_size,
            shuffle=True,
            max_queued_batches=os.cpu_count(),
            num_workers=os.cpu_count(),
            disable_warnings=True,
            warmup_queue=False,
        )

        self.batched_spectogram = get_batched_spectogram(self.config.yoho)

        self.learning_rate_schedule = optax.schedules.warmup_cosine_decay_schedule(
            0.0,
            self.hyperparameters.learning_rate,
            self.hyperparameters.warmup_updates * self.hyperparameters.accumulated_batches,
            (self.hyperparameters.updates - self.hyperparameters.warmup_updates)
            * self.hyperparameters.accumulated_batches,
            self.hyperparameters.final_learning_rate,
        )

        self.optimizer = optax.MultiSteps(
            optax.sgd(self.learning_rate_schedule, self.hyperparameters.momentum),
            self.hyperparameters.accumulated_batches,
        )

        self.state = self.load_state()
        if not self.metrics_path.exists():
            self.prepare_metrics()

    def prepare_metrics(self):
        df = pd.DataFrame({"update": [], "learning_rate": [], "loss": [], "grad_divergence": []})
        df.to_csv(self.metrics_path, index=False)

    def load_state(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                step, params, opt_state = pickle.load(f)

            state = TrainState(
                step,
                apply_fn=self.model.apply,
                params=params,
                tx=self.optimizer,
                opt_state=opt_state,
            )
            return state

        else:
            dummy_tokens = jnp.empty(
                (self.hyperparameters.batch_size, self.config.yoho.max_text_len), dtype=jnp.uint32
            )
            dummy_spectogram = jnp.empty(
                (
                    self.hyperparameters.batch_size,
                    self.config.yoho.max_audio_len,
                    self.config.yoho.n_mel_bands,
                ),
                dtype=jnp.uint32,
            )
            print(
                self.model.tabulate(
                    jax.random.key(0), dummy_tokens, dummy_spectogram, compute_flops=True
                )
            )
            variables = self.model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_spectogram)

            state = TrainState.create(
                apply_fn=self.model.apply,
                params=variables["params"],
                tx=self.optimizer,
            )

            return state

    def get_batch(self):
        audio, tokens, loss_mask = self.dataloader.get_prepared_batch()
        spectogram = self.batched_spectogram(audio)
        spectogram = jnp.astype(spectogram, jnp.float32)
        tokens = jnp.astype(tokens, jnp.uint32)
        loss_mask = jnp.astype(loss_mask, jnp.uint8)
        # TODO: normalize spectogram
        return spectogram, tokens, loss_mask

    def save_metrics(self, update: int, learning_rate: float, loss: float, grad_divergence: float):
        metrics = pd.DataFrame(
            {
                "update": [update],
                "learning_rate": [learning_rate],
                "loss": [loss],
                "grad_divergence": [grad_divergence],
            }
        )
        metrics.to_csv(self.metrics_path, mode="a", header=False, index=False)

    def run(self):
        def train_step(state, spectogram, tokens, loss_mask):
            def sample_corectness(state, spectogram, embeddings, loss_mask):
                def loss_fn(params, spectogram, embeddings, loss_mask):
                    audio_encoded = state.apply_fn(
                        {"params": params}, spectogram, method=Model.encode_audio
                    )
                    logits = state.apply_fn(
                        {"params": params}, embeddings, audio_encoded, method=Model.decode_text
                    )
                    loss = optax.softmax_cross_entropy_with_integer_labels(
                        logits[:, :-1], tokens[:, 1:]
                    )
                    loss *= loss_mask[:, :-1]
                    loss = jnp.mean(loss)
                    return loss

                grad_fn = jax.value_and_grad(loss_fn)
                loss, grads = grad_fn(state.params, spectogram, embeddings, loss_mask)
                grads = jax.lax.pmean(grads, axis_name="devices")

                grad_divergence = jnp.dot(
                    fully_flatten_tree(grads),
                    fully_flatten_tree(state.opt_state.inner_opt_state[0].trace),
                )

                grad_divergence = jnp.log(1 + jnp.e ** (jnp.e - grad_divergence))

                return grad_divergence, (grad_divergence, loss)

            embeddings = state.apply_fn(
                {"params": state.params}, tokens, method=Model.embedd_tokens
            )

            rejection_loss_fn = jax.grad(sample_corectness, argnums=1, has_aux=True)

            sample_gradients, (grad_divergence, loss) = rejection_loss_fn(
                state, spectogram, embeddings, loss_mask
            )
            sample_gradients = jnp.mean(sample_gradients**2, axis=(1, 2))
            sample_gradients = nn.softmax(sample_gradients)

            def loss_fn(params, spectogram, tokens, loss_mask):
                embeddings = state.apply_fn(
                    {"params": state.params}, tokens, method=Model.embedd_tokens
                )
                audio_encoded = state.apply_fn(
                    {"params": params}, spectogram, method=Model.encode_audio
                )
                logits = state.apply_fn(
                    {"params": params}, embeddings, audio_encoded, method=Model.decode_text
                )
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits[:, :-1], tokens[:, 1:]
                )
                loss *= loss_mask[:, :-1]
                loss = jnp.mean(loss)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, spectogram, tokens, loss_mask)
            grads = jax.lax.pmean(grads, axis_name="devices")

            state = state.apply_gradients(grads=grads)

            return state, loss, grad_divergence, sample_gradients

        train_step = jax.pmap(train_step, axis_name="devices", donate_argnums=(0,))
        device_states = jax_utils.replicate(self.state)

        acc_loss = 0
        acc_grad_div = 0
        pbar = tqdm(
            initial=int(self.state.step // self.hyperparameters.accumulated_batches),
            total=self.hyperparameters.updates,
        )
        while (
            self.state.step
            < self.hyperparameters.updates * self.hyperparameters.accumulated_batches
        ):
            accumulation_step = self.state.step % self.hyperparameters.accumulated_batches
            step = self.state.step // self.hyperparameters.accumulated_batches

            spectogram, tokens, loss_mask = list(map(common_utils.shard, self.get_batch()))

            device_states, loss, grad_div, sample_gradient = train_step(
                device_states, spectogram, tokens, loss_mask
            )
            loss = jnp.mean(loss)
            grad_div = jnp.mean(grad_div)
            acc_loss += loss
            acc_grad_div += grad_div
            print(jnp.reshape(sample_gradient, (8, -1)))

            self.state = jax_utils.unreplicate(device_states)

            pbar.set_description_str(
                f"Acc: {accumulation_step+1}/{self.hyperparameters.accumulated_batches}"
            )

            if accumulation_step == self.hyperparameters.accumulated_batches - 1:
                batch_loss = float(acc_loss / self.hyperparameters.accumulated_batches)
                batch_grad_div = float(acc_grad_div / self.hyperparameters.accumulated_batches)
                acc_loss = 0
                acc_grad_div = 0

                pbar.update()
                pbar.set_postfix_str(f"Loss: {batch_loss:.4e}")

                threading.Thread(
                    target=self.save_metrics,
                    args=(
                        int(self.state.step // self.hyperparameters.accumulated_batches),
                        float(self.learning_rate_schedule(self.state.step)),
                        batch_loss,
                        batch_grad_div,
                    ),
                ).start()

                if step % self.hyperparameters.validation_frequency == 0:
                    host_state = jax.device_get(self.state)

                    def _save():
                        with open(self.checkpoint_path, "wb") as f:
                            pickle.dump(
                                (
                                    host_state.step,
                                    host_state.params,
                                    host_state.opt_state,
                                ),
                                f,
                            )

                    threading.Thread(target=_save).start()


def main(config: SessionConfig):
    trainer = Trainer(config)
    trainer.run()
