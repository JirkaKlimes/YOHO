import pickle
import threading
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import common_utils
from flax import jax_utils
import optax
from tqdm import tqdm
import os
import pandas as pd

from yoho.src.nn.model import Model
from yoho.src.preprocessing.tokenizer import load_tokenizer
from yoho.src.preprocessing.audio import get_batched_spectogram, normalize_spectogram

from train.utils.dataloaders import TranscriptionDataloader
from train.utils.config import SessionConfig

# TODO: reject very wrong samples, save to bitmap image for visualization
# TODO: write webui and launch server to monitor training progress
# TODO: generate validation samples every few steps


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

        self.train_dataloader = TranscriptionDataloader(
            self.config,
            self.config.dataset.noisy.joinpath("./train"),
            self.tokenizer,
            self.hyperparameters.batch_size,
            shuffle=True,
            max_queued_batches=os.cpu_count(),
            num_workers=os.cpu_count(),
            disable_warnings=True,
            warmup_queue=False,
        )
        self.val_dataloader = TranscriptionDataloader(
            self.config,
            self.config.dataset.noisy.joinpath("./val"),
            self.tokenizer,
            self.hyperparameters.batch_size,
            shuffle=True,
            max_queued_batches=1,
            num_workers=1,
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
            optax.adamw(self.learning_rate_schedule),
            self.hyperparameters.accumulated_batches,
        )

        self.state = self.load_state()
        if not self.metrics_path.exists():
            self.prepare_metrics()

    def prepare_metrics(self):
        df = pd.DataFrame({"update": [], "learning_rate": [], "loss": [], "val_loss": []})
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

    def get_batch(self, validation: bool = False):
        if validation:
            audio, tokens, loss_mask = self.val_dataloader.get_prepared_batch()
        else:
            audio, tokens, loss_mask = self.train_dataloader.get_prepared_batch()
        spectogram = self.batched_spectogram(audio)
        spectogram = jnp.astype(spectogram, jnp.float32)
        tokens = jnp.astype(tokens, jnp.uint32)
        loss_mask = jnp.astype(loss_mask, jnp.uint8)
        spectogram = normalize_spectogram(spectogram)
        return spectogram, tokens, loss_mask

    def save_metrics(self, update: int, learning_rate: float, loss: float, val_loss: float):
        metrics = pd.DataFrame(
            {
                "update": [update],
                "learning_rate": [learning_rate],
                "loss": [loss],
                "val_loss": [val_loss],
            }
        )
        metrics.to_csv(self.metrics_path, mode="a", header=False, index=False)

    def run(self):
        def loss_fn(params, state, spectogram, tokens, loss_mask):
            logits = state.apply_fn({"params": params}, tokens, spectogram)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:])
            loss *= loss_mask[:, :-1]
            loss = jnp.mean(loss)
            return loss

        def train_step(state, spectogram, tokens, loss_mask):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, state, spectogram, tokens, loss_mask)
            grads = jax.lax.pmean(grads, axis_name="devices")
            state = state.apply_gradients(grads=grads)
            return state, loss

        train_step = jax.pmap(train_step, axis_name="devices", donate_argnums=(0,))
        device_states = jax_utils.replicate(self.state)

        acc_loss = 0
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

            spectogram, tokens, loss_mask = map(common_utils.shard, self.get_batch())
            device_states, loss = train_step(device_states, spectogram, tokens, loss_mask)
            loss = jnp.mean(loss)
            self.state = jax_utils.unreplicate(device_states)
            acc_loss += loss

            pbar.set_description_str(
                f"Acc: {accumulation_step + 1}/{self.hyperparameters.accumulated_batches}"
            )

            if accumulation_step == self.hyperparameters.accumulated_batches - 1:
                batch_loss = float(acc_loss / self.hyperparameters.accumulated_batches)
                acc_loss = 0
                pbar.update()
                pbar.set_postfix_str(f"Loss: {batch_loss:.4e}")

                validation_loss = None
                if step % self.hyperparameters.validation_frequency == 0:
                    spectogram, tokens, loss_mask = self.get_batch(validation=True)
                    validation_loss = loss_fn(
                        self.state.params, self.state, spectogram, tokens, loss_mask
                    )
                    validation_loss = float(validation_loss)

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

                threading.Thread(
                    target=self.save_metrics,
                    args=(
                        int(self.state.step // self.hyperparameters.accumulated_batches),
                        float(self.learning_rate_schedule(self.state.step)),
                        batch_loss,
                        validation_loss,
                    ),
                ).start()


def main(config: SessionConfig):
    trainer = Trainer(config)
    trainer.run()
