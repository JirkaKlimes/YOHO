import matplotlib.pyplot as plt
import numpy as np

from yoho.src.preprocessing.audio import mel_spectogram, normalize_spectogram
from yoho.src.tokenizer import load_tokenizer

from train.utils.config import load_config
from train.utils.dataloaders import TranscriptionDataloader

config = load_config("main")
tokenizer = load_tokenizer(config.weights.tokenizer)

dataloader = TranscriptionDataloader(
    config,
    config.dataset.noisy.joinpath("train"),
    tokenizer,
    batch_size=16,
    max_queued_batches=1,
    num_workers=1,
    disable_warnings=True,
)

audio_batch, tokens_batch, loss_mask = dataloader.get_prepared_batch()

for audio, tokens in zip(audio_batch, tokens_batch):
    tokens = tokens[: np.max(np.nonzero(tokens)) + 1]
    tokens = list(map(int, tokens))
    print(tokenizer.Decode(tokens))

    spec = mel_spectogram(audio[2 * 16000 : 4 * 16000], 400, 160, 16000, 128)
    spec = normalize_spectogram(spec)
    plt.imshow(spec.T)
    plt.show()
