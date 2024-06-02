import audiomentations as amts


augmenter = amts.Compose(
    [
        amts.AddGaussianSNR(p=0.3),
        amts.AirAbsorption(p=0.3),
        amts.Aliasing(p=0.3),
        amts.BandPassFilter(p=0.3),
        amts.BandStopFilter(p=0.3),
        amts.ClippingDistortion(p=0.3),
        amts.Gain(p=0.3),
        amts.GainTransition(p=0.3),
        amts.PeakingFilter(p=0.3),
        amts.PitchShift(p=0.3),
    ],
    p=0.8,
)

if __name__ == "__main__":
    import sounddevice

    from yoho.src.tokenizer import load_tokenizer

    from train.utils.config import load_config
    from train.utils.dataloaders import TranscriptionDataloader

    config = load_config("main")
    tokenizer = load_tokenizer(config.weights.tokenizer)

    dataloader = TranscriptionDataloader(
        [0, 1],
        config,
        tokenizer,
        batch_size=16,
        max_queued_batches=1,
        num_workers=1,
        disable_warnings=True,
        shuffle=False,
    )

    audio_batch, tokens_batch, loss_mask = dataloader.get_prepared_batch()

    for audio, tokens in zip(audio_batch, tokens_batch):
        audio_augmented = augmenter(audio, config.yoho.sample_rate)

        sounddevice.play(
            audio_augmented[:32000] * 32768, samplerate=config.yoho.sample_rate, blocking=True
        )
