from dataclasses import dataclass


@dataclass
class YOHOConfig:
    sample_rate: int = 16000
    n_mel_bands: int = 128
    n_fft: int = 400
    stft_hop: int = 160

    # Training lengths for the model
    # Exceeding these values reduces accuracy
    max_audio_len: int = 2048
    max_text_len: int = 512

    dims: int = 512

    n_audio_heads: int = 8
    n_audio_blocks: int = 6

    n_text_heads: int = 8
    n_text_blocks: int = 6

    @property
    def max_input_seconds(self):
        return self.n_samples / self.sample_rate

    @property
    def n_samples(self):
        return self.n_fft + (self.max_audio_len - 1) * self.stft_hop
