from pydantic import BaseModel


class YOHOConfig(BaseModel):
    sample_rate: int
    n_mel_bands: int
    n_fft: int
    stft_hop: int

    # Training lengths for the model
    # Exceeding these values reduces accuracy
    max_audio_len: int
    max_text_len: int

    dims: int

    n_audio_heads: int
    n_audio_blocks: int

    n_text_heads: int
    n_text_blocks: int

    @property
    def max_input_seconds(self):
        return self.n_samples / self.sample_rate

    @property
    def n_samples(self):
        return self.n_fft + (self.max_audio_len - 1) * self.stft_hop
