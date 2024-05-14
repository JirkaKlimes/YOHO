# YOHO: You Only Hear Once

## Overview

YOHO (You Only Hear Once) is system that performs ASR with speaker diarization, and speaker recognition using a single neural network.

## Features

-   ASR (Automatic Speech Recognition): Accurate transcription of speech to text.
-   Speaker Diarization: Separation of different speakers in an audio stream.
-   Speaker Recognition (Voice Prints): Recognition of known speakers via unique voice prints.

## To-Do List

-   [x] Implement the Whisper model using the JAX framework for improved performance.
-   [x] Import pre-trained weights from OpenAI's Whisper model using the Hugging Face repository.
-   [ ] Write scraper that will generate large dataset of audio tracks with transcriptions.
-   [ ] Use Whisper model as baseline and develop better model.
-   [ ] Train a Transformer VAE to reconstruct speech utterances. The decoder will use the transcript for cross-attention, allowing the latent representation to focus on extracting the voice print while offloading the text information.
-   [ ] Use the new model for voice print generation and fine-tune the original model to output the voice print after each utterance transcript.

## Development Setup

1. **Install dependencies:**

    ```sh
    poetry install
    ```

2. **Install pre-commit hooks:**

    ```sh
    pre-commit install
    ```
