# YOHO: You Only Hear Once

## Overview

YOHO (You Only Hear Once) is a modification of OpenAI's Whisper model that performs speaker diarization, and speaker identification using a single feature extractor.

## Features

-   Speech-to-Text: Accurate transcription of speech to text.
-   Speaker Diarization: Identification and separation of different speakers in an audio stream.
-   Speaker Identification (Voice Prints): Recognition of known speakers via unique voice prints.

## To-Do List

-   [ ] Implement the Whisper model using the JAX framework for improved performance.
-   [ ] Import pre-trained weights from OpenAI's Whisper model using the Hugging Face repository.
-   [ ] Extract only the necessary languages from the large-v3 model into a base model.
-   [ ] Develop a clustering-based algorithm to distinguish and segment different speakers.
-   [ ] Create a model that converts clustered vectors into voice prints for speaker identification.
-   [ ] Embed the voice-print prediction functionality into the original model, eliminating the need for a separate model.
