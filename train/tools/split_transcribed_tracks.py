import numpy as np
import srt
from pathlib import Path
import datetime as dt
import argparse
from tqdm import tqdm

from yoho.src.preprocessing.audio import load_audio, save_audio


def main(source_path, target_path, sample_rate, approx_duration):
    source_path = Path(source_path)
    target_path = Path(target_path)
    approx_chunk_duration = dt.timedelta(seconds=approx_duration)

    audio_files = list(source_path.joinpath("audio").iterdir())

    for audio_path in tqdm(audio_files, desc="Processing files"):
        name = audio_path.with_suffix(".srt").name
        transcript_path = source_path.joinpath("transcripts", name)

        with open(transcript_path, encoding="utf-8") as f:
            data = f.read()

        transcript = list(srt.sort_and_reindex(srt.parse(data)))[::-1]
        audio = load_audio(audio_path, sample_rate)

        start_time = dt.timedelta()

        chunks = []
        while transcript:
            chunk_transcription = []
            end_time = start_time + approx_chunk_duration

            while transcript:
                next_transcription = transcript.pop()
                if next_transcription.end > end_time:
                    transcript.append(next_transcription)
                    if next_transcription.start < end_time:
                        end_time = next_transcription.start
                    break
                chunk_transcription.append(next_transcription)

            start_sample = int(np.floor(start_time.total_seconds() * sample_rate))
            end_sample = int(np.ceil(end_time.total_seconds() * sample_rate))
            if chunk_transcription:
                chunk_audio = audio[start_sample : min(end_sample, len(audio) - 1)]
                chunks.append((chunk_audio, chunk_transcription))
            if end_sample >= len(audio):
                break

            start_time = end_time

        for i, (audio, transcript) in enumerate(chunks):
            chunk_audio_path = target_path.joinpath(
                "audio", audio_path.with_suffix(f".{i}.mp3").name
            )
            chunk_transcription_path = target_path.joinpath(
                "transcripts", audio_path.with_suffix(f".{i}.srt").name
            )
            save_audio(audio, chunk_audio_path, sample_rate)
            with open(chunk_transcription_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(transcript))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio files into chunks with corresponding transcripts."
    )
    parser.add_argument(
        "source_path",
        type=str,
        help="Path to the source directory containing audio and transcript files.",
    )
    parser.add_argument(
        "target_path", type=str, help="Path to the target directory to save processed chunks."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sample rate for audio processing."
    )
    parser.add_argument(
        "--approx_duration",
        type=int,
        default=48,
        help="Approximate duration of each chunk in seconds.",
    )

    args = parser.parse_args()
    main(args.source_path, args.target_path, args.sample_rate, args.approx_duration)
