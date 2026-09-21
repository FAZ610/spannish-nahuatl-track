"""Offline competition entry point for Spanish-Nahuatl transcription."""

import csv
import math
import os
import time
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DATA_DIR = Path("/code_execution/data")
CLIPS_DIR = DATA_DIR / "clips"
TEST_METADATA = DATA_DIR / "test_metadata.csv"
SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")
MODEL_DIR = Path(__file__).parent / "weights"

SAMPLE_RATE = 16_000
BATCH_SIZE = 16
MAX_NEW_TOKENS = 225
TIME_BUDGET_SECONDS = 100 * 60
BLANK_TRANSCRIPT = " "


def load_audio(path: Path) -> tuple[object, float]:
    audio, sample_rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio, len(audio) / SAMPLE_RATE


def clean_transcript(text: str) -> str:
    return " ".join(str(text).split()).strip()


def write_submission(filenames: list[str], transcripts: dict[str, str]) -> None:
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "audio_filename": filename,
            "transcript": clean_transcript(transcripts.get(filename, ""))
            or BLANK_TRANSCRIPT,
        }
        for filename in filenames
    ]
    pd.DataFrame(rows).to_csv(
        SUBMISSION_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
    )


def generate_short(
    model,
    processor,
    audios: list[object],
    device: torch.device,
    dtype: torch.dtype,
) -> list[str]:
    inputs = processor(
        audios,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=processor.feature_extractor.n_samples,
        truncation=True,
        return_attention_mask=True,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_features=inputs.input_features.to(device, dtype=dtype),
            attention_mask=inputs.attention_mask.to(device),
            language=None,
            task="transcribe",
            condition_on_prev_tokens=False,
            num_beams=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    return [
        clean_transcript(text)
        for text in processor.batch_decode(generated, skip_special_tokens=True)
    ]


def generate_long(
    model,
    processor,
    audio: object,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_features=inputs.input_features.to(device, dtype=dtype),
            attention_mask=inputs.attention_mask.to(device),
            language=None,
            task="transcribe",
            condition_on_prev_tokens=False,
            num_beams=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            return_timestamps=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    return clean_transcript(
        processor.batch_decode(generated, skip_special_tokens=True)[0]
    )


def verify_submission(filenames: list[str]) -> None:
    result = pd.read_csv(SUBMISSION_PATH)
    assert list(result.columns) == ["audio_filename", "transcript"]
    assert len(result) == len(filenames)
    assert result["audio_filename"].is_unique
    assert result["audio_filename"].tolist() == filenames
    assert result["transcript"].isna().sum() == 0
    print(f"[SUCCESS] Verified {len(result)} rows with no null transcripts")


def main() -> None:
    started = time.time()
    metadata = pd.read_csv(TEST_METADATA)
    filenames = metadata["audio_filename"].astype(str).tolist()
    write_submission(filenames, {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    transcripts: dict[str, str] = {}
    short_items: list[tuple[str, object, float]] = []
    long_items: list[tuple[str, object, float]] = []

    for filename in filenames:
        audio_path = CLIPS_DIR / filename
        if not audio_path.is_file():
            print(f"[WARNING] Missing audio at row {len(transcripts)}")
            transcripts[filename] = BLANK_TRANSCRIPT
            continue
        audio, duration = load_audio(audio_path)
        target = long_items if duration > 30.0 else short_items
        target.append((filename, audio, duration))

    short_items.sort(key=lambda item: item[2])
    total_batches = math.ceil(len(short_items) / BATCH_SIZE)
    for batch_index in range(total_batches):
        if time.time() - started >= TIME_BUDGET_SECONDS:
            print("[WARNING] Time budget reached")
            break
        batch = short_items[
            batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
        ]
        predictions = generate_short(
            model, processor, [item[1] for item in batch], device, dtype
        )
        transcripts.update(
            {item[0]: prediction for item, prediction in zip(batch, predictions)}
        )
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            print(f"[PROGRESS] Short batches {batch_index + 1}/{total_batches}")
            write_submission(filenames, transcripts)

    for index, (filename, audio, _) in enumerate(long_items, start=1):
        if time.time() - started >= TIME_BUDGET_SECONDS:
            print("[WARNING] Time budget reached")
            break
        transcripts[filename] = generate_long(
            model, processor, audio, device, dtype
        )
        if index % 5 == 0 or index == len(long_items):
            print(f"[PROGRESS] Long clips {index}/{len(long_items)}")
            write_submission(filenames, transcripts)

    write_submission(filenames, transcripts)
    verify_submission(filenames)


if __name__ == "__main__":
    main()
