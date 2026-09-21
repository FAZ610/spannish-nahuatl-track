import os
import sys
import time
import math
import csv
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Configuration and Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("SUBMISSION_DIR", "submission")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", os.path.join(ROOT_DIR, "weights"))

TEST_METADATA_PATH = os.path.join(DATA_DIR, "test_metadata.csv")
AUDIO_CLIPS_DIR = os.path.join(DATA_DIR, "clips")
OUTPUT_SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

BATCH_SIZE = 32
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 225


def clean_transcript(text: str) -> str:
    """Clean generated transcript for final submission."""
    if not isinstance(text, str):
        return ""
    # Strip unnecessary whitespaces and trailing artifacts
    return " ".join(text.split()).strip()


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio with fallback mechanism."""
    try:
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    except Exception:
        data, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return data.astype(np.float32)


def main():
    start_time = time.time()
    print("[INFO] Starting Lost in Transcription (Spanish-Nahuatl) inference pipeline...")

    # Validate paths
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at: {TEST_METADATA_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[INFO] Execution device: {device}, Precision: {dtype}")

    # Load Model & Processor from local weights directory
    model_source = WEIGHTS_DIR if os.path.exists(WEIGHTS_DIR) else ROOT_DIR
    print(f"[INFO] Loading Whisper Large v3 Turbo model from: {model_source}")

    processor = WhisperProcessor.from_pretrained(model_source, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    # Load test metadata manifest
    metadata_df = pd.read_csv(TEST_METADATA_PATH)
    total_samples = len(metadata_df)
    print(f"[INFO] Total test samples to transcribe: {total_samples}")

    audio_filenames = metadata_df["audio_filename"].tolist()
    predictions = []

    # Batch inference loop
    total_batches = math.ceil(total_samples / BATCH_SIZE)
    log_interval = max(1, total_batches // 10)  # Log ~10 progress updates to respect log quota

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_samples)
        batch_files = audio_filenames[start_idx:end_idx]

        batch_audio = []
        for fname in batch_files:
            audio_path = os.path.join(AUDIO_CLIPS_DIR, fname)
            if os.path.exists(audio_path):
                audio = load_audio(audio_path, target_sr=SAMPLE_RATE)
            else:
                # In case of missing clip, pad with 1 second of silence
                audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            batch_audio.append(audio)

        # Feature extraction
        inputs = processor(
            batch_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device, dtype=dtype)

        # Generate transcriptions
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features,
                language="spanish",
                task="transcribe",
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Decode tokens to text
        batch_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for pred in batch_preds:
            predictions.append(clean_transcript(pred))

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            progress = ((batch_idx + 1) / total_batches) * 100
            elapsed = time.time() - start_time
            print(f"[PROGRESS] Completed batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) in {elapsed:.1f}s")

    # Build submission DataFrame
    submission_df = pd.DataFrame({
        "audio_filename": audio_filenames,
        "transcript": predictions,
    })

    # Save to CSV using standard quoting rules
    submission_df.to_csv(
        OUTPUT_SUBMISSION_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
    )

    total_time = time.time() - start_time
    print(f"[SUCCESS] Predictions written to: {OUTPUT_SUBMISSION_PATH}")
    print(f"[SUCCESS] Rows: {len(submission_df)}, Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
