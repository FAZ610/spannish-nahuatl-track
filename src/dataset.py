import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from src.utils import load_audio_file, normalize_transcript


class SpanishNahuatlDataset(Dataset):
    """
    Custom PyTorch Dataset for Spanish-Nahuatl ASR speech clips.
    Loads audio, extracts log-mel spectrogram features, and tokenizes target transcripts.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_dir: str,
        processor: WhisperProcessor,
        audio_column: str = "audio_filename",
        text_column: str = "transcript",
        sampling_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        is_training: bool = True,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.is_training = is_training
        self.text_column = text_column

        # Load metadata
        df = pd.read_csv(manifest_path)
        
        # Check available columns
        if audio_column not in df.columns:
            # Fallback to first column or search
            for col in ["audio_filename", "audio_path", "filename", "clip_id", "id"]:
                if col in df.columns:
                    audio_column = col
                    break

        # Filter duration if duration column is available
        duration_cols = [c for c in df.columns if "duration" in c.lower()]
        if duration_cols:
            dur_col = duration_cols[0]
            df = df[(df[dur_col] >= min_duration) & (df[dur_col] <= max_duration)]

        # Drop rows with missing values
        if is_training and text_column in df.columns:
            df = df.dropna(subset=[audio_column, text_column])
            df = df[df[text_column].astype(str).str.strip().str.len() > 0]

        self.df = df.reset_index(drop=True)
        self.audio_column = audio_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = row[self.audio_column]
        audio_path = os.path.join(self.audio_dir, filename)

        # Load and resample audio
        audio_data = load_audio_file(audio_path, target_sr=self.sampling_rate)

        # Compute log-mel spectrogram input features (128 mel bins for Whisper Large v3 Turbo)
        input_features = self.processor.feature_extractor(
            audio_data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features[0]

        item = {
            "input_features": input_features,
            "audio_filename": filename,
        }

        # If transcript is available (training / validation), tokenize it
        if self.text_column in row and pd.notna(row[self.text_column]):
            transcript = normalize_transcript(str(row[self.text_column]))
            labels = self.processor.tokenizer(transcript).input_ids
            item["labels"] = labels

        return item
