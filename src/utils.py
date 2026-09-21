import os
import re
import random
import unicodedata
import yaml
import torch
import numpy as np
import soundfile as sf
import librosa


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str):
    """Save dictionary as YAML configuration file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def normalize_transcript(text: str) -> str:
    """
    Normalizes transcript text for training and comparable WER evaluation:
    - Removes bracketed tags (e.g., [laughter], <music>)
    - Uses Unicode-normalized lowercase text
    - Replaces punctuation with spaces
    - Cleans extra whitespace

    Letters and combining marks are preserved so Nahuatl orthography is not
    altered. This is a punctuation/case normalization, not a spelling
    conversion between dialects.
    """
    if not isinstance(text, str):
        return ""

    # Remove bracketed and parenthesized annotation tags like [laughter], <crying>, (cough)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\<.*?\>", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)

    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_audio_file(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file, convert to mono, and resample to target sampling rate.
    Uses soundfile first, falling back to librosa if necessary (e.g. for mp3 format).
    """
    try:
        # Try soundfile first (fast)
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    except Exception:
        # Fallback to librosa (handles mp3, ogg, etc.)
        data, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    return data.astype(np.float32)
