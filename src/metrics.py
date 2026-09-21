from typing import Dict, Any
import evaluate
import numpy as np
from transformers import WhisperTokenizer
from src.utils import normalize_transcript


class ComputeMetrics:
    def __init__(self, tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def __call__(self, pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize texts according to competition conventions
        pred_str_norm = [normalize_transcript(s) for s in pred_str]
        label_str_norm = [normalize_transcript(s) for s in label_str]

        # Calculate metrics (avoiding division by zero on empty references)
        filtered_pairs = [
            (p, l) for p, l in zip(pred_str_norm, label_str_norm) if len(l.strip()) > 0
        ]
        
        if not filtered_pairs:
            return {"wer": 1.0, "cer": 1.0}

        preds_filtered = [p for p, _ in filtered_pairs]
        labels_filtered = [l for _, l in filtered_pairs]

        wer = 100 * self.wer_metric.compute(predictions=preds_filtered, references=labels_filtered)
        cer = 100 * self.cer_metric.compute(predictions=preds_filtered, references=labels_filtered)

        return {"wer": wer, "cer": cer}
