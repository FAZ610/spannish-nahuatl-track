import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the speech input features and tokenized labels.
    Replaces padding token IDs in labels with -100 so they are ignored in cross-entropy loss.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract speech features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract label token sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss on padding tokens
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If decoder_start_token_id is present at the beginning of the labels, cut it
        if self.decoder_start_token_id is not None:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
