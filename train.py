import os
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.utils import set_seed, load_config
from src.model import get_model, merge_and_save_lora
from src.dataset import SpanishNahuatlDataset
from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import ComputeMetrics


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer with Whisper-safe label smoothing.

    The built-in Trainer label smoother can provide both decoder input forms
    with some Transformers/Whisper combinations. Passing labels directly to
    Whisper and smoothing its returned logits avoids that conflict.
    """

    def __init__(self, *args, label_smoothing_factor: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.whisper_label_smoothing_factor = label_smoothing_factor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        smoothing = self.whisper_label_smoothing_factor
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper Large v3 Turbo for Spanish-Nahuatl ASR")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--train_manifest", type=str, default=None, help="Override train manifest path")
    parser.add_argument("--train_audio_dir", type=str, default=None, help="Override train audio dir")
    parser.add_argument("--val_manifest", type=str, default=None, help="Override validation manifest path")
    parser.add_argument("--val_audio_dir", type=str, default=None, help="Override validation audio dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Load the base model and processor from a local full checkpoint",
    )
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override grad accum steps")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for a smoke test")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit evaluation samples for a smoke test")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a Trainer checkpoint")
    parser.add_argument(
        "--init_lora_from",
        type=str,
        default=None,
        help="Initialize from LoRA adapter weights without restoring Trainer state",
    )
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into standalone model after training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.train_manifest:
        config["data"]["train_manifest"] = args.train_manifest
    if args.train_audio_dir:
        config["data"]["train_audio_dir"] = args.train_audio_dir
    if args.val_manifest:
        config["data"]["val_manifest"] = args.val_manifest
    if args.val_audio_dir:
        config["data"]["val_audio_dir"] = args.val_audio_dir
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.base_model_path:
        config.setdefault("model", {})["base_model_path"] = args.base_model_path
    if args.num_train_epochs:
        config["training"]["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    set_seed(42)

    model_id = config["model"]["model_id"]
    base_model_source = config["model"].get("base_model_path") or model_id
    use_lora = config["training"].get("use_lora", True)
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Initializing Fine-tuning for {model_id}")
    print(f"PEFT / LoRA enabled: {use_lora}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # 1. Initialize Model & Processor
    model, processor = get_model(
        model_id=model_id,
        config_dict=config,
        use_lora=use_lora,
        lora_config_dict=config.get("lora", {}),
        lora_weights_dir=args.init_lora_from,
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
    )
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Enable gradient checkpointing for memory efficiency
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # 2. Build Datasets
    print("Loading datasets...")
    train_dataset = SpanishNahuatlDataset(
        manifest_path=config["data"]["train_manifest"],
        audio_dir=config["data"]["train_audio_dir"],
        processor=processor,
        audio_column=config["data"].get("audio_column", "audio_filename"),
        text_column=config["data"].get("text_column", "transcript"),
        sampling_rate=config["model"].get("sampling_rate", 16000),
        max_duration=config["data"].get("max_duration_seconds", 30.0),
        min_duration=config["data"].get("min_duration_seconds", 0.5),
        is_training=True,
        max_samples=args.max_train_samples,
        spec_augment=config["data"].get("spec_augment", False),
        frequency_mask_param=config["data"].get("frequency_mask_param", 15),
        time_mask_param=config["data"].get("time_mask_param", 35),
        num_frequency_masks=config["data"].get("num_frequency_masks", 2),
        num_time_masks=config["data"].get("num_time_masks", 2),
    )
    print(f"Train samples: {len(train_dataset)}")

    eval_dataset = None
    if os.path.exists(config["data"].get("val_manifest", "")):
        eval_dataset = SpanishNahuatlDataset(
            manifest_path=config["data"]["val_manifest"],
            audio_dir=config["data"]["val_audio_dir"],
            processor=processor,
            audio_column=config["data"].get("audio_column", "audio_filename"),
            text_column=config["data"].get("text_column", "transcript"),
            sampling_rate=config["model"].get("sampling_rate", 16000),
            max_duration=config["data"].get("max_duration_seconds", 30.0),
            min_duration=config["data"].get("min_duration_seconds", 0.5),
            is_training=False,
            max_samples=args.max_eval_samples,
        )
        print(f"Evaluation samples: {len(eval_dataset)}")

    # 3. Data Collator & Metrics
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_dtype=next(model.parameters()).dtype,
    )
    compute_metrics = ComputeMetrics(tokenizer=processor.tokenizer)

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config["training"].get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
        learning_rate=config["training"].get("learning_rate", 1e-4),
        warmup_steps=config["training"].get("warmup_steps", 100),
        num_train_epochs=config["training"].get("num_train_epochs", 5),
        weight_decay=config["training"].get("weight_decay", 0.01),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        logging_steps=config["training"].get("logging_steps", 25),
        eval_strategy=config["training"].get("evaluation_strategy", "steps") if eval_dataset else "no",
        eval_steps=config["training"].get("eval_steps", 100) if eval_dataset else None,
        save_strategy=config["training"].get("save_strategy", "steps"),
        save_steps=config["training"].get("save_steps", 100),
        save_total_limit=config["training"].get("save_total_limit", 3),
        load_best_model_at_end=config["training"].get("load_best_model_at_end", True) if eval_dataset else False,
        metric_for_best_model=config["training"].get("metric_for_best_model", "wer") if eval_dataset else None,
        greater_is_better=config["training"].get("greater_is_better", False) if eval_dataset else False,
        fp16=config["training"].get("fp16", False),
        bf16=config["training"].get("bf16", True),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 2),
        predict_with_generate=config["training"].get("predict_with_generate", True),
        generation_max_length=config["training"].get("generation_max_length", 225),
        # Built-in label smoothing is disabled because Whisper runtimes can
        # receive conflicting decoder inputs; the custom trainer applies it.
        label_smoothing_factor=0.0,
        report_to=["none"],
        remove_unused_columns=False,
    )

    # 5. Trainer
    trainer = WhisperSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
        processing_class=processor.feature_extractor,
        label_smoothing_factor=config["training"].get("label_smoothing_factor", 0.0),
    )

    # 6. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 7. Save Final Model & Processor
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Optional: Merge LoRA weights into a unified model folder
    if use_lora and args.merge_lora:
        merged_dir = os.path.join(output_dir, "merged_model")
        print(f"Merging LoRA weights into {merged_dir}...")
        merge_and_save_lora(
            base_model_id=base_model_source,
            lora_weights_dir=output_dir,
            output_dir=merged_dir,
            torch_dtype=config["inference"].get("torch_dtype", "float16"),
        )

    print("Training finished successfully!")


if __name__ == "__main__":
    main()
