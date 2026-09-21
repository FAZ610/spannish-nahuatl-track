import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from peft import PeftModel
from src.utils import load_config, normalize_transcript, load_audio_file
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model on dev/eval set")
    parser.add_argument("--model_dir", type=str, default="./output/whisper-large-v3-turbo-spanish-nahuatl", help="Model directory or merged checkpoint")
    parser.add_argument("--base_model_id", type=str, default="openai/whisper-large-v3-turbo", help="Base model id if using LoRA")
    parser.add_argument("--manifest", type=str, default="data/dev_metadata.csv", help="Path to evaluation manifest CSV or TSV")
    parser.add_argument("--audio_dir", type=str, default="nahuatl_dev/clips", help="Path to evaluation audio clips directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--output_csv", type=str, default="output/eval_predictions.csv", help="Path to save predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", type=str, default=None, help="Optional forced decode language")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam count for generation")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature; 0 uses deterministic decoding",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty used during beam search",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading processor and model from {args.model_dir} (Device: {device})...")

    # Load processor
    try:
        processor = WhisperProcessor.from_pretrained(args.model_dir)
    except Exception:
        processor = WhisperProcessor.from_pretrained(args.base_model_id)

    # Load model (handle both merged model and LoRA adapter)
    adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("Detected LoRA adapter checkpoint, loading base model + adapter...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    else:
        print("Loading full / merged model checkpoint...")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    model.to(device)
    model.eval()

    # Load metadata
    delimiter = "\t" if args.manifest.lower().endswith((".tsv", ".tab")) else ","
    df = pd.read_csv(args.manifest, sep=delimiter, index_col=0 if delimiter == "\t" else None)
    print(f"Loaded {len(df)} evaluation samples from {args.manifest}")

    predictions = []
    references = []
    audio_filenames = []

    # Batch processing
    for i in tqdm(range(0, len(df), args.batch_size), desc="Evaluating"):
        batch_df = df.iloc[i:i + args.batch_size]
        
        batch_audio = []
        batch_filenames = []
        batch_refs = []

        for _, row in batch_df.iterrows():
            fname = row["audio_filename"] if "audio_filename" in row else row.iloc[0]
            audio_path = os.path.join(args.audio_dir, fname)
            
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue

            audio = load_audio_file(audio_path, target_sr=16000)
            batch_audio.append(audio)
            batch_filenames.append(fname)
            if "transcript" in row:
                batch_refs.append(normalize_transcript(str(row["transcript"])))

        if not batch_audio:
            continue

        inputs = processor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=processor.feature_extractor.n_samples,
            truncation=True,
            return_attention_mask=True,
        )
        generation_inputs = {
            "input_features": inputs.input_features.to(device, dtype=dtype),
        }
        if "attention_mask" in inputs:
            generation_inputs["attention_mask"] = inputs.attention_mask.to(device)

        with torch.inference_mode():
            generation_kwargs = {
                **generation_inputs,
                **({"language": args.language} if args.language else {}),
                "task": "transcribe",
                "num_beams": args.num_beams,
                "condition_on_prev_tokens": False,
                "max_new_tokens": 225,
                "length_penalty": args.length_penalty,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature

            predicted_ids = model.generate(
                **generation_kwargs,
            )

        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        normalized_preds = [normalize_transcript(t) for t in transcriptions]

        predictions.extend(normalized_preds)
        audio_filenames.extend(batch_filenames)
        if batch_refs:
            references.extend(batch_refs)

    # Save predictions
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    res_df = pd.DataFrame({
        "audio_filename": audio_filenames,
        "transcript": predictions,
    })
    if references:
        res_df["reference"] = references

    res_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    # Compute WER & CER if references available
    if references:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        wer = 100 * wer_metric.compute(predictions=predictions, references=references)
        cer = 100 * cer_metric.compute(predictions=predictions, references=references)

        print("\n" + "=" * 50)
        print(f"Evaluation Results:")
        print(f"  WER: {wer:.2f}%")
        print(f"  CER: {cer:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
