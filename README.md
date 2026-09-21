# Spanish-Nahuatl Whisper Large v3 Turbo Fine-Tuning & Submission Pipeline

This repository contains the complete, production-ready codebase to fine-tune **Whisper Large v3 Turbo** (`openai/whisper-large-v3-turbo`) for the **Lost in Transcription: Spanish-Nahuatl Track** competition hosted by Mozilla Data Collective and DrivenData.

---

## 📁 Repository Structure

```
.
├── config.yaml               # Training, model, LoRA, and data hyperparameters
├── requirements.txt          # Python dependencies
├── train.py                  # Main training entrypoint (Seq2SeqTrainer + LoRA/PEFT)
├── prepare_data.py           # Conversation-level train/dev split generator
├── evaluate_model.py         # Offline validation and WER/CER error analysis
├── export_submission.py      # Exports & packages merged weights + main.py into submission.zip
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Audio loading, 128-mel feature extraction, and tokenization
│   ├── collator.py           # DataCollatorSpeechSeq2SeqWithPadding with -100 label masking
│   ├── metrics.py            # Normalized WER and CER calculation using evaluate & jiwer
│   ├── model.py              # Whisper Large v3 Turbo loader & LoRA configuration
│   └── utils.py              # Audio resampling, text normalization & seed management
├── submission/
│   └── main.py               # Competition container inference entrypoint
└── data/                     # Local data directory (place competition data here)
```

---

## 🚀 Quick Start

### 1. Environment Setup

Python 3.12 (or 3.10+) with PyTorch and CUDA support:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

For the cloned `nahuatl_dev` data, create a conversation-level train/dev split:

```bash
python prepare_data.py
```

The script writes `data/train_metadata.csv` and `data/dev_metadata.csv`; both
splits use audio from `nahuatl_dev/clips`.

### Combine the Tetelancingo corpus

The Tetelancingo corpus already provides an official `train`/`test` split.
Do not randomly reshuffle those rows: the five speakers occur in both splits,
so preserving the supplied split gives a useful held-out evaluation set.

After extracting both datasets, build combined manifests with:

```bash
python prepare_combined_data.py
```

This writes:

```text
data/combined/train_metadata.csv  # 227 competition + 2,412 Tetelancingo rows
data/combined/dev_metadata.csv    # 110 competition + 269 Tetelancingo rows
```

The Tetelancingo `sentence` field is used as the ASR transcript. Its
`original_sentence`, Spanish translation, and `lid_tokens` fields are not
ASR targets and are intentionally not included. Audio paths in these
manifests are repository-relative, so both datasets can be used with
`train_audio_dir: "."`.

For the full competition dataset, organize it under `./data/`:

```
data/
├── train_metadata.csv        # Columns: audio_filename, transcript, language, file_duration_seconds
├── train_clips/              # Directory with .mp3 / .wav audio files
├── dev_metadata.csv          # Validation metadata (same schema)
├── dev_clips/                # Validation audio files
├── test_metadata.csv         # Provided during evaluation / smoke test
└── clips/                    # Test clips directory
```

For the combined manifests, set the data section in `config.yaml` to:

```yaml
data:
  train_manifest: "data/combined/train_metadata.csv"
  train_audio_dir: "."
  val_manifest: "data/combined/dev_metadata.csv"
  val_audio_dir: "."
```

---

## 🏋️ Fine-Tuning Whisper Large v3 Turbo

### Run Training with LoRA (Recommended)

Fine-tuning using Parameter-Efficient Fine-Tuning (PEFT / LoRA) on query/value/key projection layers and MLP layers:

```bash
python train.py --config config.yaml
```

### Key Training Options:
- `--config`: Path to YAML configuration (default: `config.yaml`).
- `--train_manifest`: Path to training CSV manifest.
- `--train_audio_dir`: Directory containing training audio files.
- `--val_manifest`: Path to validation CSV manifest.
- `--val_audio_dir`: Directory containing validation audio files.
- `--output_dir`: Directory where checkpoints are saved.
- `--batch_size`: Per-device batch size (e.g., 8 or 16).
- `--learning_rate`: Peak learning rate (default: `5e-5` for LoRA).
- `--num_train_epochs`: Total epochs (default: `5`).
- `--merge_lora`: Automatically merge LoRA adapter into full model after training.

---

## 📊 Offline Evaluation & Error Inspection

Evaluate a trained checkpoint on your local dev/validation set to compute Word Error Rate (WER) and Character Error Rate (CER):

```bash
python evaluate_model.py \
    --model_dir ./output/whisper-large-v3-turbo-spanish-nahuatl \
    --manifest data/dev_metadata.csv \
    --audio_dir data/dev_clips \
    --batch_size 16
```

---

## 📦 Exporting & Packaging for Submission

The competition requires a standalone `submission.zip` with `main.py` at the root and all model weights bundled for offline execution in the container (1 A100 GPU, no internet access, 2-hour limit).

Run:

```bash
python export_submission.py \
    --model_dir ./output/whisper-large-v3-turbo-spanish-nahuatl \
    --output_zip submission.zip
```

This will:
1. Merge LoRA adapter weights (if applicable) into a unified `float16` checkpoint inside `./weights/`.
2. Copy the submission inference script `submission/main.py` directly to the root of the archive.
3. Package everything into `submission.zip` and verify that `main.py` is at the root.

---

## 🐳 Local Docker Testing (Before Submitting)

You can test your `submission.zip` locally using the official competition runtime container:

```bash
git clone https://github.com/drivendataorg/lost-in-transcription-runtime
cd lost-in-transcription-runtime
make build
make pack-submission SUBMISSION_PATH=/path/to/submission.zip
make test-submission
```

---

## 📝 Competition Compliance Checklist

- [x] Python 3.12 compatible.
- [x] `main.py` is at the root of `submission.zip`.
- [x] Model weights are bundled for 100% offline inference.
- [x] Reads test metadata from `data/test_metadata.csv` and audio from `data/clips/`.
- [x] Writes predictions with exact headers `audio_filename,transcript` to `submission/submission.csv`.
- [x] Respects logging limits (fewer than 500 lines, under 300 characters per line, no test data exfiltration).
- [x] Optimized for fast batch inference on A100 GPU (well within the 2-hour execution limit).
