import os
import shutil
import zipfile
import argparse
from src.model import merge_and_save_lora
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Package trained model and inference code into submission.zip")
    parser.add_argument("--model_dir", type=str, default="./output/whisper-large-v3-turbo-spanish-nahuatl", help="Trained model directory")
    parser.add_argument("--base_model_id", type=str, default="openai/whisper-large-v3-turbo", help="Base model ID")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Destination zip path")
    parser.add_argument("--temp_dir", type=str, default="./dist_submission", help="Temporary packaging directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"Packaging submission archive: {args.output_zip}")
    print("=" * 60)

    # Clean temporary directory
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
    os.makedirs(args.temp_dir, exist_ok=True)

    weights_dir = os.path.join(args.temp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Check if adapter_config exists (LoRA model)
    adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("[1/3] Merging LoRA adapter into base model for offline submission...")
        merge_and_save_lora(
            base_model_id=args.base_model_id,
            lora_weights_dir=args.model_dir,
            output_dir=weights_dir,
            torch_dtype="float16",
        )
    else:
        print("[1/3] Copying full model weights and processor files...")
        # Save model and processor in float16 to save space
        processor = WhisperProcessor.from_pretrained(args.model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype="float16",
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(weights_dir)
        processor.save_pretrained(weights_dir)

    # Use processor files from the base model for runtime-version compatibility.
    WhisperProcessor.from_pretrained(args.base_model_id).save_pretrained(weights_dir)

    # Copy submission main.py to root of temp_dir
    print("[2/3] Adding main.py entrypoint to archive root...")
    src_main_py = os.path.join("submission", "main.py")
    dst_main_py = os.path.join(args.temp_dir, "main.py")
    shutil.copy2(src_main_py, dst_main_py)

    # Create submission.zip with main.py at root
    print(f"[3/3] Creating zip archive at {args.output_zip}...")
    if os.path.exists(args.output_zip):
        os.remove(args.output_zip)

    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(args.temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, args.temp_dir)
                zipf.write(file_path, arcname=rel_path)

    # Clean up temp dir
    shutil.rmtree(args.temp_dir)

    # Verify contents of submission.zip
    print("\n--- Verifying Submission Archive ---")
    with zipfile.ZipFile(args.output_zip, "r") as zipf:
        file_list = zipf.namelist()
        if "main.py" not in file_list:
            raise RuntimeError("Verification Failed: main.py is NOT in the root of the zip archive!")
        required_paths = {
            "main.py",
            "weights/config.json",
            "weights/model.safetensors",
            "weights/preprocessor_config.json",
            "weights/tokenizer.json",
        }
        missing_paths = required_paths.difference(file_list)
        if missing_paths:
            raise RuntimeError(
                f"Verification Failed: missing archive files: {sorted(missing_paths)}"
            )
        print(f"Archive verification successful! Found {len(file_list)} files.")
        print(f"Root files: {[f for f in file_list if '/' not in f]}")
        print(f"Total archive size: {os.path.getsize(args.output_zip) / (1024 * 1024):.2f} MB")
    print("=" * 60)
    print("Ready to submit to the competition!")


if __name__ == "__main__":
    main()
