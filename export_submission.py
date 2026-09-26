import shutil
import zipfile
import argparse
import sys
from pathlib import Path
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    temp_dir = Path(args.temp_dir)
    output_zip = Path(args.output_zip)
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        weights_dir = temp_dir / "weights"
        weights_dir.mkdir(parents=True)

        adapter_config_path = Path(args.model_dir) / "adapter_config.json"
        if adapter_config_path.is_file():
            print("[1/3] Merging LoRA adapter into base model for offline submission...")
            from src.model import merge_and_save_lora

            merge_and_save_lora(
                base_model_id=args.base_model_id,
                lora_weights_dir=args.model_dir,
                output_dir=str(weights_dir),
                torch_dtype="float16",
            )
        else:
            print("[1/3] Copying full model weights and processor files...")
            processor = WhisperProcessor.from_pretrained(args.model_dir)
            model = WhisperForConditionalGeneration.from_pretrained(
                args.model_dir,
                torch_dtype="float16",
                low_cpu_mem_usage=True,
            )
            model.save_pretrained(str(weights_dir))
            processor.save_pretrained(str(weights_dir))

        # These files must come from the base model for transformers 4.57.6
        # compatibility in the offline competition container.
        base_processor = WhisperProcessor.from_pretrained(args.base_model_id)
        base_processor.save_pretrained(str(weights_dir))
        WhisperFeatureExtractor.from_pretrained(args.base_model_id).save_pretrained(
            str(weights_dir)
        )
        WhisperTokenizer.from_pretrained(args.base_model_id).save_pretrained(
            str(weights_dir)
        )

        required_files = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
        ]
        missing_files = [
            filename for filename in required_files
            if not (weights_dir / filename).is_file()
        ]
        if missing_files:
            raise RuntimeError(
                f"Export staging is incomplete; missing files: {missing_files}"
            )

        print("[2/3] Adding main.py entrypoint to archive root...")
        entrypoint = PROJECT_ROOT / "submission" / "main.py"
        if not entrypoint.is_file():
            raise RuntimeError(f"Competition entrypoint not found: {entrypoint}")
        shutil.copy2(entrypoint, temp_dir / "main.py")

        print(f"[3/3] Creating zip archive at {output_zip}...")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        output_zip.unlink(missing_ok=True)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as zipf:
            for file_path in sorted(temp_dir.rglob("*")):
                if file_path.is_file():
                    # ZIP paths must use '/' even when the exporter runs on Windows.
                    archive_path = file_path.relative_to(temp_dir).as_posix()
                    zipf.write(file_path, arcname=archive_path)

        print("\n--- Verifying Submission Archive ---")
        with zipfile.ZipFile(output_zip, "r") as zipf:
            file_list = zipf.namelist()
            required_paths = {
                "main.py",
                "weights/config.json",
                "weights/model.safetensors",
                "weights/preprocessor_config.json",
                "weights/tokenizer.json",
            }
            missing_paths = required_paths.difference(file_list)
            if missing_paths or "main.py" not in file_list:
                raise RuntimeError(
                    "Verification Failed: archive must contain root main.py and "
                    f"required files; missing: {sorted(missing_paths)}"
                )
            if any("\\" in path for path in file_list):
                raise RuntimeError("Verification Failed: archive contains Windows path separators")
            print(f"Archive verification successful! Found {len(file_list)} files.")
            print(f"Root files: {[f for f in file_list if '/' not in f]}")
            print(f"Total archive size: {output_zip.stat().st_size / (1024 * 1024):.2f} MB")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    print("=" * 60)
    print("Ready to submit to the competition!")


if __name__ == "__main__":
    main()
