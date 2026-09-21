import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)


def get_processor(
    model_id: str = "openai/whisper-large-v3-turbo",
    language: str = "spanish",
    task: str = "transcribe",
) -> WhisperProcessor:
    """Load Whisper processor and configure prefix tokens."""
    processor = WhisperProcessor.from_pretrained(
        model_id,
        language=language,
        task=task,
    )
    return processor


def get_model(
    model_id: str = "openai/whisper-large-v3-turbo",
    config_dict: dict = None,
    use_lora: bool = True,
    lora_config_dict: dict = None,
    lora_weights_dir: str = None,
    torch_dtype: str = "bfloat16",
    device_map: str = None,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Instantiate Whisper Large v3 Turbo model and processor with optional LoRA.
    """
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    
    # Load processor
    language = "spanish"
    task = "transcribe"
    if config_dict and "model" in config_dict:
        language = config_dict["model"].get("language", "spanish")
        task = config_dict["model"].get("task", "transcribe")
        
    processor = get_processor(model_id=model_id, language=language, task=task)

    # Model kwargs
    model_kwargs = {
        "torch_dtype": dtype,
    }
    
    if config_dict and config_dict.get("model", {}).get("use_flash_attention_2", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    )

    # Configure generation parameters
    force_language = True
    if config_dict and "model" in config_dict:
        force_language = config_dict["model"].get("force_language", True)
    if force_language:
        model.generation_config.language = language
    else:
        model.generation_config.language = None
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    if use_lora:
        # Default LoRA configuration
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        r = 32
        lora_alpha = 64
        lora_dropout = 0.05
        bias = "none"

        if lora_config_dict:
            target_modules = lora_config_dict.get("target_modules", target_modules)
            r = lora_config_dict.get("r", r)
            lora_alpha = lora_config_dict.get("lora_alpha", lora_alpha)
            lora_dropout = lora_config_dict.get("lora_dropout", lora_dropout)
            bias = lora_config_dict.get("bias", bias)

        if lora_weights_dir:
            print(f"Loading initial LoRA adapter weights from {lora_weights_dir}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights_dir,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor


def merge_and_save_lora(
    base_model_id: str,
    lora_weights_dir: str,
    output_dir: str,
    torch_dtype: str = "float16",
):
    """
    Merge LoRA weights back into the base model and save full offline weights for submission.
    """
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    print(f"Loading base model: {base_model_id}...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    processor = WhisperProcessor.from_pretrained(base_model_id)

    print(f"Loading LoRA weights from: {lora_weights_dir}...")
    model = PeftModel.from_pretrained(base_model, lora_weights_dir)
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Export complete!")
