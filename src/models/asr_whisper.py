import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model


def build_whisper_asr(
    model_name="openai/whisper-small",
    language="vi",
    task="transcribe",
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None
):
    """
    Xây dựng Whisper model cho ASR task.

    Args:
        model_name: Tên model pre-trained
        language: Ngôn ngữ target (vi, en, ...)
        task: Task (transcribe, translate)
        use_lora: Có dùng LoRA không
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Danh sách modules để áp dụng LoRA

    Returns:
        model, processor
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Bắt buộc config cho generation
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    return model, processor
