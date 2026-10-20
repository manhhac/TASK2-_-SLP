import yaml
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    raw_dirs: List[str] = field(default_factory=list)
    manifest_dir: str = "data/manifests"
    feature_dir: str = "data/features"
    sample_rate: int = 16000
    min_dur: float = 0.5
    max_dur: float = 30.0
    min_rms: float = 0.005
    clip_threshold: float = 0.99
    clip_ratio_max: float = 0.20
    max_chars_per_sec: float = 25.0
    min_chars_per_sec: float = 2.0
    train_ratio: float = 0.70
    dev_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    n_mels: int = 80
    tokenizer_vocab: int = 8000


@dataclass
class ModelConfig:
    asr_model: str = "openai/whisper-small"
    mt_model: str = "Helsinki-NLP/opus-mt-vi-en"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainConfig:
    output_dir: str = "outputs/baseline"
    num_epochs: int = 10
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    patience: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2024])
    use_augment: bool = False
    use_beam_search: bool = False
    beam_size: int = 5
    use_lm_fusion: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    version: str = "baseline"
    description: str = ""


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    for key, val in raw.items():
        if hasattr(cfg, key) and isinstance(val, dict):
            sub = getattr(cfg, key)
            for k, v in val.items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
        else:
            setattr(cfg, key, val)
    return cfg
