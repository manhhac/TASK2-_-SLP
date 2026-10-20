import random
import numpy as np
import torch
import torchaudio


def speed_perturb(wav: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    effects = [["speed", str(speed), "splice"], ["rate", str(sr)]]
    try:
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav.unsqueeze(0), sr, effects)
        return out.squeeze(0)
    except Exception:
        return wav


def add_gaussian_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_power = (wav ** 2).mean()
    if sig_power <= 0:
        return wav
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def add_reverb(wav: torch.Tensor, sr: int, rt60: float = 0.3) -> torch.Tensor:
    # Đơn giản: convolve với impulse response exponential decay
    n = int(rt60 * sr)
    ir = torch.exp(-torch.arange(n).float() / (rt60 * sr / 4))
    ir = ir / ir.sum()
    out = torch.nn.functional.conv1d(wav.view(1, 1, -1), ir.view(1, 1, -1), padding=n // 2)
    return out.view(-1)[: wav.size(0)]


def spec_augment(mel: torch.Tensor, freq_mask: int = 27, time_mask: int = 100,
                 n_freq: int = 2, n_time: int = 2) -> torch.Tensor:
    """mel: (n_mels, T)"""
    mel = mel.clone()
    n_mels, T = mel.shape
    for _ in range(n_freq):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0 + f, :] = 0
    for _ in range(n_time):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, T - t))
        mel[:, t0:t0 + t] = 0
    return mel


def augment_waveform(wav: torch.Tensor, sr: int, p_speed=0.35, p_noise=0.25, p_reverb=0.10):
    """Áp dụng ≥1 kỹ thuật với xác suất. Chỉ gọi trên train set."""
    if random.random() < p_speed:
        wav = speed_perturb(wav, sr, random.choice([0.9, 1.0, 1.1]))
    if random.random() < p_noise:
        snr = random.uniform(10, 20)
        wav = add_gaussian_noise(wav, snr)
    if random.random() < p_reverb:
        wav = add_reverb(wav, sr, rt60=random.uniform(0.2, 0.5))
    # chuẩn hoá biên độ sau augment
    max_val = wav.abs().max()
    if max_val > 0:
        wav = wav / max_val * 0.95
    return wav
