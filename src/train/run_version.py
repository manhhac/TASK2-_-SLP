#!/usr/bin/env python
"""Chạy tất cả các phiên bản experiment với multiple seeds.
   python -m src.train.run_version --version baseline
   python -m src.train.run_version --version all
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


VERSIONS = [
    "baseline",
    "v1_augment",
    "v2_lora",
    "v3_noise_robust_mt",
    "v4_e2e",
]


def run_asr_training(config_path, seed):
    """Chạy ASR training cho một config và seed."""
    cmd = [
        sys.executable, "-m", "src.train.train_asr",
        "--config", config_path,
        "--seed", str(seed)
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_mt_training(config_path, seed, asr_dir=None):
    """Chạy MT training cho một config và seed."""
    cmd = [sys.executable, "-m", "src.train.train_mt", "--config", config_path, "--seed", str(seed)]
    if asr_dir:
        cmd.extend(["--asr_dir", asr_dir])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_version(version, seeds=None):
    """
    Chạy tất cả seeds cho một version.

    Args:
        version: Tên version (baseline, v1, v2, v3, v4)
        seeds: List[int] seeds, mặc định từ config
    """
    if version not in VERSIONS:
        print(f"Unknown version: {version}. Available: {VERSIONS}")
        return False

    config_path = f"configs/{version}.yaml"
    cfg = load_config(config_path)

    if seeds is None:
        seeds = cfg.train.seeds

    print(f"\n{'='*60}")
    print(f"Running version: {version}")
    print(f"Config: {config_path}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")

    success = True

    # Run ASR training cho tất cả seeds
    for seed in seeds:
        print(f"\n--- ASR Training: {version}, seed={seed} ---")
        try:
            if not run_asr_training(config_path, seed):
                print(f"ASR training failed for seed {seed}")
                success = False
                continue
        except subprocess.CalledProcessError as e:
            print(f"ASR training failed for seed {seed}: {e}")
            success = False
            continue

    # V3: Cần train MT với noisy source
    if version == "v3_noise_robust_mt":
        print(f"\n--- V3: MT Training with noisy ASR transcripts ---")
        for seed in seeds:
            asr_dir = f"{cfg.train.output_dir}/seed_{seed}/best"
            try:
                if not run_mt_training(config_path, seed, asr_dir):
                    print(f"MT training failed for seed {seed}")
                    success = False
            except subprocess.CalledProcessError as e:
                print(f"MT training failed for seed {seed}: {e}")
                success = False

    # V4: End-to-end Whisper translation (không cần MT)
    if version == "v4_e2e":
        print(f"\n--- V4: End-to-end Whisper translation ---")
        # V4 dùng task=translate thay vì task=transcribe
        # Cần chỉnh sửa train_asr.py hoặc tạo script riêng
        print("Note: V4 requires Whisper with task=translate")
        print("TODO: Implement V4 training script")

    if success:
        print(f"\n✓ Version {version} completed successfully")
    else:
        print(f"\n✗ Version {version} completed with errors")

    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="Version name or 'all'")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Specific seeds to run (default: from config)")
    args = ap.parse_args()

    if args.version == "all":
        print("Running all versions...")
        all_success = True
        for v in VERSIONS:
            if not run_version(v, args.seeds):
                all_success = False
        if all_success:
            print("\n✓ All versions completed successfully")
        else:
            print("\n✗ Some versions completed with errors")
        sys.exit(0 if all_success else 1)
    else:
        success = run_version(args.version, args.seeds)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
