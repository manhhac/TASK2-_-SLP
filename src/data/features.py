"""Trích xuất và lưu feature statistics từ audio."""
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


def extract_audio_features(audio_path: str, sr: int = 16000) -> Dict:
    """
    Trích xuất features từ file audio.

    Returns:
        Dict với các thông số: duration, rms, spectral_centroid, spectral_rolloff, zero_crossing_rate
    """
    try:
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

        if len(y) == 0:
            return None

        features = {
            "duration": float(duration),
            "sr": int(sr),
            "rms": float(np.sqrt(np.mean(y ** 2))),
            "n_samples": len(y),
        }

        # Spectral features
        try:
            features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        except Exception:
            features["spectral_centroid"] = 0.0
            features["spectral_rolloff"] = 0.0
            features["zcr"] = 0.0

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def compute_feature_stats(manifest_path: str, output_path: str) -> Dict:
    """
    Tính toán thống kê tổng hợp từ features của toàn bộ dataset.

    Args:
        manifest_path: File manifest
        output_path: File lưu thống kê

    Returns:
        Dict thống kê
    """
    # Load manifest
    audio_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            audio_paths.append(row["audio_path"])

    # Extract features
    all_features = []
    for path in tqdm(audio_paths, desc="Extracting features"):
        feat = extract_audio_features(path)
        if feat is not None:
            all_features.append(feat)

    if not all_features:
        print("No valid features extracted")
        return {}

    # Compute statistics
    df = {k: [] for k in all_features[0].keys()}
    for feat in all_features:
        for k, v in feat.items():
            df[k].append(v)

    stats = {}
    for k, values in df.items():
        if isinstance(values[0], (int, float)):
            stats[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved feature stats to {output_path}")
    return stats


def extract_mel_spectrogram(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 80,
    output_path: str = None
) -> np.ndarray:
    """
    Trích xuất mel spectrogram từ audio.

    Args:
        audio_path: Đường dẫn file audio
        sr: Sample rate
        n_mels: Số mel bands
        output_path: Nếu được cung cấp, lưu mel spectrogram ra file .npy

    Returns:
        Mel spectrogram array (n_mels, T)
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mel_db)

    return mel_db


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--action", choices=["stats", "mel"], default="stats")
    ap.add_argument("--audio", default=None)
    args = ap.parse_args()

    if args.action == "stats":
        stats = compute_feature_stats(args.manifest, args.output)
        print(json.dumps(stats, indent=2))
    elif args.action == "mel":
        if args.audio:
            mel = extract_mel_spectrogram(args.audio, output_path=args.output)
            print(f"Extracted mel spectrogram shape: {mel.shape}")
        else:
            print("Please provide --audio for mel extraction")
