#!/usr/bin/env python
"""
Script tạo dữ liệu mẫu nhỏ để test nhanh pipeline.

Dữ liệu mẫu bao gồm:
- 10 audio files mẫu (sử dụng audio từ librosa examples)
- transcripts.tsv tương ứng

Sử dụng:
    python scripts/create_sample_data.py --output_dir data/raw/vietnamese/sample
"""
import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Installing required libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "soundfile"])
    import numpy as np
    import soundfile as sf


def generate_sample_audio(duration=3.0, sr=16000):
    """Generate sample audio (sine wave)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate a simple melody with multiple frequencies
    frequencies = [440, 523, 659, 784]  # A4, C5, E5, G5
    audio = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        segment_start = int(i * len(t) / len(frequencies))
        segment_end = int((i + 1) * len(t) / len(frequencies))
        audio[segment_start:segment_end] = 0.3 * np.sin(2 * np.pi * freq * t[segment_start:segment_end])

    # Add some noise
    noise = 0.01 * np.random.randn(len(audio))
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9

    return audio


def create_sample_data(output_dir, n_samples=10):
    """
    Create sample dataset for testing.

    Args:
        output_dir: Output directory
        n_samples: Number of samples to create
    """
    print("="*60)
    print("Creating Sample Dataset for Testing")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample Vietnamese sentences
    sample_sentences_vi = [
        "xin chào",
        "cảm ơn",
        "tôi tên là",
        "hôm nay là thứ hai",
        "trời đẹp quá",
        "tôi thích học tiếng anh",
        "xin lỗi",
        "tạm biệt",
        "chúc một ngày tốt lành",
        "cảm ơn rất nhiều",
    ]

    # Sample English translations
    sample_sentences_en = [
        "hello",
        "thank you",
        "my name is",
        "today is monday",
        "the weather is beautiful",
        "i like learning english",
        "sorry",
        "goodbye",
        "have a nice day",
        "thank you very much",
    ]

    # Extend if needed
    while len(sample_sentences_vi) < n_samples:
        sample_sentences_vi.extend(sample_sentences_vi)
        sample_sentences_en.extend(sample_sentences_en)

    sample_sentences_vi = sample_sentences_vi[:n_samples]
    sample_sentences_en = sample_sentences_en[:n_samples]

    # Generate audio files and metadata
    all_data = []

    for i in range(n_samples):
        # Generate audio
        audio = generate_sample_audio(duration=3.0, sr=16000)
        audio_filename = f"sample_{i:04d}.wav"
        audio_path = output_dir / audio_filename

        # Save audio
        sf.write(str(audio_path), audio, 16000)

        # Create metadata
        record = {
            "audio_filename": audio_filename,
            "text_vi": sample_sentences_vi[i],
            "text_en": sample_sentences_en[i],
            "speaker_id": f"speaker_{i % 3}",  # 3 speakers
            "gender": "male" if i % 2 == 0 else "female",
            "region": "north",
        }
        all_data.append(record)

        print(f"  Created: {audio_filename}")

    # Save transcripts.tsv
    transcripts_path = output_dir / "transcripts.tsv"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for record in all_data:
            line = "\t".join([
                record["audio_filename"],
                record["text_vi"],
                record["text_en"],
                record["speaker_id"],
                record["gender"],
                record["region"],
            ])
            f.write(line + "\n")

    print(f"\nSaved transcripts to: {transcripts_path}")
    print(f"Total samples: {len(all_data)}")
    print(f"Output directory: {output_dir}")

    return transcripts_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/raw/vietnamese/sample",
                    help="Output directory for sample data")
    ap.add_argument("--n_samples", type=int, default=10,
                    help="Number of samples to create")
    args = ap.parse_args()

    print("="*60)
    print("Create Sample Dataset")
    print("="*60)

    try:
        transcripts_path = create_sample_data(args.output_dir, args.n_samples)

        print("\n" + "="*60)
        print("Complete!")
        print("="*60)
        print(f"\nSample data created in: {args.output_dir}")
        print(f"Transcripts file: {transcripts_path}")
        print("\nNext steps:")
        print("1. Update config to use sample data:")
        print("   data:")
        print("     raw_dirs: [\"data/raw/vietnamese/sample\"]")
        print("2. Run prepare_data.py:")
        print("   python -m src.data.prepare_data --config configs/baseline.yaml")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
