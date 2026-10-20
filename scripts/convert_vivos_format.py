#!/usr/bin/env python
"""Convert VIVOS dataset format to standard transcripts.tsv format.
   python scripts/convert_vivos_format.py --vivos_dir data/raw/vietnamese/vivos/vivos --output_dir data/raw/vietnamese/vivos
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def convert_vivos(vivos_dir, output_dir):
    """
    Convert VIVOS format to transcripts.tsv

    VIVOS format:
    - test/prompts.txt: audio_id<TAB>text
    - test/genders.txt: speaker_id<TAB>gender (m/f)
    - test/waves/speaker_id/audio_id.wav

    Output format:
    - transcripts.tsv: audio_filename<TAB>text_vi<TAB>text_en<TAB>speaker_id<TAB>gender<TAB>region
    """
    vivos_dir = Path(vivos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load speaker genders
    speaker_gender = {}
    for split in ["train", "test"]:
        gender_file = vivos_dir / split / "genders.txt"
        if gender_file.exists():
            with open(gender_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        speaker_id = parts[0]
                        gender = "male" if parts[1].lower() == "m" else "female"
                        speaker_gender[speaker_id] = gender

    # Load prompts and map to audio files
    transcripts = []
    for split in ["train", "test"]:
        prompts_file = vivos_dir / split / "prompts.txt"
        waves_dir = vivos_dir / split / "waves"

        if not prompts_file.exists():
            print(f"Warning: {prompts_file} not found, skipping {split}")
            continue

        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    audio_id = parts[0]
                    text_vi = parts[1]

                    # Extract speaker_id from audio_id (e.g., VIVOSTEST01_001 -> VIVOSTEST01)
                    speaker_id = "_".join(audio_id.split("_")[:-1])

                    # Find the actual audio file
                    audio_path = None
                    if waves_dir.exists():
                        speaker_dir = waves_dir / speaker_id
                        if speaker_dir.exists():
                            # VIVOS audio files might have different extensions
                            for ext in [".wav", ".flac", ".mp3"]:
                                audio_file = speaker_dir / f"{audio_id}{ext}"
                                if audio_file.exists():
                                    audio_path = audio_file
                                    break

                    if audio_path is None:
                        print(f"Warning: Audio file not found for {audio_id}")
                        continue

                    # Translate (VIVOS only has Vietnamese, we'll leave EN blank or use a placeholder)
                    # For now, use empty string - will need to add translation later
                    text_en = ""  # Placeholder - need to add translation

                    transcripts.append({
                        "audio_filename": audio_path.name,
                        "audio_path": str(audio_path.resolve()),
                        "text_vi": text_vi,
                        "text_en": text_en,
                        "speaker_id": speaker_id,
                        "gender": speaker_gender.get(speaker_id, "unknown"),
                        "region": "north",  # VIVOS is Northern Vietnamese
                    })

    # Write transcripts.tsv
    output_file = output_dir / "transcripts.tsv"
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio_filename", "text_vi", "text_en", "speaker_id", "gender", "region"])
        for t in transcripts:
            writer.writerow([
                t["audio_filename"],
                t["text_vi"],
                t["text_en"],
                t["speaker_id"],
                t["gender"],
                t["region"]
            ])

    print(f"Converted {len(transcripts)} samples to {output_file}")
    print(f"Note: English translations are empty - you may need to add them or use a translation service")
    return output_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vivos_dir", required=True, help="Path to VIVOS directory (containing train/ and test/)")
    ap.add_argument("--output_dir", required=True, help="Output directory for transcripts.tsv")
    args = ap.parse_args()

    convert_vivos(args.vivos_dir, args.output_dir)


if __name__ == "__main__":
    main()
