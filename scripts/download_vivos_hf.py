#!/usr/bin/env python
"""
Script to download VIVOS dataset from Zenodo.

Sử dụng:
    python scripts/download_vivos_hf.py --output_dir data/raw/vietnamese
"""
import argparse
import json
import tarfile
import os
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm"])
    import requests
    from tqdm import tqdm


def download_file(url, output_path, description=""):
    """Download file with progress bar."""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Download complete: {output_path}")
    return output_path


def extract_tar(tar_path, extract_dir):
    """Extract tar.gz file."""
    print(f"Extracting {tar_path} to {extract_dir}...")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

    print(f"Extraction complete: {extract_dir}")


def download_vivos_zenodo(output_dir):
    """
    Download VIVOS dataset from Zenodo.

    VIVOS: Vietnamese Speech Corpus - 15 hours, 11,660 utterances
    Direct download: https://zenodo.org/record/7068130/files/vivos.tar.gz
    """
    print("="*60)
    print("Download VIVOS Dataset from Zenodo")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # VIVOS direct download link from Zenodo
    vivos_url = "https://zenodo.org/record/7068130/files/vivos.tar.gz?download=1"
    vivos_tar = output_dir / "vivos.tar.gz"

    # Download
    download_file(vivos_url, vivos_tar, "VIVOS Dataset (1.5 GB)")

    # Extract
    extract_tar(vivos_tar, output_dir)

    # Remove tar file
    vivos_tar.unlink()
    print(f"Removed tar file: {vivos_tar}")

    # Find the extracted directory
    # Usually extracts to a 'vivos' subdirectory
    extracted_dir = output_dir / "vivos"
    if not extracted_dir.exists():
        # Check if files are directly in output_dir
        extracted_dir = output_dir

    print(f"\nData extracted to: {extracted_dir}")
    print(f"Please check the directory structure and update config accordingly.")
    print(f"Expected structure: {extracted_dir}/train/waves/ and {extracted_dir}/test/waves/")

    return extracted_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/raw/vietnamese/vivos",
                    help="Output directory for audio files")
    args = ap.parse_args()

    print("="*60)
    print("Download VIVOS from Zenodo")
    print("="*60)

    try:
        extracted_dir = download_vivos_zenodo(args.output_dir)

        print("\n" + "="*60)
        print("Complete!")
        print("="*60)
        print(f"\nData downloaded to: {extracted_dir}")
        print("\nNext steps:")
        print("1. Check the directory structure")
        print("2. Create transcripts.tsv file manually or use existing metadata")
        print("3. Update config to point to the data")
        print("4. Run prepare_data.py:")
        print("   python -m src.data.prepare_data --config configs/baseline.yaml")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
