#!/usr/bin/env python
"""
Script tải dữ liệu train cho Task 2 ASR VI + MT EN.

Dữ liệu sẽ được tải về:
- VIVOS: Vietnamese speech corpus (ASR)
- PhoMT: Vietnamese-English parallel corpus (MT)

Sử dụng:
    python scripts/download_data.py --output_dir data/raw
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

try:
    import gdown
    import requests
    from tqdm import tqdm
except ImportError:
    print("Đang cài đặt các thư viện cần thiết...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "requests", "tqdm"])
    import gdown
    import requests
    from tqdm import tqdm


def download_with_gdown(url, output_path, description=""):
    """Tải file sử dụng gdown (cho Google Drive)."""
    print(f"Đang tải {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(url, str(output_path), quiet=False)
    print(f"✓ Đã tải xong: {output_path}")


def download_with_requests(url, output_path, description=""):
    """Tải file sử dụng requests (cho URL thông thường)."""
    print(f"Đang tải {description}...")
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

    print(f"✓ Đã tải xong: {output_path}")


def download_vivos(output_dir):
    """
    Tải VIVOS dataset từ Zenodo.

    VIVOS: Vietnamese Speech Corpus - 15 giờ, 11,660 utterances
    """
    print("\n" + "="*60)
    print("Tải VIVOS Dataset (Vietnamese Speech Corpus)")
    print("="*60)

    # VIVOS direct download link từ Zenodo
    vivos_url = "https://zenodo.org/record/7068130/files/vivos.tar.gz?download=1"
    vivos_output = Path(output_dir) / "vivos.tar.gz"

    download_with_requests(vivos_url, vivos_output, "VIVOS Dataset (1.5 GB)")

    # Giải nén
    print(f"\nĐang giải nén {vivos_output}...")
    vivos_extract_dir = Path(output_dir) / "vietnamese" / "vivos"
    vivos_extract_dir.mkdir(parents=True, exist_ok=True)

    if os.name == 'nt':  # Windows
        subprocess.run(["tar", "-xzf", str(vivos_output), "-C", str(vivos_extract_dir.parent)], check=True)
    else:  # Linux/Mac
        subprocess.run(["tar", "-xzf", str(vivos_output), "-C", str(vivos_extract_dir.parent)], check=True)

    print(f"✓ Đã giải nén vào: {vivos_extract_dir.parent}")

    # Xóa file tar.gz sau khi giải nén
    vivos_output.unlink()
    print(f"✓ Đã xóa file nén")


def download_phomt(output_dir):
    """
    Tải PhoMT dataset (Vietnamese-English parallel corpus).

    PhoMT: 3.02M cặp câu VI-EN

    Lưu ý: PhoMT cần download thủ công từ GitHub vì yêu cầu đăng ký.
    Script này sẽ hướng dẫn cách tải.
    """
    print("\n" + "="*60)
    print("PhoMT Dataset (Vietnamese-English Parallel Corpus)")
    print("="*60)

    print("\nPhoMT yêu cầu download thủ công từ GitHub vì:")
    print("- Cần đồng thuận license")
    print("- File lớn (~500MB)")
    print("\nHướng dẫn:")
    print("1. Truy cập: https://github.com/VinAIResearch/PhoMT")
    print("2. Đọc và đồng ý license")
    print("3. Download dataset")
    print("4. Giải nén vào:", Path(output_dir) / "english" / "phomt")

    # Tạo thư mục đích
    phomt_dir = Path(output_dir) / "english" / "phomt"
    phomt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Đã tạo thư mục: {phomt_dir}")
    print("Sau khi download, hãy giải nén vào thư mục này.")


def download_common_voice(output_dir):
    """
    Tải Mozilla Common Voice Vietnamese dataset.

    Common Voice: ~23 giờ tiếng Việt (20,605 clips)
    """
    print("\n" + "="*60)
    print("Mozilla Common Voice Vietnamese Dataset")
    print("="*60)

    # Common Voice direct download link
    cv_url = "https://voice.mozilla.org/en/datasets"
    cv_output = Path(output_dir) / "vietnamese" / "common_voice"

    print("\nMozilla Common Voice cần download thủ công từ website:")
    print("1. Truy cập:", cv_url)
    print("2. Chọn Vietnamese language")
    print("3. Download dataset")
    print("4. Giải nén vào:", cv_output)

    cv_output.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Đã tạo thư mục: {cv_output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/raw", help="Thư mục đầu ra")
    ap.add_argument("--dataset", choices=["vivos", "phomt", "common_voice", "all"],
                    default="all", help="Dataset để tải")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Download Script cho Task 2 ASR VI + MT EN")
    print("="*60)
    print(f"Output directory: {output_dir}")

    if args.dataset in ["vivos", "all"]:
        try:
            download_vivos(output_dir)
        except Exception as e:
            print(f"✗ Lỗi khi tải VIVOS: {e}")
            print("Bạn có thể tải thủ công từ: https://zenodo.org/record/7068130")

    if args.dataset in ["phomt", "all"]:
        download_phomt(output_dir)

    if args.dataset in ["common_voice", "all"]:
        download_common_voice(output_dir)

    print("\n" + "="*60)
    print("Hoàn tất!")
    print("="*60)
    print("\nCác bước tiếp theo:")
    print("1. Nếu đã tải VIVOS: dữ liệu đã sẵn sàng trong data/raw/vietnamese/vivos/")
    print("2. Nếu cần PhoMT: tải thủ công theo hướng dẫn ở trên")
    print("3. Chạy prepare_data.py để xử lý dữ liệu:")
    print("   python -m src.data.prepare_data --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
