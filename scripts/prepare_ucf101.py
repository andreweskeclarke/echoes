#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file with progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        with tqdm(total=total_files, desc="Extracting") as pbar:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)

def prepare_ucf101(data_dir: Path, force_download: bool = False) -> None:
    """Download and prepare UCF101 dataset."""
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset if needed
    zip_path = data_dir / "UCF101.rar"
    if not zip_path.exists() or force_download:
        print("Downloading UCF101 dataset...")
        download_file(
            "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
            zip_path
        )
    
    # Download train/test splits
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'test']:
        split_file = splits_dir / f"{split}list01.txt"
        if not split_file.exists() or force_download:
            print(f"Downloading {split} split...")
            download_file(
                f"https://www.crcv.ucf.edu/data/UCF101/{split}list01.txt",
                split_file
            )
    
    # Extract dataset if needed
    extract_dir = data_dir / "UCF101"
    if not extract_dir.exists() or force_download:
        print("Extracting UCF101 dataset...")
        # Note: This requires unrar to be installed
        os.system(f"unrar x {zip_path} {data_dir}")
    
    print("Dataset preparation complete!")

def main():
    parser = argparse.ArgumentParser(description="Prepare UCF101 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ucf101",
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if files exist"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    prepare_ucf101(data_dir, args.force_download)

if __name__ == "__main__":
    main() 