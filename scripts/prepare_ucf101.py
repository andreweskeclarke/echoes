#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from torchvision.datasets import UCF101
from tqdm import tqdm

def prepare_ucf101(data_dir: Path, force_download: bool = False) -> None:
    """Download and prepare UCF101 dataset using torchvision."""
    print("Downloading and preparing UCF101 dataset...")
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download train split (this will download the full dataset)
    train_dataset = UCF101(
        root=str(data_dir),
        annotation_path=str(data_dir / "splits"),
        frames_per_clip=16,
        train=True,
        download=True,
        num_workers=4
    )
    
    # Download test split
    test_dataset = UCF101(
        root=str(data_dir),
        annotation_path=str(data_dir / "splits"),
        frames_per_clip=16,
        train=False,
        download=True,
        num_workers=4
    )
    
    print(f"Dataset preparation complete!")
    print(f"Number of training videos: {len(train_dataset)}")
    print(f"Number of test videos: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"\nClasses: {train_dataset.classes}")

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
    
    try:
        prepare_ucf101(data_dir, args.force_download)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 