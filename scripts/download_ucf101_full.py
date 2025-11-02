#!/usr/bin/env python3
"""
Complete UCF101 dataset download and extraction script.
Downloads videos, splits, and validates the complete dataset.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from data.logging_config import setup_logging


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space for the dataset."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)

        logging.info(f"Available disk space: {free_gb:.1f}GB")
        logging.info(f"Required disk space: {required_gb:.1f}GB")

        if free_gb < required_gb:
            logging.error(
                f"Insufficient disk space! Need {required_gb:.1f}GB, "
                f"have {free_gb:.1f}GB"
            )
            return False

        return True
    except Exception as e:
        logging.error(f"Error checking disk space: {e}")
        return False


def check_dependencies() -> bool:
    """Check if required tools are installed."""
    required_tools = ["unrar", "unzip", "wget"]
    missing = []

    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        logging.error(f"Missing required tools: {', '.join(missing)}")
        logging.info("Install with: sudo apt install -y " + " ".join(missing))
        return False

    return True


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a file with progress reporting."""
    try:
        logging.info(f"Downloading {description} from {url}")
        logging.info(f"Destination: {output_path}")

        # Use wget for better progress reporting and SSL handling
        cmd = [
            "wget",
            "--no-check-certificate",  # Handle SSL cert issues
            "--progress=bar:force",
            url,
            "-O",
            str(output_path),
        ]

        result = subprocess.run(cmd, check=False, capture_output=False, text=True)

        if result.returncode == 0:
            logging.info(f"Successfully downloaded {description}")
            return True
        else:
            logging.error(f"Failed to download {description}")
            return False

    except Exception as e:
        logging.error(f"Download failed: {e}")
        return False


def extract_rar(rar_path: Path, extract_dir: Path) -> bool:
    """Extract RAR archive."""
    try:
        logging.info(f"Extracting {rar_path.name}...")

        cmd = ["unrar", "x", str(rar_path), str(extract_dir)]
        result = subprocess.run(
            cmd, check=False, cwd=extract_dir.parent, capture_output=True, text=True
        )

        if result.returncode == 0:
            logging.info("RAR extraction completed successfully")
            return True
        else:
            logging.warning("RAR extraction completed with warnings")
            logging.warning(f"stdout: {result.stdout}")
            logging.warning(f"stderr: {result.stderr}")
            # Return True even with warnings if extraction mostly worked
            return True

    except Exception as e:
        logging.error(f"RAR extraction failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract ZIP archive."""
    try:
        logging.info(f"Extracting {zip_path.name}...")

        cmd = ["unzip", str(zip_path), "-d", str(extract_dir)]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("ZIP extraction completed successfully")
            return True
        else:
            logging.error(f"ZIP extraction failed: {result.stderr}")
            return False

    except Exception as e:
        logging.error(f"ZIP extraction failed: {e}")
        return False


def count_videos(ucf101_dir: Path) -> int:
    """Count total video files in the dataset."""
    try:
        video_files = list(ucf101_dir.rglob("*.avi"))
        return len(video_files)
    except Exception as e:
        logging.error(f"Error counting videos: {e}")
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract complete UCF101 dataset"
    )
    parser.add_argument("data_dir", type=Path, help="Directory to download dataset to")
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Skip disk space check (dangerous!)",
    )
    parser.add_argument(
        "--required-space-gb",
        type=float,
        default=70.0,
        help="Required disk space in GB (default: 70)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def download_archives(dataset_dir, video_url, splits_url):
    """Download video and splits archives."""
    rar_path = dataset_dir / "UCF101.rar"
    zip_path = dataset_dir / "UCF101TrainTestSplits.zip"

    if not rar_path.exists():
        if not download_file(video_url, rar_path, "UCF101 videos (6.5GB)"):
            return False, rar_path, zip_path
    else:
        logging.info("UCF101.rar already exists, skipping download")

    if not zip_path.exists():
        if not download_file(splits_url, zip_path, "UCF101 splits (111KB)"):
            return False, rar_path, zip_path
    else:
        logging.info("UCF101 splits already exist, skipping download")

    return True, rar_path, zip_path


def extract_archives(dataset_dir, rar_path, zip_path):
    """Extract both archives."""
    ucf101_extracted = dataset_dir / "UCF-101"
    if not ucf101_extracted.exists():
        if not extract_rar(rar_path, dataset_dir):
            logging.error("RAR extraction failed!")
            return False, ucf101_extracted, None

    splits_dir = dataset_dir / "splits_01"
    if not splits_dir.exists():
        if not extract_zip(zip_path, dataset_dir):
            logging.error("ZIP extraction failed!")
            return False, ucf101_extracted, splits_dir
    else:
        logging.info("Splits directory already exists, skipping extraction")

    return True, ucf101_extracted, splits_dir


def report_and_cleanup(ucf101_extracted, splits_dir, rar_path, zip_path):
    """Report results and optionally cleanup archives."""
    video_count = count_videos(ucf101_extracted)
    logging.info("Dataset extraction complete!")
    logging.info(f"Total videos found: {video_count}")
    logging.info("Expected videos: ~13,320")

    min_expected_videos = 10000
    if video_count < min_expected_videos:
        logging.warning("Video count seems low - extraction may be incomplete")

    cleanup_choice = input("Delete downloaded archives to save space? (y/N): ").lower()
    if cleanup_choice == "y":
        if rar_path.exists():
            rar_path.unlink()
            logging.info("Deleted UCF101.rar")
        if zip_path.exists():
            zip_path.unlink()
            logging.info("Deleted splits.zip")

    logging.info("UCF101 dataset download and extraction completed successfully!")
    logging.info(f"Dataset location: {ucf101_extracted.parent}")
    logging.info(f"Videos: {ucf101_extracted}")
    logging.info(f"Splits: {splits_dir}")


def main():
    args = parse_args()

    # Setup logging
    log_file = args.data_dir / "download.log"
    setup_logging(args.log_level, log_file)

    logging.info("Starting UCF101 complete dataset download")
    logging.info(f"Target directory: {args.data_dir}")

    # Create directory structure
    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.data_dir / "ucf101"
    dataset_dir.mkdir(exist_ok=True)

    # Check disk space
    if not args.skip_space_check:
        if not check_disk_space(args.data_dir, args.required_space_gb):
            logging.error("Aborting due to insufficient disk space")
            sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        logging.error("Aborting due to missing dependencies")
        sys.exit(1)

    # Download files
    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    splits_url = (
        "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    )

    success, rar_path, zip_path = download_archives(dataset_dir, video_url, splits_url)
    if not success:
        logging.error("Downloads failed!")
        sys.exit(1)

    success, ucf101_extracted, splits_dir = extract_archives(
        dataset_dir, rar_path, zip_path
    )
    if not success:
        sys.exit(1)

    report_and_cleanup(ucf101_extracted, splits_dir, rar_path, zip_path)


if __name__ == "__main__":
    main()
