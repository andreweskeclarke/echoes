#!/usr/bin/env python3
"""Validate downloaded UCF101 dataset integrity and structure."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Expected UCF101 structure
EXPECTED_CLASSES = 101
EXPECTED_TOTAL_VIDEOS = 13320
EXPECTED_TRAIN_VIDEOS = 9537
EXPECTED_TEST_VIDEOS = 3783

# Key directories that should exist
REQUIRED_DIRS = [
    "UCF-101",
    "splits_01"
]

# Key annotation files that should exist
REQUIRED_ANNOTATION_FILES = [
    "splits_01/trainlist01.txt",
    "splits_01/testlist01.txt",
    "splits_01/classInd.txt"
]


def validate_directory_structure(dataset_path: Path) -> list[str]:
    """Validate that required directories exist."""
    issues = []

    for required_dir in REQUIRED_DIRS:
        dir_path = dataset_path / required_dir
        if not dir_path.exists():
            issues.append(f"Missing required directory: {required_dir}")
        elif not dir_path.is_dir():
            issues.append(f"Path exists but is not a directory: {required_dir}")

    return issues


def validate_annotation_files(dataset_path: Path) -> list[str]:
    """Validate that annotation files exist and have expected content."""
    issues = []

    for required_file in REQUIRED_ANNOTATION_FILES:
        file_path = dataset_path / required_file
        if not file_path.exists():
            issues.append(f"Missing required file: {required_file}")
        elif not file_path.is_file():
            issues.append(f"Path exists but is not a file: {required_file}")
        elif file_path.stat().st_size == 0:
            issues.append(f"File is empty: {required_file}")

    return issues


def count_video_files(dataset_path: Path) -> tuple[int, dict[str, int]]:
    """Count video files by class."""
    ucf101_dir = dataset_path / "UCF-101"
    if not ucf101_dir.exists():
        return 0, {}

    video_counts = {}
    total_videos = 0

    for class_dir in ucf101_dir.iterdir():
        if class_dir.is_dir():
            video_files = list(class_dir.glob("*.avi"))
            video_counts[class_dir.name] = len(video_files)
            total_videos += len(video_files)

    return total_videos, video_counts


def validate_class_counts(dataset_path: Path) -> list[str]:
    """Validate class counts and structure."""
    issues = []

    total_videos, video_counts = count_video_files(dataset_path)

    # Check total number of classes
    num_classes = len(video_counts)
    if num_classes != EXPECTED_CLASSES:
        issues.append(f"Expected {EXPECTED_CLASSES} classes, found {num_classes}")

    # Check total number of videos (with tolerance for slight variations)
    if abs(total_videos - EXPECTED_TOTAL_VIDEOS) > 100:
        issues.append(f"Expected ~{EXPECTED_TOTAL_VIDEOS} videos, found {total_videos}")

    # Check for empty classes
    empty_classes = [cls for cls, count in video_counts.items() if count == 0]
    if empty_classes:
        issues.append(f"Found {len(empty_classes)} empty classes: {empty_classes[:5]}...")

    return issues


def validate_split_consistency(dataset_path: Path) -> list[str]:
    """Validate that split files are consistent with actual videos."""
    issues = []

    try:
        # Read train split
        trainlist_path = dataset_path / "splits_01" / "trainlist01.txt"
        if trainlist_path.exists():
            with open(trainlist_path) as f:
                train_videos = set()
                for line in f:
                    if line.strip():
                        video_path = line.strip().split()[0]  # Remove class label
                        train_videos.add(video_path)

            logger.info(f"Found {len(train_videos)} videos in train split")

            # Check if train videos exist
            missing_train = []
            for video_path in list(train_videos)[:10]:  # Check first 10
                full_path = dataset_path / "UCF-101" / video_path
                if not full_path.exists():
                    missing_train.append(video_path)

            if missing_train:
                issues.append(f"Missing train videos: {missing_train}")

        # Read test split
        testlist_path = dataset_path / "splits_01" / "testlist01.txt"
        if testlist_path.exists():
            with open(testlist_path) as f:
                test_videos = set(line.strip() for line in f if line.strip())

            logger.info(f"Found {len(test_videos)} videos in test split")

            # Check if test videos exist
            missing_test = []
            for video_path in list(test_videos)[:10]:  # Check first 10
                full_path = dataset_path / "UCF-101" / video_path
                if not full_path.exists():
                    missing_test.append(video_path)

            if missing_test:
                issues.append(f"Missing test videos: {missing_test}")

    except Exception as e:
        issues.append(f"Error validating split consistency: {e}")

    return issues


def calculate_dataset_checksum(dataset_path: Path) -> str:
    """Calculate a checksum for the dataset structure (not full content)."""
    hasher = hashlib.md5()

    # Hash the structure and file sizes (not full content for speed)
    for annotation_file in REQUIRED_ANNOTATION_FILES:
        file_path = dataset_path / annotation_file
        if file_path.exists():
            hasher.update(f"{annotation_file}:{file_path.stat().st_size}".encode())

    # Hash class counts
    total_videos, video_counts = count_video_files(dataset_path)
    for class_name in sorted(video_counts.keys()):
        hasher.update(f"{class_name}:{video_counts[class_name]}".encode())

    return hasher.hexdigest()


def validate_dataset(dataset_path: Path) -> bool:
    """Validate the entire dataset and return True if valid."""
    logger.info(f"Validating dataset at: {dataset_path}")

    all_issues = []

    # Check directory structure
    logger.info("Checking directory structure...")
    all_issues.extend(validate_directory_structure(dataset_path))

    # Check annotation files
    logger.info("Checking annotation files...")
    all_issues.extend(validate_annotation_files(dataset_path))

    # Check class counts
    logger.info("Checking class counts and video files...")
    all_issues.extend(validate_class_counts(dataset_path))

    # Check split consistency
    logger.info("Checking split file consistency...")
    all_issues.extend(validate_split_consistency(dataset_path))

    # Calculate and report checksum
    checksum = calculate_dataset_checksum(dataset_path)
    logger.info(f"Dataset structure checksum: {checksum}")

    # Report results
    if all_issues:
        logger.error(f"Dataset validation failed with {len(all_issues)} issues:")
        for issue in all_issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("Dataset validation passed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate UCF101 dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file path",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=Path(args.log_file) if args.log_file else None
    )

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    try:
        is_valid = validate_dataset(dataset_path)
        sys.exit(0 if is_valid else 1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()