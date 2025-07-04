#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.blob_storage import BlobStorage
from data.local_storage import LocalStorage
from data.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def create_storage_backend(storage_type: str, **kwargs) -> object:
    """Create storage backend based on type."""
    if storage_type == "local":
        return LocalStorage(kwargs.get("root_dir", Path("data")))
    elif storage_type == "blob":
        return BlobStorage(
            account_name=kwargs.get("account_name"),
            account_key=kwargs.get("account_key"),
            connection_string=kwargs.get("connection_string"),
            container_name=kwargs.get("container_name", "datasets"),
            local_cache_dir=kwargs.get("local_cache_dir"),
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def prepare_ucf101(
    storage_type: str = "local",
    force_download: bool = False,
    **storage_kwargs
) -> None:
    """Download and prepare UCF101 dataset using specified storage backend."""
    logger.info(f"Preparing UCF101 dataset with {storage_type} storage...")

    # Create storage backend
    storage = create_storage_backend(storage_type, **storage_kwargs)

    try:
        # Download dataset
        storage.download_dataset("ucf101", force=force_download)

        # Get local path for verification
        local_path = storage.get_local_path("ucf101")
        logger.info(f"Dataset available at: {local_path}")

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise
    finally:
        # Clean up if needed
        if hasattr(storage, 'cleanup'):
            storage.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Prepare UCF101 dataset")
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "blob"],
        default="local",
        help="Storage backend to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory for local storage",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if files exist",
    )

    # Azure Blob Storage arguments
    parser.add_argument(
        "--account-name",
        type=str,
        help="Azure storage account name",
    )
    parser.add_argument(
        "--account-key",
        type=str,
        help="Azure storage account key",
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        help="Azure storage connection string",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="datasets",
        help="Azure blob container name",
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        help="Local cache directory for blob storage",
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

    # Check for required Azure arguments
    if args.storage_type == "blob":
        if not args.connection_string and not (args.account_name and args.account_key):
            # Try to get from environment
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

            if connection_string:
                args.connection_string = connection_string
            elif account_name and account_key:
                args.account_name = account_name
                args.account_key = account_key
            else:
                logger.error("Error: For blob storage, must provide either:")
                logger.error("  --connection-string or --account-name and --account-key")
                logger.error("  Or set AZURE_STORAGE_CONNECTION_STRING environment variable")
                sys.exit(1)

    try:
        prepare_ucf101(
            storage_type=args.storage_type,
            force_download=args.force_download,
            root_dir=Path(args.data_dir),
            account_name=args.account_name,
            account_key=args.account_key,
            connection_string=args.connection_string,
            container_name=args.container_name,
            local_cache_dir=Path(args.local_cache_dir) if args.local_cache_dir else None,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
