"""Unit tests for prepare_ucf101 script."""
import tempfile
from pathlib import Path

import pytest

from data.local_storage import LocalStorage
from scripts.prepare_ucf101 import create_storage_backend, prepare_ucf101


class FakeStorage:
    """Fake storage backend for testing."""

    def __init__(self):
        self.downloaded = False
        self.local_path = Path("/fake/path")
        self.cleaned_up = False

    def download_dataset(self, dataset_name, force=False):
        """Fake download."""
        self.downloaded = True
        if dataset_name != "ucf101":
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def get_local_path(self, dataset_name):
        """Return fake local path."""
        return self.local_path

    def cleanup(self):
        """Mark as cleaned up."""
        self.cleaned_up = True


class TestCreateStorageBackend:
    """Test cases for create_storage_backend function."""

    def test_create_local_storage(self):
        """Test creating local storage backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "test"
            result = create_storage_backend("local", root_dir=root_dir)

            assert isinstance(result, LocalStorage)
            assert result.root_dir == root_dir

    def test_create_unsupported_storage(self):
        """Test creating unsupported storage backend raises exception."""
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_storage_backend("unsupported")


class TestPrepareUCF101:
    """Test cases for prepare_ucf101 function."""

    def test_prepare_ucf101_calls_storage_methods(self):
        """Test prepare_ucf101 calls expected storage methods."""
        fake_storage = FakeStorage()

        # Inject fake storage by overriding create_storage_backend
        def fake_create_storage(storage_type, **kwargs):
            return fake_storage

        # Replace the function temporarily
        import scripts.prepare_ucf101
        original_create = scripts.prepare_ucf101.create_storage_backend
        scripts.prepare_ucf101.create_storage_backend = fake_create_storage

        try:
            prepare_ucf101("local", force_download=True, root_dir=Path("/test"))

            # Verify storage methods were called
            assert fake_storage.downloaded is True
            assert fake_storage.cleaned_up is True

        finally:
            # Restore original function
            scripts.prepare_ucf101.create_storage_backend = original_create

    def test_prepare_ucf101_cleans_up_on_exception(self):
        """Test prepare_ucf101 cleans up even when exception occurs."""
        fake_storage = FakeStorage()

        # Make download raise an exception
        def failing_download(dataset_name, force=False):
            fake_storage.downloaded = True
            raise Exception("Download failed")

        fake_storage.download_dataset = failing_download

        # Inject fake storage
        def fake_create_storage(storage_type, **kwargs):
            return fake_storage

        import scripts.prepare_ucf101
        original_create = scripts.prepare_ucf101.create_storage_backend
        scripts.prepare_ucf101.create_storage_backend = fake_create_storage

        try:
            with pytest.raises(Exception, match="Download failed"):
                prepare_ucf101("local", root_dir=Path("/test"))

            # Verify cleanup was still called
            assert fake_storage.cleaned_up is True

        finally:
            scripts.prepare_ucf101.create_storage_backend = original_create

    def test_prepare_ucf101_handles_storage_without_cleanup(self):
        """Test prepare_ucf101 works with storage that doesn't have cleanup."""
        class StorageWithoutCleanup:
            def download_dataset(self, dataset_name, force=False):
                pass

            def get_local_path(self, dataset_name):
                return Path("/fake/path")

        storage = StorageWithoutCleanup()

        # Inject storage
        def fake_create_storage(storage_type, **kwargs):
            return storage

        import scripts.prepare_ucf101
        original_create = scripts.prepare_ucf101.create_storage_backend
        scripts.prepare_ucf101.create_storage_backend = fake_create_storage

        try:
            # Should not raise exception
            prepare_ucf101("local", root_dir=Path("/test"))

        finally:
            scripts.prepare_ucf101.create_storage_backend = original_create