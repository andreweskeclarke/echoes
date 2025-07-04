"""Unit tests for BlobStorage class."""
import shutil
import tempfile
from pathlib import Path

import pytest

from data.blob_storage import BlobStorage


class FakeBlobServiceClient:
    """Fake blob service client for testing without Azure."""

    def __init__(self):
        self.containers = {}

    def create_container(self, name):
        """Create a fake container."""
        self.containers[name] = {}

    def get_container_client(self, container):
        """Get a fake container client."""
        return FakeContainerClient(self.containers.get(container, {}))

    def get_blob_client(self, container, blob):
        """Get a fake blob client."""
        return FakeBlobClient(self.containers.get(container, {}), blob)


class FakeContainerClient:
    """Fake container client for testing."""

    def __init__(self, blobs):
        self.blobs = blobs

    def list_blobs(self, name_starts_with=None, max_results=None):
        """List fake blobs."""
        if name_starts_with:
            matching = [FakeBlob(name) for name in self.blobs
                       if name.startswith(name_starts_with)]
        else:
            matching = [FakeBlob(name) for name in self.blobs]

        if max_results:
            return matching[:max_results]
        return matching

    def get_blob_client(self, blob_name):
        """Get a fake blob client."""
        return FakeBlobClient(self.blobs, blob_name)


class FakeBlob:
    """Fake blob for testing."""

    def __init__(self, name):
        self.name = name


class FakeBlobClient:
    """Fake blob client for testing."""

    def __init__(self, blobs, blob_name):
        self.blobs = blobs
        self.blob_name = blob_name

    def upload_blob(self, data, overwrite=False):
        """Fake upload blob."""
        self.blobs[self.blob_name] = data.read()

    def download_blob(self):
        """Fake download blob."""
        return FakeDownloadResult(self.blobs.get(self.blob_name, b""))


class FakeDownloadResult:
    """Fake download result for testing."""

    def __init__(self, data):
        self.data = data

    def readall(self):
        """Return fake data."""
        return self.data


class TestBlobStorage:
    """Test cases for BlobStorage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fake_client = FakeBlobServiceClient()

        # Create BlobStorage with fake client
        self.storage = BlobStorage.__new__(BlobStorage)
        self.storage.blob_service_client = self.fake_client
        self.storage.container_name = "test_container"
        self.storage.local_cache_dir = self.temp_dir

        # Create container
        self.fake_client.create_container("test_container")

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_exists_returns_false_when_no_blob_exists(self):
        """Test exists returns False when no blob exists."""
        result = self.storage.exists("test_path")
        assert result is False

    def test_exists_returns_true_when_blob_exists(self):
        """Test exists returns True when blob exists."""
        # Add a fake blob
        self.fake_client.containers["test_container"]["test_path/file.txt"] = b"data"

        result = self.storage.exists("test_path")
        assert result is True

    def test_list_files(self):
        """Test list_files returns blob names."""
        # Add fake blobs
        self.fake_client.containers["test_container"]["test_path/file1.txt"] = b"data1"
        self.fake_client.containers["test_container"]["test_path/file2.txt"] = b"data2"

        result = self.storage.list_files("test_path")
        assert "test_path/file1.txt" in result
        assert "test_path/file2.txt" in result

    def test_cleanup_removes_cache_directory(self):
        """Test cleanup removes local cache directory."""
        # Verify cache directory exists
        assert self.temp_dir.exists()

        # Call cleanup
        self.storage.cleanup()

        # Verify cache directory is removed
        assert not self.temp_dir.exists()

    def test_get_local_path_existing_cache(self):
        """Test get_local_path returns cached path when cache is valid."""
        # Create cache directory with some content
        cache_path = self.temp_dir / "ucf101"
        cache_path.mkdir()
        (cache_path / "test_file.txt").write_text("test")

        result = self.storage.get_local_path("ucf101")
        assert result == cache_path

    def test_get_local_path_invalid_cache_syncs_from_blob(self):
        """Test get_local_path syncs from blob storage when cache is invalid."""
        # Add fake blob data
        self.fake_client.containers["test_container"]["ucf101/test_file.txt"] = b"test content"

        result = self.storage.get_local_path("ucf101")

        # Verify sync was performed
        assert result == self.temp_dir / "ucf101"
        assert (self.temp_dir / "ucf101" / "test_file.txt").exists()
        assert (self.temp_dir / "ucf101" / "test_file.txt").read_bytes() == b"test content"

    def test_get_local_path_unsupported_dataset(self):
        """Test get_local_path raises exception for unsupported dataset."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            self.storage.get_local_path("unsupported_dataset")

    def test_download_dataset_unsupported(self):
        """Test download_dataset raises exception for unsupported dataset."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            self.storage.download_dataset("unsupported_dataset")

    def test_sync_from_blob_storage(self):
        """Test _sync_from_blob_storage downloads all blobs."""
        # Add fake blob data
        self.fake_client.containers["test_container"]["ucf101/file1.txt"] = b"content1"
        self.fake_client.containers["test_container"]["ucf101/subdir/file2.txt"] = b"content2"

        # Sync from blob storage
        local_path = self.temp_dir / "test_dataset"
        self.storage._sync_from_blob_storage("ucf101", local_path)

        # Verify files were created
        assert (local_path / "file1.txt").exists()
        assert (local_path / "file1.txt").read_bytes() == b"content1"
        assert (local_path / "subdir" / "file2.txt").exists()
        assert (local_path / "subdir" / "file2.txt").read_bytes() == b"content2"