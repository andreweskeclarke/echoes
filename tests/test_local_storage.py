"""Unit tests for LocalStorage class."""
import shutil
import tempfile
from pathlib import Path

import pytest

from data.local_storage import LocalStorage


class TestLocalStorage:
    """Test cases for LocalStorage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = LocalStorage(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init_creates_root_directory(self):
        """Test that initialization creates root directory."""
        new_temp_dir = self.temp_dir / "new_storage"
        assert not new_temp_dir.exists()

        storage = LocalStorage(new_temp_dir)
        assert new_temp_dir.exists()
        assert storage.root_dir == new_temp_dir

    def test_exists_with_missing_path(self):
        """Test exists returns False for non-existent path."""
        assert not self.storage.exists("nonexistent_dataset")

    def test_exists_with_existing_path(self):
        """Test exists returns True for existing path."""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()

        assert self.storage.exists("test_dataset")

    def test_list_files_empty_directory(self):
        """Test list_files returns empty list for empty directory."""
        dataset_path = self.temp_dir / "empty_dataset"
        dataset_path.mkdir()

        files = self.storage.list_files("empty_dataset")
        assert files == []

    def test_list_files_with_files(self):
        """Test list_files returns all files in directory."""
        dataset_path = self.temp_dir / "test_dataset"
        dataset_path.mkdir()

        # Create test files
        (dataset_path / "file1.txt").write_text("test1")
        (dataset_path / "file2.txt").write_text("test2")
        (dataset_path / "subdir").mkdir()
        (dataset_path / "subdir" / "file3.txt").write_text("test3")

        files = self.storage.list_files("test_dataset")
        assert len(files) == 3
        assert any("file1.txt" in f for f in files)
        assert any("file2.txt" in f for f in files)
        assert any("file3.txt" in f for f in files)

    def test_list_files_nonexistent_path(self):
        """Test list_files returns empty list for non-existent path."""
        files = self.storage.list_files("nonexistent")
        assert files == []

    def test_get_local_path_nonexistent_dataset(self):
        """Test get_local_path raises exception for non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            self.storage.get_local_path("ucf101")

    def test_get_local_path_existing_dataset(self):
        """Test get_local_path returns correct path for existing dataset."""
        dataset_path = self.temp_dir / "ucf101"
        dataset_path.mkdir()

        result = self.storage.get_local_path("ucf101")
        assert result == dataset_path

    def test_get_local_path_unsupported_dataset(self):
        """Test get_local_path raises exception for unsupported dataset."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            self.storage.get_local_path("unsupported_dataset")

    def test_cleanup_does_nothing(self):
        """Test cleanup method does nothing for local storage."""
        # Should not raise any exceptions
        self.storage.cleanup()

    def test_download_dataset_unsupported(self):
        """Test download_dataset raises exception for unsupported dataset."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            self.storage.download_dataset("unsupported_dataset")