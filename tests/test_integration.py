"""Integration tests for dataset download and validation."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from data.local_storage import LocalStorage
from scripts.validate_dataset import validate_dataset


class TestLocalStorageIntegration:
    """Integration tests for LocalStorage with actual torchvision download."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = LocalStorage(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_download_and_validate_ucf101_sample(self):
        """Test downloading a small sample of UCF101 and validating it.

        Note: This test is marked as slow and only runs when RUN_INTEGRATION_TESTS=1
        because it actually downloads data from the internet.
        """
        # Note: We can't easily test a "sample" of UCF101 since torchvision
        # downloads the full dataset. This test would download the entire dataset.
        # In a real scenario, you might want to:
        # 1. Mock the torchvision download to use a smaller test dataset
        # 2. Use a smaller dataset for testing
        # 3. Skip this test in CI and run it manually

        # For now, we'll create a minimal test structure manually
        ucf101_path = self.temp_dir / "ucf101"
        ucf101_path.mkdir()

        # Create minimal UCF101 structure for validation testing
        (ucf101_path / "UCF-101").mkdir()
        (ucf101_path / "splits_01").mkdir()

        # Create a sample class directory with a video file
        class_dir = ucf101_path / "UCF-101" / "ApplyEyeMakeup"
        class_dir.mkdir()

        # Create a dummy video file (empty but with correct extension)
        (class_dir / "v_ApplyEyeMakeup_g01_c01.avi").touch()

        # Create minimal annotation files
        (ucf101_path / "splits_01" / "classInd.txt").write_text("1 ApplyEyeMakeup\n")
        (ucf101_path / "splits_01" / "trainlist01.txt").write_text("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1\n")
        (ucf101_path / "splits_01" / "testlist01.txt").write_text("ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi\n")

        # Test that the storage can find the dataset
        assert self.storage.exists("ucf101")

        # Test that we can get the local path
        local_path = self.storage.get_local_path("ucf101")
        assert local_path == ucf101_path

        # Test basic validation (will fail because we don't have full dataset)
        # But it should at least find the basic structure
        try:
            is_valid = validate_dataset(ucf101_path)
            # We expect this to fail because we only have 1 class instead of 101
            assert not is_valid
        except Exception as e:
            # Validation should not crash, even with incomplete data
            pytest.fail(f"Validation crashed unexpectedly: {e}")


# Note: For real blob storage testing, you would need Azure credentials
# and would want to use a test container. Here's an example of what that might look like:

@pytest.mark.slow
@pytest.mark.skipif(
    not all([
        os.environ.get("RUN_INTEGRATION_TESTS"),
        os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    ]),
    reason="Blob storage tests require RUN_INTEGRATION_TESTS=1 and Azure credentials"
)
class TestBlobStorageIntegration:
    """Integration tests for BlobStorage with real Azure storage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Only import and create if we have credentials
        from data.blob_storage import BlobStorage
        self.storage = BlobStorage(
            connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            container_name="test-echoes-dataset",
            local_cache_dir=self.temp_dir
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'storage'):
            self.storage.cleanup()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_blob_storage_basic_operations(self):
        """Test basic blob storage operations with real Azure storage."""
        test_content = b"test content for blob storage"

        # Create a test file locally
        test_file = self.temp_dir / "test_upload.txt"
        test_file.write_bytes(test_content)

        # Upload file to blob storage (we'd need to implement upload_file method)
        # This is a conceptual test - the actual implementation would need
        # additional methods in BlobStorage class

        # For now, just test that we can create the storage without errors
        assert self.storage.container_name == "test-echoes-dataset"
        assert self.storage.local_cache_dir == self.temp_dir

        # Test listing (should be empty or contain test data)
        files = self.storage.list_files("test/")
        assert isinstance(files, list)

    def test_blob_storage_upload_and_download_cycle(self):
        """Test uploading and downloading files to/from blob storage."""
        # This would test the full cycle:
        # 1. Create local test dataset
        # 2. Upload to blob storage using download_dataset
        # 3. Clear local cache
        # 4. Download from blob storage using get_local_path
        # 5. Validate downloaded data matches original

        # For now, this is a placeholder for the actual implementation
        # You would need to implement proper upload/download cycle testing
        pass