import json
from pathlib import Path

import pytest

from remote.config import load_azure_config


class TestLoadAzureConfig:
    def test_valid_config(self, tmp_path: Path):
        config_file = tmp_path / ".azure-config.json"
        config_data = {
            "resource_group": "my-rg",
            "location": "eastus",
            "data_disk": {
                "name": "my-disk",
                "resource_group": "disk-rg",
                "mount_point": "/mnt/data",
            },
        }
        config_file.write_text(json.dumps(config_data))

        result = load_azure_config(config_file)

        assert result["resource_group"] == "my-rg"
        assert result["location"] == "eastus"
        assert result["data_disk"]["name"] == "my-disk"
        assert result["data_disk"]["mount_point"] == "/mnt/data"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        config_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_azure_config(config_file)

        assert "Azure configuration file not found" in str(exc_info.value)
        assert str(config_file) in str(exc_info.value)

    def test_invalid_json_raises_value_error(self, tmp_path: Path):
        config_file = tmp_path / ".azure-config.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ValueError) as exc_info:
            load_azure_config(config_file)

        assert "Invalid JSON" in str(exc_info.value)

    def test_empty_file_raises_value_error(self, tmp_path: Path):
        config_file = tmp_path / ".azure-config.json"
        config_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            load_azure_config(config_file)

        assert "Invalid JSON" in str(exc_info.value)
