import json
from pathlib import Path
from typing import TypedDict


class DataDiskConfig(TypedDict):
    name: str
    resource_group: str
    mount_point: str


class AzureConfig(TypedDict):
    resource_group: str
    location: str
    data_disk: DataDiskConfig


def load_azure_config(config_path: Path | None = None) -> AzureConfig:
    if config_path is None:
        config_path = Path(__file__).parent.parent / ".azure-config.json"

    if not config_path.exists():
        error_msg = f"""
Azure configuration file not found: {config_path}

Please create .azure-config.json in the project root with the following format:

{{
  "resource_group": "your-resource-group-name",
  "location": "your-azure-region",
  "data_disk": {{
    "name": "your-data-disk-name",
    "resource_group": "disk-resource-group-name",
    "mount_point": "/mnt/your_data"
  }}
}}

"""
        raise FileNotFoundError(error_msg)

    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}") from e
