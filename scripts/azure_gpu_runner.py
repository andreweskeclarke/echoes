#!/usr/bin/env python3
"""
Azure GPU Runner - Automates running ML experiments on Azure GPU VMs.

This script:
1. Spins up an Azure GPU VM with auto-destroy tag
2. Sets up the environment and codebase
3. Runs the specified experiment
4. Downloads results
5. Cleans up the VM

Usage:
    python scripts/azure_gpu_runner.py experiments/train_simple.py \\
        --vm-size Standard_NC6s_v3
"""

import argparse
import atexit
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from data.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def load_azure_config():
    """Load Azure configuration from .azure-config.json file."""
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


def _default_ssh_key() -> str:
    return os.path.expanduser("~/.ssh/id_ed25519.pub")


@dataclass
class VMConfig:
    resource_group: str
    vm_name: str
    vm_size: str = "Standard_D2as_v4"
    location: str = "eastus"
    auto_destroy_hours: int = 4
    ssh_key_path: str = field(default_factory=_default_ssh_key)
    vnet_name: str = "headquarters-vnet"
    subnet_name: str = "default"


class AzureGPURunner:
    def __init__(self, config):
        self.config = config
        self.vm_ip = None
        self._vm_created = False

    def _run_az_command(
        self, cmd: list, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run Azure CLI command with error handling."""
        logger.info(f"Running: az {' '.join(cmd)}")
        result = subprocess.run(
            ["az", *cmd], check=False, capture_output=capture_output, text=True
        )

        if result.returncode != 0:
            logger.error(f"Azure CLI command failed: {result.stderr}")
            raise RuntimeError(f"Azure CLI command failed: {result.stderr}")

        return result

    def _run_ssh_command(
        self, cmd: str, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run command on the remote VM via SSH."""
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"aclarke@{self.vm_ip}",
            cmd,
        ]
        logger.info(f"SSH: {cmd}")
        result = subprocess.run(
            ssh_cmd, check=False, capture_output=capture_output, text=True
        )

        if result.returncode != 0 and capture_output:
            logger.error(f"SSH command failed: {result.stderr}")

        return result

    def create_vm(self) -> str:
        """Create Azure GPU VM with auto-destroy tag and return its IP address."""
        logger.info(
            f"Creating VM {self.config.vm_name} in resource group "
            f"{self.config.resource_group}"
        )

        # Calculate auto-destroy time (current time + hours in UTC)
        auto_destroy_time = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() + self.config.auto_destroy_hours * 3600),
        )

        # Create VM with auto-destroy tag, no public IP (use private networking)
        create_cmd = [
            "vm",
            "create",
            "--resource-group",
            self.config.resource_group,
            "--name",
            self.config.vm_name,
            "--image",
            "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest",
            "--size",
            self.config.vm_size,
            "--admin-username",
            "aclarke",
            "--ssh-key-values",
            self.config.ssh_key_path,
            "--location",
            self.config.location,
            "--vnet-name",
            self.config.vnet_name,
            "--subnet",
            self.config.subnet_name,
            "--public-ip-address",
            "",
            "--tags",
            f"AutoShutdownTime={auto_destroy_time}",
            "CreatedBy=echoes-gpu-runner",
            "--output",
            "json",
        ]

        result = self._run_az_command(create_cmd)
        vm_info = json.loads(result.stdout)
        self.vm_ip = vm_info["privateIpAddress"]
        self._vm_created = True

        logger.info(f"VM created successfully. IP: {self.vm_ip}")
        logger.info(f"VM will auto-destroy at: {auto_destroy_time} UTC")

        # Wait for VM to be ready
        logger.info("Waiting for VM to be ready...")
        time.sleep(60)  # Give VM time to boot

        # Test SSH connection
        max_retries = 10
        for i in range(max_retries):
            result = self._run_ssh_command("echo 'Connection test'")
            if result.returncode == 0:
                logger.info("SSH connection established")
                break
            else:
                if i == max_retries - 1:
                    raise RuntimeError(
                        f"Could not establish SSH connection after {max_retries} "
                        "attempts"
                    )
                logger.info(f"SSH attempt {i + 1} failed, retrying in 30s...")
                time.sleep(30)

        return self.vm_ip

    def setup_environment(self, data_dir: str):
        """Set up the conda environment and codebase on the VM."""
        logger.info("Setting up environment on VM...")
        logger.info("Uploading codebase...")
        subprocess.run(
            [
                "rsync",
                "-avz",
                "--exclude=.git",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=.pytest_cache",
                f"{Path(__file__).parent.parent}/",
                f"aclarke@{self.vm_ip}:~/echoes/",
            ],
            check=True,
        )

        # Set up auto-shutdown script as backup
        auto_shutdown_script = f"""#!/bin/bash
echo "Setting up auto-shutdown in {self.config.auto_destroy_hours} hours..."
(sleep {self.config.auto_destroy_hours * 3600} && sudo shutdown -h now) &
"""

        setup_commands = [
            f"echo '{auto_shutdown_script}' > ~/auto_shutdown.sh",
            "chmod +x ~/auto_shutdown.sh",
            "nohup ~/auto_shutdown.sh > ~/auto_shutdown.log 2>&1 &",
            "cd ~/echoes",
            f"mkdir -p {data_dir}/logs {data_dir}/mlruns {data_dir}/tfruns",
            "rm -rf ~/echoes/logs ~/echoes/mlruns ~/echoes/tfruns",
            f"ln -sf {data_dir}/logs ~/echoes/logs",
            f"ln -sf {data_dir}/mlruns ~/echoes/mlruns",
            f"ln -sf {data_dir}/tfruns ~/echoes/tfruns",
            (
                "wget "
                "https://github.com/conda-forge/miniforge/releases/latest/download/"
                "Miniforge3-Linux-x86_64.sh "
                "-O miniforge.sh"
            ),
            "bash miniforge.sh -b -p $HOME/miniforge",
            "rm miniforge.sh",
            (
                "cd ~/echoes && export PATH=$HOME/miniforge/bin:$PATH && "
                "mamba env create -f environment.yml -y 2>&1"
            ),
        ]

        for cmd in setup_commands:
            result = self._run_ssh_command(cmd, capture_output=False)
            if result.returncode != 0:
                raise RuntimeError(f"Environment setup failed at command: {cmd}")

        logger.info("Environment setup completed")

    def copy_dataset(
        self, data_dir: str = "/tmp/echoes_data", source_path: str = "/mnt/echoes_data"
    ):
        """Copy UCF101 dataset from headquarters to remote VM via push-based rsync."""
        logger.info(f"Copying dataset from {source_path} to {self.vm_ip}:{data_dir}...")

        result = self._run_ssh_command(f"mkdir -p {data_dir}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create directory {data_dir} on remote VM")

        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e",
            "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            f"{source_path}/ucf101/",
            f"aclarke@{self.vm_ip}:{data_dir}/ucf101/",
        ]
        logger.info(f"Running: {' '.join(rsync_cmd)}")
        result = subprocess.run(rsync_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError("Dataset copy failed")

        logger.info("Dataset copy completed")

    def _get_vm_ip(self, vm_name: str, resource_group: str) -> str:
        """Get private IP of a VM (for VNet communication)."""
        result = self._run_az_command(
            [
                "vm",
                "show",
                "--resource-group",
                resource_group,
                "--name",
                vm_name,
                "--show-details",
                "--query",
                "privateIps",
                "--output",
                "tsv",
            ]
        )
        return result.stdout.strip()

    def run_experiment(
        self, experiment_script: str, data_dir: str = "/tmp/echoes_data"
    ) -> dict[str, Any]:
        """Run the experiment script on the VM."""
        logger.info(f"Running experiment: {experiment_script}")

        experiment_cmd = f"""
        cd ~/echoes &&
        export PATH=$HOME/miniforge/bin:$PATH &&
        source $HOME/miniforge/etc/profile.d/conda.sh &&
        conda activate echoes &&
        export CUDA_VISIBLE_DEVICES=0 &&
        sed -i 's|/mnt/echoes_data|{data_dir}|g' {experiment_script} &&
        python {experiment_script}
        """

        result = self._run_ssh_command(experiment_cmd, capture_output=False)

        if result.returncode != 0:
            raise RuntimeError(
                f"Experiment failed with return code: {result.returncode}"
            )
        else:
            logger.info("Experiment completed successfully")

    def download_results(
        self, local_results_dir: str = "/mnt/echoes_data/azure_results"
    ):
        """Download experiment results from the VM."""
        logger.info("Downloading results...")

        Path(local_results_dir).mkdir(exist_ok=True)

        get_latest_run_cmd = (
            "ls -t ~/echoes/mlruns/*/*/meta.yaml 2>/dev/null | head -1 | "
            "cut -d'/' -f6 || echo 'no_runs'"
        )
        result = self._run_ssh_command(get_latest_run_cmd)

        if result.returncode == 0 and result.stdout.strip() != "no_runs":
            latest_run_id = result.stdout.strip()
            experiment_id_cmd = (
                "ls -t ~/echoes/mlruns/*/*/meta.yaml 2>/dev/null | head -1 | "
                "cut -d'/' -f5 || echo 'no_exp'"
            )
            exp_result = self._run_ssh_command(experiment_id_cmd)

            if exp_result.returncode == 0 and exp_result.stdout.strip() != "no_exp":
                experiment_id = exp_result.stdout.strip()
                rsync_cmd = (
                    f"rsync -avz aclarke@{self.vm_ip}:~/echoes/mlruns/"
                    f"{experiment_id}/{latest_run_id}/ "
                    f"{local_results_dir}/mlruns/{experiment_id}/{latest_run_id}/"
                )

                try:
                    subprocess.run(rsync_cmd, shell=True, check=True)
                    logger.info(
                        f"Downloaded experiment {experiment_id}, run {latest_run_id}"
                    )
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to download with command: {rsync_cmd}")
            else:
                logger.warning("Could not determine experiment ID")
        else:
            logger.warning("No MLflow runs found to download")

        try:
            logs_cmd = (
                f"rsync -avz aclarke@{self.vm_ip}:~/echoes/logs/ "
                f"{local_results_dir}/logs/"
            )
            subprocess.run(logs_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            logger.warning("Failed to download logs")

        logger.info(f"Results downloaded to {local_results_dir}")

    def _get_vm_resources(self) -> dict[str, str]:
        """Get associated resource IDs for the VM before deletion."""
        try:
            result = self._run_az_command(
                [
                    "vm",
                    "show",
                    "--resource-group",
                    self.config.resource_group,
                    "--name",
                    self.config.vm_name,
                    "--output",
                    "json",
                ]
            )
            vm_info = json.loads(result.stdout)

            resources = {}
            if vm_info.get("storageProfile", {}).get("osDisk", {}).get("managedDisk"):
                resources["os_disk"] = vm_info["storageProfile"]["osDisk"]["name"]

            if vm_info.get("networkProfile", {}).get("networkInterfaces"):
                nic_id = vm_info["networkProfile"]["networkInterfaces"][0]["id"]
                resources["nic"] = nic_id.split("/")[-1]

                nic_result = self._run_az_command(
                    ["network", "nic", "show", "--ids", nic_id, "--output", "json"]
                )
                nic_info = json.loads(nic_result.stdout)
                if nic_info.get("networkSecurityGroup"):
                    nsg_id = nic_info["networkSecurityGroup"]["id"]
                    resources["nsg"] = nsg_id.split("/")[-1]

            return resources
        except Exception as e:
            logger.warning(f"Could not get VM resources: {e}")
            return {}

    def cleanup(self):
        """Delete the VM and associated resources (NIC, NSG, disk)."""
        if not self._vm_created:
            logger.info("No VM to cleanup")
            return

        logger.info(f"Deleting VM {self.config.vm_name} (aclarke@{self.vm_ip})")

        resources = self._get_vm_resources()

        try:
            self._run_az_command(
                [
                    "vm",
                    "delete",
                    "--resource-group",
                    self.config.resource_group,
                    "--name",
                    self.config.vm_name,
                    "--yes",
                ]
            )
            logger.info("VM deleted successfully")
            self._vm_created = False
        except Exception as e:
            logger.error(f"Failed to delete VM: {e}")
            return

        if resources.get("nic"):
            try:
                self._run_az_command(
                    [
                        "network",
                        "nic",
                        "delete",
                        "--resource-group",
                        self.config.resource_group,
                        "--name",
                        resources["nic"],
                    ]
                )
                logger.info(f"NIC {resources['nic']} deleted")
            except Exception as e:
                logger.warning(f"Failed to delete NIC: {e}")

        if resources.get("nsg"):
            try:
                self._run_az_command(
                    [
                        "network",
                        "nsg",
                        "delete",
                        "--resource-group",
                        self.config.resource_group,
                        "--name",
                        resources["nsg"],
                    ]
                )
                logger.info(f"NSG {resources['nsg']} deleted")
            except Exception as e:
                logger.warning(f"Failed to delete NSG: {e}")

        if resources.get("os_disk"):
            try:
                self._run_az_command(
                    [
                        "disk",
                        "delete",
                        "--resource-group",
                        self.config.resource_group,
                        "--name",
                        resources["os_disk"],
                        "--yes",
                    ]
                )
                logger.info(f"OS disk {resources['os_disk']} deleted")
            except Exception as e:
                logger.warning(f"Failed to delete OS disk: {e}")

    @contextmanager
    def vm_context(self):
        """Context manager for VM lifecycle."""
        try:
            # Register cleanup handlers for unexpected termination
            def cleanup_handler(*args):
                logger.info("Received termination signal, cleaning up VM...")
                self.cleanup()

            signal.signal(signal.SIGTERM, cleanup_handler)
            signal.signal(signal.SIGINT, cleanup_handler)
            atexit.register(self.cleanup)

            # Create and yield VM
            self.create_vm()
            yield self

        finally:
            # Always cleanup
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run ML experiments on Azure VMs")
    parser.add_argument(
        "experiment_script", help="Path to the experiment script to run"
    )
    parser.add_argument(
        "--resource-group", help="Azure resource group (overrides config)"
    )
    parser.add_argument("--vm-size", default="Standard_D2as_v4", help="Azure VM size")
    parser.add_argument("--location", help="Azure location (overrides config)")
    parser.add_argument("--ssh-key", help="Path to SSH public key")
    parser.add_argument("--data-dir", help="Data directory on VM (overrides config)")
    parser.add_argument(
        "--results-dir",
        default="/mnt/echoes_data/azure_results",
        help="Local results directory",
    )
    parser.add_argument(
        "--auto-destroy-hours",
        type=int,
        default=4,
        help="Hours after which VM auto-destroys",
    )

    args = parser.parse_args()

    setup_logging("INFO")
    azure_config = load_azure_config()
    resource_group = args.resource_group or azure_config["resource_group"]
    location = args.location or azure_config["location"]
    # Use local directory on remote VM, not headquarters mount point
    data_dir = args.data_dir or "/home/aclarke/echoes_data"

    vm_name = f"echoes-vm-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    experiment_path = Path(args.experiment_script)
    if not experiment_path.exists():
        logger.error(f"Experiment script not found: {args.experiment_script}")
        sys.exit(1)

    config = VMConfig(
        resource_group=resource_group,
        vm_name=vm_name,
        vm_size=args.vm_size,
        location=location,
        ssh_key_path=args.ssh_key or _default_ssh_key(),
        auto_destroy_hours=args.auto_destroy_hours,
    )
    runner = AzureGPURunner(config)

    with runner.vm_context():
        logger.info(f"VM is ready at {runner.vm_ip}")
        runner.setup_environment(data_dir)
        runner.copy_dataset(data_dir)
        runner.run_experiment(args.experiment_script, data_dir)
        runner.download_results(args.results_dir)


if __name__ == "__main__":
    main()
