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
    python scripts/azure_gpu_runner.py experiments/train_simple.py --vm-size Standard_NC6s_v3
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
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
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


class AzureGPURunner:
    def __init__(
        self,
        resource_group: str,
        vm_name: str,
        vm_size: str = "Standard_D2as_v4",
        location: str = "eastus",
        ssh_key_path: str | None = None,
        auto_destroy_hours: int = 4,
    ):
        self.resource_group = resource_group
        self.vm_name = vm_name
        self.vm_size = vm_size
        self.location = location
        default_key = os.path.expanduser("~/.ssh/id_ed25519.pub")
        self.ssh_key_path = ssh_key_path or default_key
        self.auto_destroy_hours = auto_destroy_hours
        self.vm_ip = None
        self._vm_created = False

    def _run_az_command(
        self, cmd: list, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run Azure CLI command with error handling."""
        logger.info(f"Running: az {' '.join(cmd)}")
        result = subprocess.run(
            ["az"] + cmd, check=False, capture_output=capture_output, text=True
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
            f"Creating VM {self.vm_name} in resource group {self.resource_group}"
        )

        # Calculate auto-destroy time (current time + hours in UTC)
        auto_destroy_time = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() + self.auto_destroy_hours * 3600),
        )

        # Create VM with auto-destroy tag
        create_cmd = [
            "vm",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            self.vm_name,
            "--image",
            "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest",
            "--size",
            self.vm_size,
            "--admin-username",
            "aclarke",
            "--ssh-key-values",
            self.ssh_key_path,
            "--location",
            self.location,
            "--tags",
            f"AutoShutdownTime={auto_destroy_time}",
            "CreatedBy=echoes-gpu-runner",
            "--output",
            "json",
        ]

        result = self._run_az_command(create_cmd)
        vm_info = json.loads(result.stdout)
        self.vm_ip = vm_info["publicIpAddress"]
        self._vm_created = True

        logger.info(f"VM created successfully. IP: {self.vm_ip}")
        logger.info(f"VM will auto-destroy at: {auto_destroy_time} UTC")

        # Set up auto-shutdown on the VM itself as backup
        self._setup_vm_auto_shutdown()

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
                        f"Could not establish SSH connection after {max_retries} attempts"
                    )
                logger.info(f"SSH attempt {i + 1} failed, retrying in 30s...")
                time.sleep(30)

        return self.vm_ip

    def _setup_vm_auto_shutdown(self):
        """Set up auto-shutdown script on the VM as a backup safety measure."""
        logger.info("Setting up VM auto-shutdown as backup safety...")

        # Create auto-shutdown script that runs after the specified hours
        shutdown_script = f"""#!/bin/bash
# Auto-shutdown script created by echoes-gpu-runner
sleep {self.auto_destroy_hours * 3600}
sudo shutdown -h now
"""

        # Note: We'll set this up after SSH is available in setup_environment

    def setup_environment(self, data_dir: str):
        """Set up the conda environment and codebase on the VM."""
        logger.info("Setting up environment on VM...")

        # Upload project code
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
echo "Setting up auto-shutdown in {self.auto_destroy_hours} hours..."
(sleep {self.auto_destroy_hours * 3600} && sudo shutdown -h now) &
"""

        # Install conda and create environment
        setup_commands = [
            f"echo '{auto_shutdown_script}' > ~/auto_shutdown.sh",
            "chmod +x ~/auto_shutdown.sh",
            "nohup ~/auto_shutdown.sh > ~/auto_shutdown.log 2>&1 &",
            "cd ~/echoes",
            # Create data directories on VM
            f"mkdir -p {data_dir}/logs {data_dir}/mlruns {data_dir}/tfruns",
            # Remove local directories and create symlinks
            "rm -rf ~/echoes/logs ~/echoes/mlruns ~/echoes/tfruns",
            f"ln -sf {data_dir}/logs ~/echoes/logs",
            f"ln -sf {data_dir}/mlruns ~/echoes/mlruns",
            f"ln -sf {data_dir}/tfruns ~/echoes/tfruns",
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh",
            "bash miniconda.sh -b -p $HOME/miniconda",
            "export PATH=$HOME/miniconda/bin:$PATH && conda init bash",
            # Accept conda Terms of Service for required channels
            "export PATH=$HOME/miniconda/bin:$PATH && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
            "export PATH=$HOME/miniconda/bin:$PATH && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
            # Use mamba for faster package resolution, fallback to conda
            "cd ~/echoes && export PATH=$HOME/miniconda/bin:$PATH && (conda install mamba -c conda-forge -y && mamba env create -f environment.yml -y || conda env create -f environment.yml -y)",
        ]

        for cmd in setup_commands:
            result = self._run_ssh_command(cmd, capture_output=False)
            if result.returncode != 0:
                raise RuntimeError(f"Environment setup failed at command: {cmd}")

        logger.info("Environment setup completed")

    def copy_dataset(
        self, data_dir: str = "/tmp/echoes_data", source_vm_ip: str = None
    ):
        """Copy UCF101 dataset from source VM."""
        if not source_vm_ip:
            azure_config = load_azure_config()
            source_vm_ip = self._get_vm_ip(
                "headquarters", azure_config["resource_group"]
            )

        logger.info(f"Copying dataset from {source_vm_ip} to {self.vm_ip}...")

        copy_commands = [
            f"mkdir -p {data_dir}",
            f"rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' aclarke@{source_vm_ip}:/mnt/echoes_data/ucf101/ {data_dir}/ucf101/",
        ]

        for cmd in copy_commands:
            result = self._run_ssh_command(cmd, capture_output=False)
            if result.returncode != 0:
                raise RuntimeError(f"Dataset copy failed at command: {cmd}")

        logger.info("Dataset copy completed")

    def _get_vm_ip(self, vm_name: str, resource_group: str) -> str:
        """Get public IP of a VM."""
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
                "publicIps",
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

        # Modify the experiment script to use the VM's data directory
        experiment_cmd = f"""
        cd ~/echoes && 
        source ~/.bashrc && 
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

        # Get latest MLflow run ID from remote (MLflow runs are now on mounted disk via symlink)
        get_latest_run_cmd = "ls -t ~/echoes/mlruns/*/*/meta.yaml 2>/dev/null | head -1 | cut -d'/' -f6 || echo 'no_runs'"
        result = self._run_ssh_command(get_latest_run_cmd)

        if result.returncode == 0 and result.stdout.strip() != "no_runs":
            latest_run_id = result.stdout.strip()
            experiment_id_cmd = "ls -t ~/echoes/mlruns/*/*/meta.yaml 2>/dev/null | head -1 | cut -d'/' -f5 || echo 'no_exp'"
            exp_result = self._run_ssh_command(experiment_id_cmd)

            if exp_result.returncode == 0 and exp_result.stdout.strip() != "no_exp":
                experiment_id = exp_result.stdout.strip()

                # Download only the latest run
                download_commands = [
                    f"rsync -avz aclarke@{self.vm_ip}:~/echoes/mlruns/{experiment_id}/{latest_run_id}/ {local_results_dir}/mlruns/{experiment_id}/{latest_run_id}/",
                ]

                for cmd in download_commands:
                    try:
                        subprocess.run(cmd, shell=True, check=True)
                        logger.info(
                            f"Downloaded experiment {experiment_id}, run {latest_run_id}"
                        )
                    except subprocess.CalledProcessError:
                        logger.warning(f"Failed to download with command: {cmd}")
            else:
                logger.warning("Could not determine experiment ID")
        else:
            logger.warning("No MLflow runs found to download")

        # Always try to download logs (small)
        try:
            subprocess.run(
                f"rsync -avz aclarke@{self.vm_ip}:~/echoes/logs/ {local_results_dir}/logs/",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("Failed to download logs")

        logger.info(f"Results downloaded to {local_results_dir}")

    def cleanup(self):
        """Delete the VM and associated resources."""
        if not self._vm_created:
            logger.info("No VM to cleanup")
            return

        logger.info(f"Deleting VM {self.vm_name} (aclarke@{self.vm_ip})")

        time.sleep(99999)

        delete_cmd = [
            "vm",
            "delete",
            "--resource-group",
            self.resource_group,
            "--name",
            self.vm_name,
            "--yes",
        ]

        try:
            self._run_az_command(delete_cmd)
            logger.info("VM deleted successfully")
            self._vm_created = False
        except Exception as e:
            logger.error(f"Failed to delete VM: {e}")

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
    parser.add_argument("--vm-name", help="VM name (default: auto-generated)")
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
    parser.add_argument(
        "--skip-dataset", action="store_true", help="Skip dataset download"
    )

    args = parser.parse_args()

    setup_logging("INFO")
    azure_config = load_azure_config()
    resource_group = args.resource_group or azure_config["resource_group"]
    location = args.location or azure_config["location"]
    # Use local directory on remote VM, not headquarters mount point
    data_dir = args.data_dir or "/home/aclarke/echoes_data"

    if not args.vm_name:
        timestamp = int(time.time())
        args.vm_name = f"echoes-vm-{timestamp}"

    experiment_path = Path(args.experiment_script)
    if not experiment_path.exists():
        logger.error(f"Experiment script not found: {args.experiment_script}")
        sys.exit(1)

    runner = AzureGPURunner(
        resource_group=resource_group,
        vm_name=args.vm_name,
        vm_size=args.vm_size,
        location=location,
        ssh_key_path=args.ssh_key,
        auto_destroy_hours=args.auto_destroy_hours,
    )

    with runner.vm_context():
        logger.info(f"VM is ready at {runner.vm_ip}")
        runner.setup_environment(data_dir)

        if not args.skip_dataset:
            runner.copy_dataset(data_dir)

        runner.run_experiment(args.experiment_script, data_dir)
        runner.download_results(args.results_dir)


if __name__ == "__main__":
    main()
