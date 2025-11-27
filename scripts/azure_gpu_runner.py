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
import threading
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
    sync_interval_minutes: int = 5


SSH_OPTS = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"


class AzureGPURunner:
    def __init__(self, config):
        self.config = config
        self.vm_ip = None
        self._vm_created = False
        self._sync_stop_event = threading.Event()
        self._sync_thread = None
        self._results_dir = None

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
                "-az",
                "-e",
                SSH_OPTS,
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
            "-az",
            "--info=progress2",
            "-e",
            SSH_OPTS,
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

        script_name = Path(experiment_script).stem
        log_file = f"~/echoes/logs/{script_name}_{self.config.vm_name}.log"

        experiment_cmd = f"""
        cd ~/echoes &&
        export PATH=$HOME/miniforge/bin:$PATH &&
        source $HOME/miniforge/etc/profile.d/conda.sh &&
        conda activate echoes &&
        export CUDA_VISIBLE_DEVICES=0 &&
        sed -i 's|/mnt/echoes_data|{data_dir}|g' {experiment_script} &&
        python {experiment_script} 2>&1 | tee {log_file}
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

        try:
            Path(f"{local_results_dir}/mlruns").mkdir(parents=True, exist_ok=True)
            mlruns_cmd = (
                f"rsync -az -e '{SSH_OPTS}' aclarke@{self.vm_ip}:~/echoes/mlruns/ "
                f"{local_results_dir}/mlruns/"
            )
            subprocess.run(mlruns_cmd, shell=True, check=True)
            logger.info("Downloaded MLflow runs")
        except subprocess.CalledProcessError:
            logger.warning("Failed to download MLflow runs")

        try:
            Path(f"{local_results_dir}/logs").mkdir(parents=True, exist_ok=True)
            logs_cmd = (
                f"rsync -az -e '{SSH_OPTS}' aclarke@{self.vm_ip}:~/echoes/logs/ "
                f"{local_results_dir}/logs/"
            )
            subprocess.run(logs_cmd, shell=True, check=True)
            logger.info("Downloaded logs")
        except subprocess.CalledProcessError:
            logger.warning("Failed to download logs")

        try:
            Path(f"{local_results_dir}/tfruns").mkdir(parents=True, exist_ok=True)
            tfruns_cmd = (
                f"rsync -az -e '{SSH_OPTS}' aclarke@{self.vm_ip}:~/echoes/tfruns/ "
                f"{local_results_dir}/tfruns/"
            )
            subprocess.run(tfruns_cmd, shell=True, check=True)
            logger.info("Downloaded TensorBoard runs")
        except subprocess.CalledProcessError:
            logger.warning("Failed to download TensorBoard runs")

        logger.info(f"Results downloaded to {local_results_dir}")

    def start_background_sync(self, results_dir: str):
        """Start background thread that periodically syncs results."""
        if self.config.sync_interval_minutes <= 0:
            logger.info("Background sync disabled (sync_interval_minutes <= 0)")
            return

        self._results_dir = results_dir
        self._sync_stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._background_sync_loop, daemon=True
        )
        self._sync_thread.start()
        logger.info(
            f"Background sync started (every {self.config.sync_interval_minutes} min)"
        )

    def stop_background_sync(self):
        """Stop the background sync thread."""
        if self._sync_thread is None:
            return

        self._sync_stop_event.set()
        self._sync_thread.join(timeout=10)
        self._sync_thread = None
        logger.info("Background sync stopped")

    def _background_sync_loop(self):
        """Background loop that periodically syncs results from VM."""
        interval_seconds = self.config.sync_interval_minutes * 60
        sync_count = 0

        while not self._sync_stop_event.wait(timeout=interval_seconds):
            sync_count += 1
            logger.info(f"Background sync #{sync_count} starting...")
            try:
                self._do_quiet_sync()
                logger.info(f"Background sync #{sync_count} completed")
            except Exception as e:
                logger.warning(f"Background sync #{sync_count} failed: {e}")

    def _do_quiet_sync(self):
        """Perform a quiet rsync without verbose logging."""
        if not self._results_dir or not self.vm_ip:
            return

        for subdir in ["mlruns", "tfruns", "logs"]:
            try:
                Path(f"{self._results_dir}/{subdir}").mkdir(parents=True, exist_ok=True)
                cmd = (
                    f"rsync -az -e '{SSH_OPTS}' "
                    f"aclarke@{self.vm_ip}:~/echoes/{subdir}/ "
                    f"{self._results_dir}/{subdir}/"
                )
                subprocess.run(
                    cmd, shell=True, check=True, capture_output=True, timeout=120
                )
            except subprocess.CalledProcessError:
                pass
            except subprocess.TimeoutExpired:
                logger.warning(f"Background sync of {subdir} timed out")

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

            def cleanup_handler(*args):
                logger.info("Received termination signal, cleaning up VM...")
                self.cleanup()
                sys.exit(1)

            signal.signal(signal.SIGTERM, cleanup_handler)
            signal.signal(signal.SIGINT, cleanup_handler)
            atexit.register(self.cleanup)

            self.create_vm()
            yield self

        finally:
            self.cleanup()


def cleanup_all_vms(resource_group: str):
    """Delete all echoes VMs in the resource group."""
    logger.info(f"Cleaning up all echoes VMs in {resource_group}...")
    query = "[?starts_with(name, 'echoes-vm')].name"
    result = subprocess.run(
        ["az", "vm", "list", "-g", resource_group, "--query", query, "-o", "tsv"],
        check=False,
        capture_output=True,
        text=True,
    )
    vms = [v for v in result.stdout.strip().split("\n") if v]
    if not vms:
        logger.info("No echoes VMs found")
        return
    for vm in vms:
        logger.info(f"Deleting VM: {vm}")
        config = VMConfig(resource_group=resource_group, vm_name=vm)
        runner = AzureGPURunner(config)
        runner._vm_created = True
        runner.cleanup()


def sync_results_only(vm_ip: str, results_dir: str):
    """Sync results from an existing VM."""
    logger.info(f"Syncing results from {vm_ip}")
    config = VMConfig(resource_group="", vm_name="")
    runner = AzureGPURunner(config)
    runner.vm_ip = vm_ip
    runner.download_results(results_dir)


def main():
    parser = argparse.ArgumentParser(description="Run ML experiments on Azure VMs")
    parser.add_argument(
        "experiment_script",
        nargs="?",
        help="Path to the experiment script to run",
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
        default="/mnt/echoes_data",
        help="Local results directory (mlruns/, logs/, tfruns/ subdirs)",
    )
    parser.add_argument(
        "--auto-destroy-hours",
        type=int,
        default=4,
        help="Hours after which VM auto-destroys",
    )
    parser.add_argument(
        "--sync-only",
        metavar="VM_IP",
        help="Only sync results from existing VM at this IP",
    )
    parser.add_argument(
        "--cleanup-vms",
        action="store_true",
        help="Delete all echoes VMs in the resource group",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=5,
        help="Minutes between background syncs (0 to disable)",
    )

    args = parser.parse_args()

    setup_logging("INFO")

    if args.cleanup_vms:
        azure_config = load_azure_config()
        rg = args.resource_group or azure_config["resource_group"]
        cleanup_all_vms(rg)
        return

    if args.sync_only:
        sync_results_only(args.sync_only, args.results_dir)
        return

    if not args.experiment_script:
        parser.error("experiment_script is required unless using --sync-only")

    azure_config = load_azure_config()
    resource_group = args.resource_group or azure_config["resource_group"]
    location = args.location or azure_config["location"]
    data_dir = args.data_dir or "/home/aclarke/echoes_data"

    experiment_path = Path(args.experiment_script)
    if not experiment_path.exists():
        logger.error(f"Experiment script not found: {args.experiment_script}")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    vm_name = f"echoes-vm-{timestamp}"

    config = VMConfig(
        resource_group=resource_group,
        vm_name=vm_name,
        vm_size=args.vm_size,
        location=location,
        ssh_key_path=args.ssh_key or _default_ssh_key(),
        auto_destroy_hours=args.auto_destroy_hours,
        sync_interval_minutes=args.sync_interval,
    )
    runner = AzureGPURunner(config)

    with runner.vm_context():
        logger.info(f"VM is ready at {runner.vm_ip}")
        runner.setup_environment(data_dir)
        runner.copy_dataset(data_dir)
        runner.start_background_sync(args.results_dir)
        try:
            runner.run_experiment(args.experiment_script, data_dir)
        finally:
            runner.stop_background_sync()
        runner.download_results(args.results_dir)


if __name__ == "__main__":
    main()
