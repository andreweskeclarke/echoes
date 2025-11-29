import atexit
import json
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.logging_config import get_logger
from remote.ssh import (
    CommandResult,
    CommandRunner,
    SshClient,
    SubprocessRunner,
)

logger = get_logger(__name__)


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
    project_name: str = "echoes"
    conda_env_name: str = "echoes"


class AzureGPURunner:
    def __init__(
        self,
        config: VMConfig,
        runner: CommandRunner | None = None,
        ssh_client_factory: type[SshClient] | None = None,
    ):
        self.config = config
        self._runner = runner or SubprocessRunner()
        self._ssh_client_factory = ssh_client_factory or SshClient
        self.vm_ip: str | None = None
        self._vm_created = False
        self._sync_stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._results_dir: str | None = None
        self._ssh: SshClient | None = None

    @property
    def ssh(self) -> SshClient:
        if self._ssh is None:
            if self.vm_ip is None:
                raise RuntimeError("VM IP not set - create or connect to a VM first")
            self._ssh = self._ssh_client_factory(self.vm_ip, runner=self._runner)
        return self._ssh

    @property
    def remote_project_dir(self) -> str:
        """Remote directory where project code is deployed."""
        return f"~/{self.config.project_name}"

    def _run_az_command(
        self, cmd: list[str], capture_output: bool = True
    ) -> CommandResult:
        """Run Azure CLI command with error handling."""
        logger.info(f"Running: az {' '.join(cmd)}")
        result = self._runner.run(["az", *cmd], capture_output=capture_output)

        if not result.success:
            logger.error(f"Azure CLI command failed: {result.stderr}")
            raise RuntimeError(f"Azure CLI command failed: {result.stderr}")

        return result

    def create_vm(self) -> str:
        """Create Azure GPU VM with auto-destroy tag and return its IP address."""
        logger.info(
            f"Creating VM {self.config.vm_name} in resource group "
            f"{self.config.resource_group}"
        )

        auto_destroy_time = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() + self.config.auto_destroy_hours * 3600),
        )

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
            "--security-type",
            "Standard",
            "--tags",
            f"AutoShutdownTime={auto_destroy_time}",
            "CreatedBy=echoes-gpu-runner",
            "--output",
            "json",
        ]

        result = self._run_az_command(create_cmd)
        vm_info = json.loads(result.stdout)
        self.vm_ip = vm_info["privateIpAddress"]
        self._ssh = None
        self._vm_created = True

        logger.info(f"VM created successfully. IP: {self.vm_ip}")
        logger.info(f"VM will auto-destroy at: {auto_destroy_time} UTC")

        logger.info("Waiting for VM to be ready...")
        time.sleep(60)

        if not self.ssh.test_connection(max_retries=10, retry_delay=30):
            raise RuntimeError("Could not establish SSH connection after 10 attempts")

        return self.vm_ip

    def connect_to_existing_vm(self, vm_ip: str = "", vm_name: str = "") -> str:
        """Connect to an existing VM by IP or name."""
        if vm_ip:
            self.vm_ip = vm_ip
        elif vm_name:
            logger.info(f"Resolving IP for VM {vm_name}...")
            self.vm_ip = self._get_vm_ip(vm_name, self.config.resource_group)
        else:
            raise ValueError("Either vm_ip or vm_name must be provided")

        self._ssh = None
        logger.info(f"Connecting to existing VM at {self.vm_ip}...")

        if not self.ssh.test_connection(max_retries=3, retry_delay=5):
            raise RuntimeError("Could not establish SSH connection after 3 attempts")

        return self.vm_ip

    def _setup_gpu_drivers(self):
        """Install NVIDIA drivers and verify nvidia-smi is available."""
        logger.info("GPU detected, installing NVIDIA drivers...")
        gpu_commands = [
            "sudo apt-get update",
            "sudo apt-get install -y ubuntu-drivers-common",
            "sudo ubuntu-drivers install",
        ]

        for cmd in gpu_commands:
            result = self.ssh.run_command(cmd, capture_output=False)
            if not result.success:
                raise RuntimeError(f"GPU driver installation failed at: {cmd}")

        logger.info("NVIDIA drivers installed, rebooting VM...")
        self.ssh.run_command("sudo reboot", capture_output=False)

        logger.info("Waiting for VM to reboot...")
        time.sleep(30)

        if not self.ssh.test_connection(max_retries=10, retry_delay=30):
            raise RuntimeError("Could not reconnect after reboot")

        logger.info("VM rebooted successfully")

        logger.info("Detecting installed NVIDIA driver version...")
        driver_check = self.ssh.run_command(
            "dpkg -l | grep nvidia-driver- | awk '{print $2}' | head -1"
        )
        if driver_check.success and driver_check.stdout.strip():
            driver_pkg = driver_check.stdout.strip()
            version = driver_pkg.replace("nvidia-driver-", "").split("-")[0]
            logger.info(f"Detected driver version: {version}")

            logger.info(f"Installing nvidia-utils-{version} for nvidia-smi...")
            utils_result = self.ssh.run_command(
                f"sudo apt-get install -y nvidia-utils-{version}", capture_output=False
            )
            if not utils_result.success:
                logger.warning(
                    f"Failed to install nvidia-utils-{version}, nvidia-smi may not work"
                )
        else:
            logger.warning("Could not detect NVIDIA driver version")

        logger.info("Verifying nvidia-smi...")
        smi_result = self.ssh.run_command("nvidia-smi", capture_output=True)
        if smi_result.success:
            logger.info("nvidia-smi verification successful")
            logger.info(smi_result.stdout)
        else:
            logger.warning("nvidia-smi not available after installation")

    def setup_environment(self, data_dir: str):
        """Set up the conda environment and codebase on the VM."""
        logger.info("Setting up environment on VM...")
        logger.info("Uploading codebase...")

        result = self.ssh.rsync_to_remote(
            f"{Path(__file__).parent.parent}/",
            f"{self.remote_project_dir}/",
            excludes=[".git", "__pycache__", "*.pyc", ".pytest_cache"],
        )
        if not result.success:
            raise RuntimeError("Failed to upload codebase")

        auto_shutdown_script = f"""#!/bin/bash
echo "Setting up auto-shutdown in {self.config.auto_destroy_hours} hours..."
(sleep {self.config.auto_destroy_hours * 3600} && sudo shutdown -h now) &
"""

        proj_dir = self.remote_project_dir
        base_setup_commands = [
            f"echo '{auto_shutdown_script}' > ~/auto_shutdown.sh",
            "chmod +x ~/auto_shutdown.sh",
            "nohup ~/auto_shutdown.sh > ~/auto_shutdown.log 2>&1 &",
            f"cd {proj_dir}",
            f"mkdir -p {data_dir}/logs {data_dir}/mlruns {data_dir}/tfruns",
            f"rm -rf {proj_dir}/logs {proj_dir}/mlruns {proj_dir}/tfruns",
            f"ln -sf {data_dir}/logs {proj_dir}/logs",
            f"ln -sf {data_dir}/mlruns {proj_dir}/mlruns",
            f"ln -sf {data_dir}/tfruns {proj_dir}/tfruns",
        ]

        for cmd in base_setup_commands:
            result = self.ssh.run_command(cmd, capture_output=False)
            if not result.success:
                raise RuntimeError(f"Environment setup failed at command: {cmd}")

        logger.info("Checking for GPU...")
        gpu_check = self.ssh.run_command("lspci | grep -i nvidia")
        if gpu_check.success:
            self._setup_gpu_drivers()
        else:
            logger.info("No GPU detected, skipping driver installation")

        conda_setup_commands = [
            (
                "wget "
                "https://github.com/conda-forge/miniforge/releases/latest/download/"
                "Miniforge3-Linux-x86_64.sh "
                "-O miniforge.sh"
            ),
            "bash miniforge.sh -b -p $HOME/miniforge",
            "rm miniforge.sh",
            (
                f"cd {proj_dir} && export PATH=$HOME/miniforge/bin:$PATH && "
                "mamba env create -f environment.yml -y 2>&1"
            ),
        ]

        for cmd in conda_setup_commands:
            result = self.ssh.run_command(cmd, capture_output=False)
            if not result.success:
                raise RuntimeError(f"Environment setup failed at command: {cmd}")

        logger.info("Environment setup completed")

    def copy_dataset(
        self, data_dir: str = "/tmp/echoes_data", source_path: str = "/mnt/echoes_data"
    ):
        """Copy UCF101 dataset from headquarters to remote VM via push-based rsync."""
        logger.info(f"Copying dataset from {source_path} to {self.vm_ip}:{data_dir}...")

        result = self.ssh.run_command(f"mkdir -p {data_dir}")
        if not result.success:
            raise RuntimeError(f"Failed to create directory {data_dir} on remote VM")

        result = self.ssh.rsync_to_remote(
            f"{source_path}/ucf101/",
            f"{data_dir}/ucf101/",
            show_progress=True,
        )
        if not result.success:
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
        proj_dir = self.remote_project_dir
        log_file = f"{proj_dir}/logs/{script_name}_{self.config.vm_name}.log"

        experiment_cmd = f"""
        cd {proj_dir} &&
        export PATH=$HOME/miniforge/bin:$PATH &&
        source $HOME/miniforge/etc/profile.d/conda.sh &&
        conda activate {self.config.conda_env_name} &&
        export CUDA_VISIBLE_DEVICES=0 &&
        sed -i 's|/mnt/echoes_data|{data_dir}|g' {experiment_script} &&
        python {experiment_script} 2>&1 | tee {log_file}
        """

        result = self.ssh.run_command(experiment_cmd, capture_output=False)

        if not result.success:
            raise RuntimeError(
                f"Experiment failed with return code: {result.returncode}"
            )

        logger.info("Experiment completed successfully")
        return {}

    def download_results(self, local_results_dir: str):
        """Download experiment results from the VM."""
        logger.info("Downloading results...")

        Path(local_results_dir).mkdir(parents=True, exist_ok=True)

        for subdir in ["mlruns", "logs", "tfruns"]:
            try:
                Path(f"{local_results_dir}/{subdir}").mkdir(parents=True, exist_ok=True)
                result = self.ssh.rsync_from_remote(
                    f"{self.remote_project_dir}/{subdir}/",
                    f"{local_results_dir}/{subdir}/",
                )
                if result.success:
                    logger.info(f"Downloaded {subdir}")
                else:
                    logger.warning(f"Failed to download {subdir}")
            except Exception:
                logger.warning(f"Failed to download {subdir}")

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
                self.ssh.rsync_from_remote(
                    f"{self.remote_project_dir}/{subdir}/",
                    f"{self._results_dir}/{subdir}/",
                    timeout=120,
                )
            except Exception:
                pass

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

            resources: dict[str, str] = {}
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


def cleanup_all_vms(
    resource_group: str,
    project_name: str = "echoes",
    runner: CommandRunner | None = None,
):
    """Delete all project VMs in the resource group."""
    runner = runner or SubprocessRunner()
    logger.info(f"Cleaning up all {project_name} VMs in {resource_group}...")

    query = f"[?starts_with(name, '{project_name}-vm')].name"
    result = runner.run(
        ["az", "vm", "list", "-g", resource_group, "--query", query, "-o", "tsv"],
    )
    vms = [v for v in result.stdout.strip().split("\n") if v]

    if not vms:
        logger.info(f"No {project_name} VMs found")
        return

    for vm in vms:
        logger.info(f"Deleting VM: {vm}")
        config = VMConfig(
            resource_group=resource_group, vm_name=vm, project_name=project_name
        )
        vm_runner = AzureGPURunner(config, runner=runner)
        vm_runner._vm_created = True
        vm_runner.cleanup()


def sync_results_only(
    vm_ip: str,
    results_dir: str,
    runner: CommandRunner | None = None,
):
    """Sync results from an existing VM."""
    logger.info(f"Syncing results from {vm_ip}")
    config = VMConfig(resource_group="", vm_name="")
    vm_runner = AzureGPURunner(config, runner=runner)
    vm_runner.vm_ip = vm_ip
    vm_runner._ssh = None
    vm_runner.download_results(results_dir)
