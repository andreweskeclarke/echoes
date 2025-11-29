#!/usr/bin/env python3
"""
Azure GPU Runner - Automates running ML experiments on Azure GPU VMs.

This script supports two modes:

1. New VM mode (default): Creates a new VM, runs experiment, cleans up
2. Existing VM mode: Uses an already-running VM (--vm-ip or --vm-name)

Usage:
    # Create new VM and run experiment
    python scripts/azure_gpu_runner.py experiments/train_simple.py \\
        --vm-size Standard_NC6s_v3

    # Run on existing VM by IP
    python scripts/azure_gpu_runner.py experiments/train_simple.py \\
        --vm-ip 10.0.0.5

    # Run on existing VM by name
    python scripts/azure_gpu_runner.py experiments/train_simple.py \\
        --vm-name my-gpu-vm

    # Skip setup/dataset on existing VM (faster for repeat runs)
    python scripts/azure_gpu_runner.py experiments/train_simple.py \\
        --vm-ip 10.0.0.5 --skip-setup --skip-dataset
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.logging_config import get_logger, setup_logging
from remote.azure import AzureGPURunner, VMConfig, cleanup_all_vms, sync_results_only
from remote.config import load_azure_config

logger = get_logger(__name__)


def _default_ssh_key() -> str:
    return os.path.expanduser("~/.ssh/id_ed25519.pub")


def run_on_existing_vm(
    runner: AzureGPURunner,
    args: argparse.Namespace,
    data_dir: str,
):
    """Run experiment on an existing VM."""
    runner.connect_to_existing_vm(vm_ip=args.vm_ip or "", vm_name=args.vm_name or "")
    logger.info(f"Connected to existing VM at {runner.vm_ip}")

    if not args.skip_setup:
        runner.setup_environment(data_dir)
    if not args.skip_dataset:
        runner.copy_dataset(data_dir)

    runner.start_background_sync(args.results_dir)
    try:
        runner.run_experiment(args.experiment_script, data_dir)
    finally:
        runner.stop_background_sync()
    runner.download_results(args.results_dir)


def run_on_new_vm(
    runner: AzureGPURunner,
    args: argparse.Namespace,
    data_dir: str,
):
    """Create a new VM, run experiment, and cleanup."""
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


def main():
    parser = argparse.ArgumentParser(description="Run ML experiments on Azure VMs")
    parser.add_argument(
        "experiment_script",
        nargs="?",
        help="Path to the experiment script to run",
    )
    parser.add_argument(
        "--resource-group",
        help="Azure resource group (overrides config)",
    )
    parser.add_argument(
        "--vm-size",
        default="Standard_D2as_v4",
        help="Azure VM size",
    )
    parser.add_argument(
        "--location",
        help="Azure location (overrides config)",
    )
    parser.add_argument(
        "--ssh-key",
        help="Path to SSH public key",
    )
    parser.add_argument(
        "--data-dir",
        help="Data directory on VM (overrides config)",
    )
    parser.add_argument(
        "--results-dir",
        default="/mnt/echoes_data/azure_results",
        help="Local results directory for downloading mlruns/, logs/, tfruns/",
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
    parser.add_argument(
        "--vm-ip",
        help="IP address of existing VM (skips VM creation/cleanup)",
    )
    parser.add_argument(
        "--vm-name",
        help="Name of existing VM (skips VM creation/cleanup)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip environment setup (for existing VMs already configured)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset copy (for existing VMs with data)",
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

    experiment_path = Path(args.experiment_script)
    if not experiment_path.exists():
        logger.error(f"Experiment script not found: {args.experiment_script}")
        sys.exit(1)

    use_existing_vm = args.vm_ip or args.vm_name
    azure_config = load_azure_config() if not use_existing_vm or args.vm_name else {}
    resource_group = args.resource_group or azure_config.get("resource_group", "")
    location = args.location or azure_config.get("location", "centralus")
    data_dir = args.data_dir or "/home/aclarke/echoes_data"

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

    if use_existing_vm:
        run_on_existing_vm(runner, args, data_dir)
    else:
        run_on_new_vm(runner, args, data_dir)


if __name__ == "__main__":
    main()
