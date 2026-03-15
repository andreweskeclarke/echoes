#!/usr/bin/env python3
"""Run small ESN experiment via LocalRunner and verify MLflow logging."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.logging_config import get_logger, setup_logging
from remote.local import LocalRunner, LocalRunnerConfig

logger = get_logger(__name__)


def main() -> None:
    setup_logging("INFO")

    project_dir = str(Path(__file__).parent.parent)
    config = LocalRunnerConfig(
        working_dir=project_dir,
        conda_env_name="echoes",
        project_name="echoes",
    )
    runner = LocalRunner(config=config)

    logger.info(f"Project dir: {project_dir}")
    logger.info("Launching ESN experiment via LocalRunner...")

    result = runner.run_experiment(
        script_path="experiments/esn_local_small.py",
        timeout=600,
    )

    if result.returncode == 0:
        logger.info("Experiment completed successfully.")
        logger.info("MLflow run logged to /mnt/echoes_data/mlruns")
    else:
        logger.error(f"Experiment failed with returncode={result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
