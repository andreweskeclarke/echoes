import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from data.logging_config import get_logger
from remote.ssh import CommandResult, CommandRunner, SubprocessRunner

logger = get_logger(__name__)


@dataclass
class LocalRunnerConfig:
    working_dir: str = field(default_factory=lambda: str(Path.cwd()))
    conda_env_name: str = "echoes"
    project_name: str = "echoes"


class LocalRunner:
    """Runs experiment commands directly via subprocess on the local machine.

    Implements the same interface as AzureGPURunner so experiments can switch
    between local and cloud execution without code changes.
    """

    def __init__(
        self,
        config: LocalRunnerConfig | None = None,
        runner: CommandRunner | None = None,
    ):
        self.config = config or LocalRunnerConfig()
        self._runner = runner or SubprocessRunner()

    @property
    def remote_project_dir(self) -> str:
        """Project directory (local path for LocalRunner)."""
        return self.config.working_dir

    def run_command(self, cmd: str, capture_output: bool = True) -> CommandResult:
        """Run a shell command in the project working directory."""
        logger.info(f"Local: {cmd}")
        result = self._runner.run(
            cmd,
            capture_output=capture_output,
            shell=True,
        )
        if not result.success and capture_output:
            logger.error(f"Command failed: {result.stderr}")
        return result

    def run_python(
        self,
        script_path: str,
        args: list[str] | None = None,
        capture_output: bool = True,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run a Python script in the conda environment."""
        conda_python = self._conda_python_path()
        cmd_parts = [conda_python, script_path] + (args or [])
        cmd = " ".join(cmd_parts)
        logger.info(f"Running script: {script_path}")
        result = self._runner.run(
            cmd,
            capture_output=capture_output,
            shell=True,
            timeout=timeout,
        )
        if not result.success and capture_output:
            logger.error(f"Script failed:\n{result.stderr}")
        return result

    def run_experiment(
        self,
        script_path: str,
        args: list[str] | None = None,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run an experiment script, streaming output to the terminal."""
        conda_python = self._conda_python_path()
        cmd_parts = [conda_python, script_path] + (args or [])
        cmd = " ".join(cmd_parts)
        logger.info(f"Starting experiment: {script_path}")
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            cwd=self.config.working_dir,
            timeout=timeout,
        )
        return CommandResult(
            returncode=result.returncode,
            stdout="",
            stderr="",
        )

    def _conda_python_path(self) -> str:
        """Return the python executable path for the configured conda env."""
        import shutil

        conda_base = subprocess.run(
            "conda info --base",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if conda_base:
            candidate = (
                Path(conda_base)
                / "envs"
                / self.config.conda_env_name
                / "bin"
                / "python"
            )
            if candidate.exists():
                return str(candidate)
        fallback = shutil.which("python") or "python"
        return fallback
