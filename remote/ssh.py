import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from data.logging_config import get_logger

logger = get_logger(__name__)

# The same IP can correspond to many servers over time, namely the private IP 10.0.0.10
SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_OPTS_LIST = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    def run(
        self,
        cmd: list[str],
        capture_output: bool = True,
        check: bool = False,
        shell: bool = False,
        timeout: int | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Real command runner using subprocess."""

    def run(
        self,
        cmd: list[str],
        capture_output: bool = True,
        check: bool = False,
        shell: bool = False,
        timeout: int | None = None,
    ) -> CommandResult:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            check=check,
            text=True,
            shell=shell,
            timeout=timeout,
        )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout if capture_output else "",
            stderr=result.stderr if capture_output else "",
        )


class SshClient:
    """SSH client for executing commands on remote hosts."""

    def __init__(
        self,
        host: str,
        user: str = "aclarke",
        runner: CommandRunner | None = None,
    ):
        self.host = host
        self.user = user
        self._runner = runner or SubprocessRunner()

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def run_command(self, cmd: str, capture_output: bool = True) -> CommandResult:
        """Run a command on the remote host via SSH."""
        ssh_cmd = ["ssh", *SSH_OPTS_LIST, self.target, cmd]
        logger.info(f"SSH: {cmd}")

        result = self._runner.run(ssh_cmd, capture_output=capture_output)

        if not result.success and capture_output:
            logger.error(f"SSH command failed: {result.stderr}")

        return result

    def test_connection(self, max_retries: int = 10, retry_delay: int = 30) -> bool:
        """Test SSH connection with retries."""
        for i in range(max_retries):
            result = self.run_command("echo 'Connection test'")
            if result.success:
                logger.info("SSH connection established")
                return True
            if i < max_retries - 1:
                logger.info(
                    f"SSH attempt {i + 1} failed, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

        return False

    def rsync_to_remote(
        self,
        local_path: str | Path,
        remote_path: str,
        excludes: list[str] | None = None,
        show_progress: bool = False,
    ) -> CommandResult:
        """Rsync files from local to remote host."""
        cmd = ["rsync", "-az", "-e", f"ssh {SSH_OPTS}"]

        if show_progress:
            cmd.append("--info=progress2")

        for exclude in excludes or []:
            cmd.extend(["--exclude", exclude])

        cmd.extend([f"{local_path}", f"{self.target}:{remote_path}"])

        logger.info(f"Rsync to remote: {local_path} -> {remote_path}")
        return self._runner.run(cmd, capture_output=not show_progress)

    def rsync_from_remote(
        self,
        remote_path: str,
        local_path: str | Path,
        show_progress: bool = False,
        timeout: int | None = None,
    ) -> CommandResult:
        """Rsync files from remote host to local."""
        cmd = ["rsync", "-azL", "-e", f"ssh {SSH_OPTS}"]

        if show_progress:
            cmd.append("--info=progress2")

        cmd.extend([f"{self.target}:{remote_path}", f"{local_path}"])

        logger.info(f"Rsync from remote: {remote_path} -> {local_path}")
        return self._runner.run(cmd, capture_output=not show_progress, timeout=timeout)
