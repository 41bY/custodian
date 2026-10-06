"""This module implements basic kinds of jobs for FHI-aims runs."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
from typing import TYPE_CHECKING

from monty.shutil import decompress_dir

from custodian.custodian import Job

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Backed up with a ".orig" suffix by AimsJob.setup. parameters.json is the record of the
# input parameters written by pymatgen (and read back by atomate2); it is optional.
AIMS_INPUT_FILES = ("control.in", "geometry.in", "parameters.json")

# Renamed with the job suffix by AimsJob.postprocess. The inputs keep their
# names: atomate2 reads geometry.in and parameters.json without suffix.
AIMS_OUTPUT_FILES = ("geometry.in.next_step", "hessian.aims")


class AimsJob(Job):
    """
    A basic FHI-aims job. Just runs whatever is in the directory. But conceivably
    can be a complex processing of inputs etc. with initialization.
    """

    def __init__(
        self,
        aims_cmd: str | Sequence[str],
        output_file: str = "aims.out",
        stderr_file: str = "std_err.txt",
        suffix: str = "",
        final: bool = True,
        backup: bool = True,
        terminate_timeout: float = 10.0,
    ) -> None:
        """
        Args:
            aims_cmd (str | list[str]): Command to run FHI-aims, as a list of args (e.g.
                ["srun", "aims.x"]) or as a string, which is split shell-like. The command
                is run without a shell, so it must not redirect its output: AimsJob writes
                the standard output to output_file and the standard error to stderr_file.
            output_file (str): Name of the file to direct standard output to, i.e. the main
                FHI-aims output. Defaults to "aims.out".
            stderr_file (str): Name of the file to direct standard error to. FHI-aims
                repeats its fatal messages there, and the errors of srun, MPI and GPU
                libraries end up there too. Defaults to "std_err.txt".
            suffix (str): A suffix to be appended to the final output. E.g., to rename
                aims.out to aims.out.relax1, provide ".relax1" as the suffix.
            final (bool): Indicating whether this is the final FHI-aims job in a series.
                The suffixed files are moved if True, copied if False. Defaults to True.
            backup (bool): Whether to backup the initial input files. If True, control.in,
                geometry.in and (if present) parameters.json are copied with a ".orig"
                appended. Defaults to True.
            terminate_timeout (float): Timeout in seconds to wait for graceful termination
                (SIGTERM) before escalating to SIGKILL. Defaults to 10.0 seconds.
        """
        self.aims_cmd = tuple(shlex.split(aims_cmd)) if isinstance(aims_cmd, str) else tuple(aims_cmd)
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.suffix = suffix
        self.final = final
        self.backup = backup
        self.terminate_timeout = terminate_timeout

    def setup(self, directory: str = "./") -> None:
        """
        Performs initial setup for AimsJob: decompresses the directory and backs up the
        inputs.
        """
        decompress_dir(directory)

        if self.backup:
            for file in AIMS_INPUT_FILES:
                path = os.path.join(directory, file)
                try:
                    shutil.copy(path, f"{path}.orig")
                except FileNotFoundError:
                    if file != "parameters.json":  # Mandatory files
                        raise

    def run(self, directory: str = "./") -> subprocess.Popen:
        """
        Perform the actual FHI-aims run.

        Returns:
            (subprocess.Popen) Used for monitoring.
        """
        cmd = list(self.aims_cmd)
        logger.info(f"Running {' '.join(cmd)}")
        with (
            open(os.path.join(directory, self.output_file), "w") as f_std,
            open(os.path.join(directory, self.stderr_file), "w", buffering=1) as f_err,
        ):
            # use line buffering for stderr
            self._aims_process = subprocess.Popen(
                cmd, cwd=directory, stdout=f_std, stderr=f_err, start_new_session=True
            )
            return self._aims_process

    def postprocess(self, directory: str = "./") -> None:
        """Postprocessing includes renaming files where necessary."""
        for file in (*AIMS_OUTPUT_FILES, self.output_file):
            file = os.path.join(directory, file)
            if os.path.isfile(file):
                if self.final and self.suffix != "":
                    shutil.move(file, f"{file}{self.suffix}")
                elif self.suffix != "":
                    shutil.copy(file, f"{file}{self.suffix}")

    def terminate(self, directory: str = "./") -> None:
        """Kill all FHI-aims processes associated with the current job.

        SIGTERM to the whole process group (the job is started in a new
        session, so the group holds the MPI launcher and its children),
        then SIGKILL, then the launcher process alone as a last resort.

        Tries to kill the entire process group (safest for MPI jobs), then waits
        to confirm termination. Escalates SIGTERM → SIGKILL → parent process fallback.

        Note: The parent process fallback may leave behind ghost MPI child processes
        (less likely with srun since SLURM purportedly cleans up process trees).

        Args:
            directory: Unused, kept for API compatibility with base class.
        """
        pid = self._aims_process.pid

        if self._aims_process.poll() is not None:
            logger.warning(f"Process {pid} already terminated")
            return

        if os.name != "nt":
            # Look up process group ID
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                logger.warning(f"Process group for {pid} not found")
                return

            # Send SIGTERM to the entire process group
            logger.info(f"Sending SIGTERM to process group {pgid}")
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} not found")
                return
            except OSError as exc:
                logger.warning(f"SIGTERM to process group {pgid} failed: {exc}")
            else:
                # Wait for graceful termination (only if SIGTERM was sent)
                try:
                    self._aims_process.wait(timeout=self.terminate_timeout)
                    logger.info(f"Process {pid} terminated gracefully")
                    return
                except subprocess.TimeoutExpired:
                    logger.warning(f"SIGTERM timeout ({self.terminate_timeout}s), sending SIGKILL")

            # Escalate to SIGKILL
            logger.info(f"Sending SIGKILL to process group {pgid}")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                logger.warning(f"Process group {pgid} not found")
                return
            except OSError as exc:
                logger.warning(f"SIGKILL to process group {pgid} failed: {exc}")
            else:
                # Wait for process to die (only if SIGKILL was sent)
                try:
                    self._aims_process.wait(timeout=self.terminate_timeout)
                    logger.info(f"Process {pid} killed with SIGKILL")
                    return
                except subprocess.TimeoutExpired:
                    pass  # Fall through to parent process fallback

        # Fall back to killing the parent launcher process (Windows or if above failed)
        logger.warning(f"Falling back to killing parent process {pid}")
        try:
            self._aims_process.terminate()
            self._aims_process.wait(timeout=self.terminate_timeout)
            logger.info(f"Process {pid} terminated")
        except subprocess.TimeoutExpired:
            self._aims_process.kill()
            self._aims_process.wait()
            logger.info(f"Process {pid} killed")
