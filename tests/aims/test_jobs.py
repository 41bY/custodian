import gzip
import os
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from pyfhiaims.control.control import AimsControl

from custodian.aims.jobs import AIMS_OUTPUT_FILES, AimsJob
from custodian.custodian import Custodian
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

# Stands in for FHI-aims: one line on each stream, as at the end of a normal run.
FAKE_AIMS = [sys.executable, "-c", "import sys; print('Have a nice day.'); print('on stderr', file=sys.stderr)"]


@pytest.fixture
def static_dir(tmp_path: "Path") -> "Path":
    """A copy of a pymatgen-generated FHI-aims input set (bulk Si, PBE, light)."""
    shutil.copytree(f"{TEST_FILES}/aims/static", tmp_path, dirs_exist_ok=True)
    return tmp_path


class TestAimsJob:
    def test_as_from_dict(self) -> None:
        override = [{"dict": "control.in", "action": {"_set": {"parameters->sc_iter_limit": 300}}}]
        job = AimsJob(["srun", "aims.x"], suffix=".relax1", final=False, settings_override=override)
        job2 = AimsJob.from_dict(job.as_dict())
        assert isinstance(job2, AimsJob)
        assert job2.aims_cmd == ("srun", "aims.x")
        assert job2.suffix == ".relax1"
        assert job2.final is False
        assert job2.settings_override == override

    def test_cmd_from_string(self) -> None:
        assert AimsJob("srun -n 4 aims.x").aims_cmd == ("srun", "-n", "4", "aims.x")

    @pytest.mark.parametrize("file", ["control.in", "geometry.in"])
    def test_setup_missing_input_raises(self, static_dir: "Path", file: str) -> None:
        (static_dir / file).unlink()
        with pytest.raises(FileNotFoundError):
            AimsJob(FAKE_AIMS).setup(directory=str(static_dir))

    def test_setup_backs_up_inputs(self, static_dir: "Path") -> None:
        AimsJob(FAKE_AIMS).setup(directory=str(static_dir))
        for file in ("control.in", "geometry.in"):
            assert (static_dir / f"{file}.orig").read_text() == (static_dir / file).read_text()
        assert not (static_dir / "parameters.json.orig").exists()

    def test_setup_settings_override(self, static_dir: "Path") -> None:
        original = {file: (static_dir / file).read_text() for file in ("control.in", "geometry.in")}
        (static_dir / "geometry.in.next_step").write_text("next step")
        override = [
            {"dict": "control.in", "action": {"_set": {"parameters->sc_iter_limit": 300}}},
            {"file": "geometry.in.next_step", "action": {"_file_copy": {"dest": "geometry.in"}}},
        ]
        AimsJob(FAKE_AIMS, settings_override=override).setup(directory=str(static_dir))

        assert AimsControl.from_file(static_dir / "control.in").parameters["sc_iter_limit"] == "300"
        assert (static_dir / "geometry.in").read_text() == "next step"
        # the backups are taken before the override
        for file, content in original.items():
            assert (static_dir / f"{file}.orig").read_text() == content

    def test_setup_no_backup(self, static_dir: "Path") -> None:
        AimsJob(FAKE_AIMS, backup=False).setup(directory=str(static_dir))
        assert not list(static_dir.glob("*.orig"))

    def test_setup_decompresses(self, static_dir: "Path") -> None:
        geometry = static_dir / "geometry.in"
        with open(geometry, "rb") as f_in, gzip.open(f"{geometry}.gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        geometry.unlink()

        AimsJob(FAKE_AIMS).setup(directory=str(static_dir))
        assert geometry.is_file()
        assert (static_dir / "geometry.in.orig").is_file()

    def test_run_redirects_streams(self, static_dir: "Path") -> None:
        job = AimsJob(FAKE_AIMS)
        process = job.run(directory=str(static_dir))
        assert process.wait(timeout=60) == 0
        assert (static_dir / job.output_file).read_text() == "Have a nice day.\n"
        assert (static_dir / job.stderr_file).read_text() == "on stderr\n"

    def test_custodian_run(self, static_dir: "Path") -> None:
        job = AimsJob(FAKE_AIMS)
        run_log = Custodian(handlers=[], jobs=[job], directory=str(static_dir)).run()
        assert len(run_log) == 1
        assert run_log[0]["corrections"] == []
        assert (static_dir / "custodian.json").is_file()
        assert (static_dir / "control.in.orig").is_file()
        assert "Have a nice day." in (static_dir / job.output_file).read_text()

    @pytest.mark.parametrize("final", [True, False])
    def test_postprocess_suffix(self, static_dir: "Path", final: bool) -> None:
        job = AimsJob(FAKE_AIMS, suffix=".relax1", final=final)
        outputs = dict.fromkeys((*AIMS_OUTPUT_FILES, job.output_file))
        for file in (*outputs, job.stderr_file):
            (static_dir / file).write_text(file)

        job.postprocess(directory=str(static_dir))
        for file in outputs:
            assert (static_dir / f"{file}.relax1").read_text() == file
            assert (static_dir / file).exists() is not final
        # stderr and the inputs keep their names
        for file in (job.stderr_file, "control.in", "geometry.in", "parameters.json"):
            assert (static_dir / file).is_file()
            assert not (static_dir / f"{file}.relax1").exists()

    def test_postprocess_without_suffix(self, static_dir: "Path") -> None:
        job = AimsJob(FAKE_AIMS)
        (static_dir / job.output_file).write_text("output")
        before = sorted(os.listdir(static_dir))
        job.postprocess(directory=str(static_dir))
        assert sorted(os.listdir(static_dir)) == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group tests")
class TestAimsJobTerminate:
    """Tests for AimsJob.terminate() POSIX process group handling."""

    @pytest.fixture
    def mocks(self) -> "Generator[SimpleNamespace, None, None]":
        """Create AimsJob with mocked process and os functions."""
        job = AimsJob(["srun", "aims.x"])
        process = Mock(pid=12345)
        job._aims_process = process

        with (
            patch("custodian.aims.jobs.logger") as logger,
            patch("os.killpg") as killpg,
            patch("os.getpgid", return_value=67890),
        ):
            yield SimpleNamespace(job=job, process=process, logger=logger, killpg=killpg)

    def test_already_finished(self, mocks: SimpleNamespace) -> None:
        mocks.process.poll.return_value = 0
        mocks.job.terminate()

        mocks.logger.warning.assert_called_with("Process 12345 already terminated")
        mocks.killpg.assert_not_called()

    def test_sigterm_success(self, mocks: SimpleNamespace) -> None:
        mocks.process.poll.return_value = None
        mocks.job.terminate()

        mocks.killpg.assert_called_once_with(67890, signal.SIGTERM)
        mocks.process.wait.assert_called_once_with(timeout=10.0)
        mocks.logger.info.assert_any_call("Process 12345 terminated gracefully")
        mocks.process.kill.assert_not_called()

    def test_sigkill_after_timeout(self, mocks: SimpleNamespace) -> None:
        mocks.process.poll.return_value = None
        mocks.process.wait.side_effect = [subprocess.TimeoutExpired("aims", 10), None]
        mocks.job.terminate()

        assert mocks.killpg.call_count == 2
        mocks.killpg.assert_any_call(67890, signal.SIGTERM)
        mocks.killpg.assert_any_call(67890, signal.SIGKILL)
        mocks.logger.info.assert_any_call("Process 12345 killed with SIGKILL")

    def test_fallback_after_sigkill_timeout(self, mocks: SimpleNamespace) -> None:
        mocks.process.poll.return_value = None
        mocks.process.wait.side_effect = [
            subprocess.TimeoutExpired("aims", 10),  # after SIGTERM
            subprocess.TimeoutExpired("aims", 10),  # after SIGKILL
            None,  # after fallback terminate
        ]
        mocks.job.terminate()

        mocks.logger.warning.assert_any_call("Falling back to killing parent process 12345")
        mocks.process.terminate.assert_called_once()

    def test_integration_with_real_process(self, tmp_path: "Path") -> None:
        job = AimsJob(["sleep", "30"])
        process = job.run(directory=str(tmp_path))
        pgid = os.getpgid(process.pid)

        with patch("custodian.aims.jobs.logger"):
            job.terminate()

        assert process.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
