import json
import shutil
from typing import TYPE_CHECKING

import pytest

from custodian.aims.failures import REPORT_FILE, build_report, current_attempt, write_report
from custodian.aims.handlers import ScfMonitorHandler, UnconvergedErrorHandler
from custodian.aims.jobs import AimsJob
from custodian.utils import backup
from tests.aims.test_handlers import aborted_run, copy_files
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from pathlib import Path

# The correction between the two runs of two_runs, as custodian.json holds it
LONGER = {
    "handler": {"@module": "custodian.aims.handlers", "@class": "UnconvergedErrorHandler"},
    "errors": ["Unconverged", "slow", "gap"],
    "actions": [{"dict": "control.in", "action": {"_set": {"parameters->sc_iter_limit": 300}}}],
}


def read_report(tmp_path: "Path") -> dict:
    return json.loads((tmp_path / REPORT_FILE).read_text())


def two_runs(tmp_path: "Path") -> str:
    """A run stopped by sc_iter_limit, backed up, then a converged run, and custodian.json."""
    directory = copy_files("scf_unconverged", tmp_path)
    backup(["aims.out", "control.in", "geometry.in"], directory=directory)
    shutil.copy(f"{TEST_FILES}/aims/static/aims.out", tmp_path / "aims.out")
    (tmp_path / "custodian.json").write_text(json.dumps([{"corrections": [LONGER]}]))
    return directory


class TestBuildReport:
    def test_runs_from_the_backups_and_custodian_json(self, tmp_path: "Path") -> None:
        report = build_report(two_runs(tmp_path))
        first, second = report["attempts"]
        assert (first["source"], second["source"]) == ("error.1.tar.gz", ".")
        assert (first["converged"], second["converged"]) == (False, True)
        assert len(first["scf"]["charge_density_change"]) == 3
        assert len(second["scf"]["wall_time"]) == 10
        assert first["corrections"] == [{**LONGER, "handler": "UnconvergedErrorHandler"}]
        assert second["corrections"] == []
        assert report["status"] == "converged"
        assert report["structure"] == {"n_atoms": 2, "composition": {"Si": 2}}
        # no sc_accuracy_rho in control.in: the one FHI-aims chose
        assert second["sc_accuracy_rho"] == pytest.approx(1e-6)
        assert second["sc_accuracy_rho_set_by"] == "FHI-aims"

    def test_pending_correction(self, tmp_path: "Path") -> None:
        pending = {"handler": "UnconvergedErrorHandler", "errors": ["Unconverged", "slow", "gap"], "actions": None}
        report = build_report(copy_files("scf_unconverged", tmp_path), pending)
        assert report["attempts"][0]["corrections"] == [pending]
        assert report["status"] == "no correction left"

    def test_run_repeated_by_its_backup_counts_once(self, tmp_path: "Path") -> None:
        # custodian stopped right after the correction: the directory still holds the backed-up run
        directory = copy_files("scf_unconverged", tmp_path)
        backup(["aims.out", "control.in"], directory=directory)
        assert [run["source"] for run in build_report(directory)["attempts"]] == ["error.1.tar.gz"]

    def test_unreadable_files_do_not_fail_the_run(self, tmp_path: "Path") -> None:
        directory = copy_files("scf_unconverged", tmp_path)
        (tmp_path / "custodian.json").write_text("not json")
        assert write_report(directory) is None
        assert not (tmp_path / REPORT_FILE).exists()


class TestHandlersWriteTheReport:
    def test_correction_before_its_backup(self, tmp_path: "Path") -> None:
        directory = aborted_run(tmp_path)
        (tmp_path / "abort_scf").write_text("stalled")
        UnconvergedErrorHandler().correct(directory=directory)
        (run,) = read_report(tmp_path)["attempts"]
        assert run["source"] == "."  # written before the backup
        assert run["corrections"][-1]["errors"][:2] == ["Unconverged", "stalled"]
        # classified as the check that stopped it: the 100 iterations ("elsi_restart write 100")
        # before the one FHI-aims ended
        assert run["aborted"]
        assert (run["classification"]["iterations"], run["classification"]["profile"]) == ([2, 101], "stalled")
        assert (tmp_path / "error.1.tar.gz").is_file()
        assert current_attempt(directory) == 2
        # read again afterwards, the run comes from its backup
        assert [run["source"] for run in build_report(directory)["attempts"]] == ["error.1.tar.gz"]

    @pytest.mark.parametrize(
        ("n_iterations", "checkpoint", "profile", "stop"),
        [(100, 0, None, False), (101, 101, "stalled", True)],  # before and past the first checkpoint
    )
    def test_monitor_check(
        self, tmp_path: "Path", n_iterations: int, checkpoint: int, profile: str | None, stop: bool
    ) -> None:
        directory = aborted_run(tmp_path, n_iterations)
        assert ScfMonitorHandler(checkpoint_interval=100).check(directory=directory) is stop
        (check,) = read_report(tmp_path)["checks"]
        assert {key: check[key] for key in ("attempt", "iteration", "checkpoint", "profile", "stop")} == {
            "attempt": 1,
            "iteration": n_iterations,
            "checkpoint": checkpoint,
            "profile": profile,
            "stop": stop,
        }
        assert (check["slope"] is None) is (profile is None)

    def test_checks_stay_in_the_report(self, tmp_path: "Path") -> None:
        directory = aborted_run(tmp_path, 101)
        ScfMonitorHandler(checkpoint_interval=100).check(directory=directory)
        (tmp_path / "abort_scf").write_text("stalled")
        UnconvergedErrorHandler().correct(directory=directory)
        assert len(read_report(tmp_path)["checks"]) == 1

    def test_without_report(self, tmp_path: "Path") -> None:
        directory = aborted_run(tmp_path, 101)
        assert ScfMonitorHandler(checkpoint_interval=100, report=False).check(directory=directory)
        (tmp_path / "abort_scf").write_text("stalled")
        UnconvergedErrorHandler(report=False).correct(directory=directory)
        assert not (tmp_path / REPORT_FILE).exists()


@pytest.mark.parametrize("keep", [False, True])
def test_job_success(tmp_path: "Path", keep: bool) -> None:
    # the report of a job that succeeds goes, or is completed with the converged run
    directory = two_runs(tmp_path)
    write_report(directory)
    AimsJob(["aims.x"], keep_failures_report=keep).postprocess(directory=directory)
    assert (tmp_path / REPORT_FILE).exists() is keep
    if keep:
        assert read_report(tmp_path)["status"] == "converged"
