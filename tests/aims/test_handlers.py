import gzip
import re
import shutil
from typing import TYPE_CHECKING

import pytest
from pyfhiaims.control.control import AimsControl

from custodian.aims.handlers import RoundChargeErrorHandler, ScfMonitorHandler, UnconvergedErrorHandler
from custodian.aims.interpreter import AimsModder
from custodian.aims.utils import ScfProfile
from custodian.utils import tracked_lru_cache
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from pathlib import Path

# Bulk Si (PBE, light) run with FHI-aims 260331_1: a converged SCF, an SCF stopped by
# sc_iter_limit 3, and a relaxation stopped by max_relaxation_steps 1. With FHI-aims 261002:
# a run stopped by initial charges adding up to 1.2e-5, its rerun with
# override_initial_charge_check, and the stalled SCF cycle of a magnetic ordering of a
# phyllosilicate (r2SCAN, Kerker, mixing 0.1) stopped with abort_scf after 102 iterations.
KERKER = {"preconditioner": "kerker 2.0", "charge_mix_param": 0.1, "spin_mix_param": 0.1, "prec_mix_param": 0.1}
LOWGAP = {"adjust_scf_param": "lowgap charge_mix_param 0.04"}
DAMPED = {"charge_mix_param": 0.05, "spin_mix_param": 0.05, "prec_mix_param": 0.05}


@pytest.fixture(autouse=True)
def _clear_tracked_cache() -> None:
    """Clear the cache of the stored functions between the tests."""
    tracked_lru_cache.tracked_cache_clear()


def copy_files(name: str, tmp_path: "Path") -> str:
    shutil.copytree(f"{TEST_FILES}/aims/{name}", tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


def set_parameters(parameters: dict) -> dict:
    return {f"parameters->{key}": val for key, val in parameters.items()}


def aborted_run(tmp_path: "Path", n_iterations: int | None = None) -> str:
    """The SCF cycle stopped with abort_scf, or cut while it runs, after n_iterations."""
    directory = copy_files("scf_aborted", tmp_path)
    output = gzip.decompress((tmp_path / "aims.out.gz").read_bytes()).decode()
    if n_iterations is not None:
        ends = [match.end() for match in re.finditer(r"End self-consistency iteration.*\n", output)]
        output = output[: ends[n_iterations - 1] + 1000]  # with the start of the next iteration
    (tmp_path / "aims.out").write_text(output)
    return directory


class TestUnconvergedErrorHandler:
    def test_check_converged(self, tmp_path: "Path") -> None:
        assert not UnconvergedErrorHandler().check(directory=copy_files("static", tmp_path))

    @pytest.mark.parametrize(
        ("profile", "modification", "expected"),
        [
            # still converging: more iterations, then a faster mixing, then a longer Pulay history
            (ScfProfile.SLOW, {"_unset": {"parameters->sc_iter_limit": 1}}, {"sc_iter_limit": 300}),
            (ScfProfile.SLOW, {}, {"sc_iter_limit": 1000}),
            (
                ScfProfile.SLOW,
                {"_set": set_parameters({"sc_iter_limit": 1000})},
                {"adjust_scf_param": "lowgap charge_mix_param 0.04"},
            ),
            (ScfProfile.SLOW, {"_set": set_parameters({"sc_iter_limit": 1000, **LOWGAP})}, {"n_max_pulay": 14}),
            (ScfProfile.SLOW, {"_set": set_parameters({"sc_iter_limit": 1000, **LOWGAP, "n_max_pulay": 14})}, None),
            # not converging: the Kerker preconditioner, then a damped mixing, then more iterations
            (ScfProfile.STALLED, {"_unset": {"parameters->sc_iter_limit": 1}}, {"sc_iter_limit": 300, **KERKER}),
            (ScfProfile.OSCILLATING, {}, KERKER),
            (ScfProfile.STALLED, {"_set": set_parameters(KERKER)}, DAMPED),
            (
                ScfProfile.OSCILLATING,
                {"_set": set_parameters({**KERKER, "charge_mix_param": 0.05})},
                {"sc_iter_limit": 1000},
            ),
            (
                ScfProfile.STALLED,
                {"_set": set_parameters({**KERKER, "charge_mix_param": 0.05, "sc_iter_limit": 1000})},
                None,
            ),
        ],
    )
    def test_check_correct_electronic(
        self,
        tmp_path: "Path",
        monkeypatch: pytest.MonkeyPatch,
        profile: ScfProfile,
        modification: dict,
        expected: dict | None,
    ) -> None:
        directory = copy_files("scf_unconverged", tmp_path)
        if modification:
            AimsModder(directory=directory).apply_actions([{"dict": "control.in", "action": modification}])
        monkeypatch.setattr("custodian.aims.handlers.scf_profile", lambda scf_steps: profile)

        handler = UnconvergedErrorHandler()
        assert handler.check(directory=directory)
        dct = handler.correct(directory=directory)
        if expected is None:
            assert dct == {"errors": ["Unconverged", profile.value], "actions": None}
            return
        assert dct == {
            "errors": ["Unconverged", profile.value],
            "actions": [{"dict": "control.in", "action": {"_set": set_parameters(expected)}}],
        }
        parameters = AimsControl.from_file(f"{directory}/control.in").parameters
        assert {key: parameters[key] for key in expected} == {key: str(val) for key, val in expected.items()}
        assert (tmp_path / "error.1.tar.gz").is_file()

    def test_check_correct_aborted(self, tmp_path: "Path") -> None:
        # The run stopped by ScfMonitorHandler is corrected as any unconverged SCF cycle
        directory = aborted_run(tmp_path)
        handler = UnconvergedErrorHandler()
        assert handler.check(directory=directory)
        assert handler.correct(directory=directory) == {
            "errors": ["Unconverged", ScfProfile.STALLED.value],
            "actions": [{"dict": "control.in", "action": {"_set": set_parameters(DAMPED)}}],
        }

    def test_check_correct_ionic(self, tmp_path: "Path") -> None:
        directory = copy_files("relax_max_steps", tmp_path)
        handler = UnconvergedErrorHandler()
        assert handler.check(directory=directory)
        dct = handler.correct(directory=directory)
        assert dct == {
            "errors": ["Unconverged"],
            "actions": [{"file": "geometry.in.next_step", "action": {"_file_copy": {"dest": "geometry.in"}}}],
        }
        assert (tmp_path / "geometry.in").read_text() == (tmp_path / "geometry.in.next_step").read_text()


class TestScfMonitorHandler:
    def test_settings_override(self) -> None:
        settings = {"sc_iter_limit": 1000, "elsi_restart": "read_and_write 100"}
        assert ScfMonitorHandler().settings_override == [
            {"dict": "control.in", "action": {"_set": set_parameters(settings)}}
        ]
        # without restarts, no density matrices are written
        assert ScfMonitorHandler(rel_accuracy_residue=None).settings_override == [
            {"dict": "control.in", "action": {"_set": set_parameters({"sc_iter_limit": 1000})}}
        ]

    @pytest.mark.parametrize(
        ("n_iterations", "profile", "stopping", "expected"),
        [
            (100, None, False, False),  # before the first checkpoint, iteration 101
            (101, None, False, True),  # stalled at the checkpoint
            (101, ScfProfile.SLOW, False, False),  # still converging
            (101, None, True, False),  # abort_scf already written
            (None, None, False, False),  # FHI-aims has stopped: UnconvergedErrorHandler takes over
        ],
    )
    def test_check(
        self,
        tmp_path: "Path",
        monkeypatch: pytest.MonkeyPatch,
        n_iterations: int | None,
        profile: ScfProfile | None,
        stopping: bool,
        expected: bool,
    ) -> None:
        directory = aborted_run(tmp_path, n_iterations)
        if profile is not None:
            monkeypatch.setattr("custodian.aims.handlers.scf_profile", lambda scf_steps: profile)
        if stopping:
            (tmp_path / "abort_scf").touch()
        assert ScfMonitorHandler().check(directory=directory) is expected

    @pytest.mark.parametrize(
        ("rel_accuracy_residue", "restart"),
        # the density change at the checkpoint is 15 times sc_accuracy_rho
        [(100, True), (10, False)],
    )
    def test_correct(self, tmp_path: "Path", rel_accuracy_residue: float, restart: bool) -> None:
        directory = aborted_run(tmp_path, 101)
        for spin in (1, 2):
            (tmp_path / f"D_spin_0{spin}_kpt_000001.csc").touch()

        handler = ScfMonitorHandler(rel_accuracy_residue=rel_accuracy_residue)
        assert handler.check(directory=directory)
        actions = [{"file": "abort_scf", "action": {"_file_create": {"content": ""}}}]
        if not restart:
            actions.append({"file": "D_spin_*.csc", "action": {"_file_delete": {"mode": "actual"}}})
        assert handler.correct(directory=directory) == {
            "errors": ["Non-converging SCF", ScfProfile.STALLED.value],
            "actions": actions,
        }
        assert (tmp_path / "abort_scf").is_file()
        assert len(list(tmp_path.glob("D_spin_*.csc"))) == (2 if restart else 0)


class TestRoundChargeErrorHandler:
    @pytest.mark.parametrize("killed", [False, True])
    def test_check_override(self, tmp_path: "Path", killed: bool) -> None:
        # With override_initial_charge_check, FHI-aims reports the mismatch as a warning. A run
        # killed right after it leaves the warning as the last block of aims.out, like an error.
        directory = copy_files("initial_charge_override", tmp_path)
        if killed:
            output = (tmp_path / "aims.out").read_text()
            (tmp_path / "aims.out").write_text(output[: output.index("  Number of empty states per atom")])
        assert not RoundChargeErrorHandler().check(directory=directory)

    @pytest.mark.parametrize(
        ("charge_tol", "actions"),
        [
            (1e-3, [{"dict": "control.in", "action": {"_set": {"parameters->override_initial_charge_check": True}}}]),
            # The initial charges add up to 1.2e-5: beyond this tolerance it is not rounding
            (1e-6, None),
        ],
    )
    def test_check_correct(self, tmp_path: "Path", charge_tol: float, actions: list | None) -> None:
        directory = copy_files("initial_charge", tmp_path)
        geometry = (tmp_path / "geometry.in").read_text()

        handler = RoundChargeErrorHandler(charge_tol=charge_tol)
        assert handler.check(directory=directory)
        assert handler.correct(directory=directory) == {"errors": ["RoundCharge"], "actions": actions}
        parameters = AimsControl.from_file(f"{directory}/control.in").parameters
        assert parameters.get("override_initial_charge_check") == (".true." if actions else None)
        assert (tmp_path / "geometry.in").read_text() == geometry
        assert (tmp_path / "error.1.tar.gz").is_file() is bool(actions)
