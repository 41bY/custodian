import shutil
from typing import TYPE_CHECKING

import pytest
from pyfhiaims.control.control import AimsControl

from custodian.aims.handlers import UnconvergedErrorHandler
from custodian.aims.interpreter import AimsModder
from custodian.utils import tracked_lru_cache
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from pathlib import Path

# Bulk Si (PBE, light) run with FHI-aims 260331_1: a converged SCF, an SCF stopped by
# sc_iter_limit 3, and a relaxation stopped by max_relaxation_steps 1.
KERKER = {"preconditioner": "kerker 2.0", "charge_mix_param": 0.1, "spin_mix_param": 0.1, "prec_mix_param": 0.1}


@pytest.fixture(autouse=True)
def _clear_tracked_cache() -> None:
    """Clear the cache of the stored functions between the tests."""
    tracked_lru_cache.tracked_cache_clear()


def copy_files(name: str, tmp_path: "Path") -> str:
    shutil.copytree(f"{TEST_FILES}/aims/{name}", tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


def set_parameters(parameters: dict) -> dict:
    return {f"parameters->{key}": val for key, val in parameters.items()}


class TestUnconvergedErrorHandler:
    def test_check_converged(self, tmp_path: "Path") -> None:
        assert not UnconvergedErrorHandler().check(directory=copy_files("static", tmp_path))

    @pytest.mark.parametrize(
        ("modification", "expected"),
        [
            ({"_unset": {"parameters->sc_iter_limit": 1}}, {"sc_iter_limit": 300}),
            ({}, {"n_max_pulay": 14}),
            ({"_set": set_parameters({"n_max_pulay": 14})}, KERKER),
            (
                {"_set": set_parameters({"n_max_pulay": 14, **KERKER})},
                {"charge_mix_param": 0.05, "spin_mix_param": 0.05, "prec_mix_param": 0.05},
            ),
            (
                {"_set": set_parameters({"n_max_pulay": 14, **KERKER, "charge_mix_param": 0.02})},
                {"sc_iter_limit": 1000},
            ),
            (
                {
                    "_set": set_parameters(
                        {"n_max_pulay": 14, **KERKER, "charge_mix_param": 0.02, "sc_iter_limit": 1000}
                    )
                },
                None,
            ),
        ],
    )
    def test_check_correct_electronic(self, tmp_path: "Path", modification: dict, expected: dict | None) -> None:
        directory = copy_files("scf_unconverged", tmp_path)
        if modification:
            AimsModder(directory=directory).apply_actions([{"dict": "control.in", "action": modification}])

        handler = UnconvergedErrorHandler()
        assert handler.check(directory=directory)
        dct = handler.correct(directory=directory)
        if expected is None:
            assert dct == {"errors": ["Unconverged"], "actions": None}
            return
        assert dct == {
            "errors": ["Unconverged"],
            "actions": [{"dict": "control.in", "action": {"_set": set_parameters(expected)}}],
        }
        parameters = AimsControl.from_file(f"{directory}/control.in").parameters
        assert {key: parameters[key] for key in expected} == {key: str(val) for key, val in expected.items()}
        assert (tmp_path / "error.1.tar.gz").is_file()

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
