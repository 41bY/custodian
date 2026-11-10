import shutil
from typing import TYPE_CHECKING

import pytest
from pyfhiaims.control.control import AimsControl

from custodian.aims.interpreter import AimsModder
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def static_dir(tmp_path: "Path") -> "Path":
    """A copy of a pymatgen-generated FHI-aims input set (bulk Si, PBE, light)."""
    shutil.copytree(f"{TEST_FILES}/aims/static", tmp_path, dirs_exist_ok=True)
    return tmp_path


class TestAimsModder:
    def test_dict_action(self, static_dir: "Path") -> None:
        parameters = (static_dir / "parameters.json").read_text()
        modification = {"_set": {"parameters->sc_iter_limit": 300}, "_unset": {"parameters->occupation_type": 1}}
        AimsModder(directory=str(static_dir)).apply_actions([{"dict": "control.in", "action": modification}])

        control = AimsControl.from_file(static_dir / "control.in")
        assert control.parameters["sc_iter_limit"] == "300"
        assert "occupation_type" not in control.parameters
        assert list(control.species_defaults) == ["Si"]
        # as for VASP, only the inputs of the code are modified
        assert (static_dir / "parameters.json").read_text() == parameters

    def test_file_action(self, static_dir: "Path") -> None:
        control = (static_dir / "control.in").read_text()
        (static_dir / "geometry.in.next_step").write_text("next step")
        AimsModder(directory=str(static_dir)).apply_actions(
            [{"file": "geometry.in.next_step", "action": {"_file_copy": {"dest": "geometry.in"}}}]
        )
        assert (static_dir / "geometry.in").read_text() == "next step"
        assert (static_dir / "control.in").read_text() == control

    def test_file_action_wildcards(self, static_dir: "Path") -> None:
        for spin in (1, 2):
            (static_dir / f"D_spin_0{spin}_kpt_000001.csc").touch()
        AimsModder(directory=str(static_dir)).apply_actions(
            [
                {"file": "D_spin_*.csc", "action": {"_file_delete": {"mode": "actual"}}},
                {"file": "abort_scf", "action": {"_file_create": {"content": ""}}},
            ]
        )
        assert not list(static_dir.glob("D_spin_*"))
        assert (static_dir / "abort_scf").is_file()
