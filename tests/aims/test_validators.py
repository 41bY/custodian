import shutil
from typing import TYPE_CHECKING

import pytest

from custodian.aims.validators import AimsOutputValidator
from custodian.utils import tracked_lru_cache
from tests.conftest import TEST_FILES

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_tracked_cache() -> None:
    """Clear the cache of the stored functions between the tests."""
    tracked_lru_cache.tracked_cache_clear()


class TestAimsOutputValidator:
    @pytest.mark.parametrize(
        ("name", "failed"), [("static", False), ("relax_max_steps", False), ("scf_unconverged", True)]
    )
    def test_check(self, tmp_path: "Path", name: str, failed: bool) -> None:
        shutil.copytree(f"{TEST_FILES}/aims/{name}", tmp_path, dirs_exist_ok=True)
        assert AimsOutputValidator().check(directory=str(tmp_path)) is failed

    def test_check_without_output(self, tmp_path: "Path") -> None:
        assert AimsOutputValidator().check(directory=str(tmp_path))
