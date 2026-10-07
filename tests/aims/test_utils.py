import numpy as np
import pytest

from custodian.aims.utils import ScfProfile, scf_profile

ITERATIONS = np.arange(200)
DECREASING = -0.02 * ITERATIONS  # 2 decades every 100 iterations
PLATEAU = np.maximum(-0.05 * ITERATIONS, -2)  # 2 decades in 40 iterations, then flat


@pytest.mark.parametrize(
    ("log_charge", "log_spin", "expected"),
    [
        (DECREASING, None, ScfProfile.SLOW),
        (PLATEAU, None, ScfProfile.STALLED),
        (DECREASING, PLATEAU, ScfProfile.STALLED),  # the spin stops at -2 when the charge gets there
        (PLATEAU + 0.3 * (-1) ** ITERATIONS, None, ScfProfile.OSCILLATING),
        (np.zeros(200), None, ScfProfile.OSCILLATING),  # it never went down
        (DECREASING[:15], None, ScfProfile.SLOW),  # too few iterations to tell
    ],
)
def test_scf_profile(log_charge, log_spin, expected) -> None:
    scf_steps = [{"charge_density_change": 10**x} for x in log_charge]
    if log_spin is not None:
        for step, x in zip(scf_steps, log_spin, strict=True):
            step["spin_density_change"] = 10**x
    assert scf_profile(scf_steps) == expected
