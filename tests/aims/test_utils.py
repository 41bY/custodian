import numpy as np
import pytest

from custodian.aims.utils import ScfProfile, scf_gap, scf_profile, scf_profile_metrics

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
        (np.zeros(200), None, ScfProfile.STALLED),  # flat from the start
        (DECREASING[:15], None, ScfProfile.SLOW),  # too few iterations to tell
        (DECREASING[:0], None, ScfProfile.SLOW),  # no iteration yet
    ],
)
def test_scf_profile(log_charge, log_spin, expected) -> None:
    scf_steps = [{"charge_density_change": 10**x} for x in log_charge]
    if log_spin is not None:
        for step, x in zip(scf_steps, log_spin, strict=True):
            step["spin_density_change"] = 10**x
    assert scf_profile(scf_steps) == expected


@pytest.mark.parametrize(
    ("log_charge", "expected"),
    [
        (DECREASING, (ScfProfile.SLOW, pytest.approx(-0.02), pytest.approx(0, abs=1e-12))),
        (PLATEAU, (ScfProfile.STALLED, pytest.approx(0), pytest.approx(0, abs=1e-12))),
        (DECREASING[:15], (ScfProfile.SLOW, None, None)),  # too few iterations to tell
    ],
)
def test_scf_profile_metrics(log_charge, expected) -> None:
    # the slope of the smoothed curve in decades per iteration, and the scatter around it
    assert scf_profile_metrics([{"charge_density_change": 10**x} for x in log_charge]) == expected


@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        ([1.0] * 30 + [0.0] * 40, 0.0),  # metallic over the last 50 iterations
        ([0.0] * 30 + [1.0] * 40, 1.0),
        ([], 0.0),  # FHI-aims printed none
    ],
)
def test_scf_gap(gaps, expected) -> None:
    assert scf_gap([{"gap": gap} for gap in gaps]) == expected
