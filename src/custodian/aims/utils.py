"""Utility methods for FHI-aims error handlers."""

import logging
import os
from enum import Enum

import numpy as np
from pyfhiaims.geometry.geometry import AimsGeometry

logger = logging.getLogger(__name__)


class ScfProfile(str, Enum):
    """How an unconverged SCF cycle behaved (see scf_profile)."""

    SLOW = "slow"
    STALLED = "stalled"
    OSCILLATING = "oscillating"


def scf_profile(scf_steps: list[dict]) -> ScfProfile:
    """Classify an unconverged SCF cycle from the change of the density at each iteration.

    The larger of the charge and spin density changes is taken in log10 and smoothed over 9
    iterations; the last half of the cycle decides the category:
    - SLOW: the smoothed curve still decreases, by more than 0.1 decades per 100 iterations;
    - OSCILLATING: otherwise, if the iterations scatter around it by 0.1 decades or more, or
      if it never went down;
    - STALLED: a plateau after an initial decrease, possibly rising at the end.
    The total energy change is not used: it changes sign and jumps at every iteration.

    Args:
        scf_steps: The SCF iterations, as parsed by pyfhiaims (AimsStdout.get_image(-1).scf).

    Returns:
        ScfProfile: The category.
    """
    changes = [
        max(s["charge_density_change"], s.get("spin_density_change", 0))
        for s in scf_steps
        if "charge_density_change" in s
    ]
    rho = np.log10(np.maximum(changes, 1e-15))
    smooth = np.convolve(rho, np.ones(9) / 9, mode="valid")
    if len(smooth) < 10:  # too few iterations to tell
        return ScfProfile.SLOW
    half = len(smooth) // 2
    if np.polyfit(np.arange(len(smooth) - half), smooth[half:], 1)[0] < -1e-3:
        return ScfProfile.SLOW
    # The plateau, if any, starts where the smoothed curve comes within 0.2 decades of its minimum
    plateau = np.argmax(smooth <= smooth.min() + 0.2) + 4
    if np.std((rho[4:-4] - smooth)[half:]) >= 0.1 or plateau < 10:
        return ScfProfile.OSCILLATING
    return ScfProfile.STALLED


def is_valid_geometry(filename: str, directory: str = "./") -> bool:
    """Check if a geometry file (e.g. geometry.in.next_step) is valid and can be parsed.

    Args:
        filename: Name of the file (e.g., "geometry.in.next_step")
        directory: Directory containing the file

    Returns:
        True if the file exists, is non-empty, and can be parsed as a valid
        FHI-aims geometry file. False otherwise.
    """
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        logger.warning(f"{filename} does not exist or is empty in {directory}")
        return False
    try:
        AimsGeometry.from_file(filepath)
        return True
    except Exception as exc:
        logger.warning(f"{filename} could not be parsed: {exc}")
        return False
