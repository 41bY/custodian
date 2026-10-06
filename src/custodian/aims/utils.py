"""Utility methods for FHI-aims error handlers."""

import logging
import os

from pyfhiaims.geometry.geometry import AimsGeometry

logger = logging.getLogger(__name__)


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
