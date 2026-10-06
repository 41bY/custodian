"""Helper functions for dealing with FHI-aims files."""

from pyfhiaims.outputs.stdout import AimsStdout

from custodian.utils import tracked_lru_cache


@tracked_lru_cache
def load_aims_stdout(filepath):
    """
    Load AimsStdout object from file path.
    Caches the output for reuse.

    Args:
        filepath: path to the main FHI-aims output (aims.out).

    Returns:
        The AimsStdout object
    """
    return AimsStdout(filepath)
