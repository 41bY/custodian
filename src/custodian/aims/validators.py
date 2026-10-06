"""Implements various validators, e.g., check if aims.out is complete, for FHI-aims."""

from __future__ import annotations

import logging
import os
from collections import deque

from custodian.aims.io import load_aims_stdout
from custodian.custodian import Validator


class AimsOutputValidator(Validator):
    """Checks that FHI-aims ended normally, i.e. that aims.out ends with "Have a nice day."."""

    def __init__(self, output_file: str = "aims.out", stderr_file: str = "std_err.txt") -> None:
        """
        Args:
            output_file (str): Name of file FHI-aims standard output is directed to.
                Defaults to "aims.out".
            stderr_file (str): Name of file FHI-aims standard error is direct to.
                Defaults to "std_err.txt".
        """
        self.output_file = output_file
        self.stderr_file = stderr_file
        self.logger = logging.getLogger(type(self).__name__)

    def check(self, directory="./") -> bool:
        """Check for errors."""
        try:
            if load_aims_stdout(os.path.join(directory, self.output_file)).is_finished_ok:
                return False
        except Exception:
            pass

        exception_context: dict[str, str] = {}
        for key, filename in (("output_file_tail", self.output_file), ("stderr_file_tail", self.stderr_file)):
            if os.path.isfile(os.path.join(directory, filename)):
                with open(os.path.join(directory, filename)) as file:
                    exception_context[key] = "".join(deque(file, maxlen=10))
        self.logger.error("FHI-aims did not end normally", extra=exception_context)
        return True
