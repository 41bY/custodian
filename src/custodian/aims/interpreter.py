"""Implements various interpreters and modders for FHI-aims."""

import os

from pyfhiaims.control.control import AimsControl
from pyfhiaims.geometry.geometry import AimsGeometry

from custodian.ansible.actions import DictActions, FileActions
from custodian.ansible.interpreter import Modder


class AimsModder(Modder):
    """A Modder for FHI-aims inputs."""

    def __init__(self, actions=None, strict=True, control=None, directory="./") -> None:
        """Initialize a Modder for FHI-aims inputs.

        Args:
            actions ([Action]): A sequence of supported actions. See
                :mod:`custodian.ansible.actions`. Default is None,
                which means DictActions and FileActions are supported.
            strict (bool): Indicating whether to use strict mode. In non-strict
                mode, unsupported actions are simply ignored without any
                errors raised. In strict mode, if an unsupported action is
                supplied, a ValueError is raised. Defaults to True.
            control (AimsControl): The control.in from the current directory.
                Initialized automatically if not passed (but passing it will
                avoid having to re-parse the directory).
            directory (str): The directory containing the FHI-aims inputs.
        """
        self.control = control or AimsControl.from_file(os.path.join(directory, "control.in"))
        self.directory = directory
        actions = actions or [FileActions, DictActions]
        super().__init__(actions, strict, directory=directory)

    def apply_actions(self, actions) -> None:
        """
        Applies a list of actions to the FHI-aims inputs and rewrites the modified
        control.in.

        Args:
            actions (dict): A list of actions of the form {'file': filename,
                'action': moddermodification} or {'dict': 'control.in',
                'action': moddermodification}, the latter acting on the keys of
                AimsControl.as_dict(), e.g. {"_set": {"parameters->sc_iter_limit": 300}}.
        """
        species_defaults = self.control.species_defaults  # not kept by as_dict/from_dict
        modified = []
        for action in actions:
            if "dict" in action:
                if action["dict"] != "control.in":
                    raise ValueError(f"Unrecognized dict: {action['dict']}")
                modified.append(action["dict"])
                self.control = self.modify_object(action["action"], self.control)
            elif "file" in action:
                self.modify(action["action"], action["file"])
            else:
                raise ValueError(f"Unrecognized format: {action}")
        if not modified:
            return

        # Species not conserved in serialization: recover them from geometry
        geometry = AimsGeometry.from_file(os.path.join(self.directory, "geometry.in"))
        for symbol, species in species_defaults.items():
            geometry.set_species(symbol, species)
        self.control.write_file(geometry, self.directory, overwrite=True)
