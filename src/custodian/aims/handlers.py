"""
This module implements specific error handlers for FHI-aims runs. These handlers
try to detect common errors in FHI-aims runs and attempt to fix them on the fly
by modifying the input files.
"""

import os

from pyfhiaims.control.control import AimsControl

from custodian.aims.interpreter import AimsModder
from custodian.aims.io import load_aims_stdout
from custodian.aims.utils import ScfProfile, is_valid_geometry, scf_profile
from custodian.custodian import ErrorHandler
from custodian.utils import backup

AIMS_BACKUP_FILES = {
    "control.in",
    "geometry.in",
    "geometry.in.next_step",
    "hessian.aims",
    "aims.out",
    "std_err.txt",
}


class UnconvergedErrorHandler(ErrorHandler):
    """
    Check if a run is converged.

    An unconverged SCF cycle is corrected according to how its density change behaved (see
    custodian.aims.utils.scf_profile):
    - SLOW, still converging: more iterations, then a faster mixing, then a longer Pulay history;
    - STALLED or OSCILLATING: the Kerker preconditioner, then a damped mixing, then more
      iterations.
    """

    is_monitor = False

    def __init__(
        self, output_filename: str = "aims.out", sc_iter_limit: int = 300, max_sc_iter_limit: int = 1000
    ) -> None:
        """Initialize the handler with the output file to check.

        Args:
            output_filename (str): Filename for the FHI-aims output. Change
                this only if it is different from the default (unlikely).
            sc_iter_limit (int): sc_iter_limit set when control.in has none. An explicit
                sc_iter_limit also switches off the automatic early stop of the SCF
                cycle, which FHI-aims applies only without one. Defaults to 300.
            max_sc_iter_limit (int): sc_iter_limit of the last attempt. Defaults to 1000.
        """
        self.output_filename = output_filename
        self.sc_iter_limit = sc_iter_limit
        self.max_sc_iter_limit = max_sc_iter_limit

    def check(self, directory="./") -> bool:
        """Check for error."""
        try:
            out = load_aims_stdout(os.path.join(directory, self.output_filename))
            if not out.converged:
                return True
        except Exception:
            pass
        return False

    def correct(self, directory="./"):
        """Perform corrections."""
        out = load_aims_stdout(os.path.join(directory, self.output_filename))
        params = AimsControl.from_file(os.path.join(directory, "control.in")).parameters
        actions = []
        errors = ["Unconverged"]
        if not out.get_image(-1).converged:
            profile = scf_profile(out.get_image(-1).scf)
            errors.append(profile.value)
            limit = int(params.get("sc_iter_limit", 0))
            # Without sc_iter_limit, FHI-aims may stop the SCF cycle early, at iteration 100
            new_settings = {} if limit else {"sc_iter_limit": self.sc_iter_limit}
            # Setting charge_mix_param or spin_mix_param switches off adjust_scf: set both.
            if profile is ScfProfile.SLOW and limit:
                if limit < self.max_sc_iter_limit:
                    new_settings = {"sc_iter_limit": self.max_sc_iter_limit}
                elif "charge_mix_param" not in params and "adjust_scf_param" not in params:
                    # Twice the mixing adjust_scf picks for low-gap systems (0.02); 0.2 above the gap
                    new_settings = {"adjust_scf_param": "lowgap charge_mix_param 0.04"}
                elif int(params.get("n_max_pulay", 8)) < 14:
                    new_settings = {"n_max_pulay": 14}
            elif profile is not ScfProfile.SLOW:
                if "preconditioner" not in params:
                    new_settings.update(
                        {
                            "preconditioner": "kerker 2.0",
                            "charge_mix_param": 0.1,
                            "spin_mix_param": 0.1,
                            "prec_mix_param": 0.1,
                        }
                    )
                elif float(params.get("charge_mix_param", 0.05)) > 0.05:
                    # While the Kerker preconditioner is on, FHI-aims mixes the charge with prec_mix_param
                    new_settings.update({"charge_mix_param": 0.05, "spin_mix_param": 0.05, "prec_mix_param": 0.05})
                elif limit < self.max_sc_iter_limit:
                    new_settings["sc_iter_limit"] = self.max_sc_iter_limit

            if new_settings:
                new_settings = {f"parameters->{key}": val for key, val in new_settings.items()}
                actions.append({"dict": "control.in", "action": {"_set": new_settings}})
                # Unlike VASP, FHI-aims stops a relaxation at its first unconverged SCF
                # cycle, so continue from its last geometry.
                if out.metadata.relax and is_valid_geometry("geometry.in.next_step", directory):
                    actions.append({"file": "geometry.in.next_step", "action": {"_file_copy": {"dest": "geometry.in"}}})

        elif is_valid_geometry("geometry.in.next_step", directory):
            # Just continue optimizing from the last geometry
            actions.append({"file": "geometry.in.next_step", "action": {"_file_copy": {"dest": "geometry.in"}}})

        if actions:
            backup(AIMS_BACKUP_FILES, directory=directory)
            AimsModder(directory=directory).apply_actions(actions)
            return {"errors": errors, "actions": actions}

        # Unfixable error. Just return None for actions.
        return {"errors": errors, "actions": None}
