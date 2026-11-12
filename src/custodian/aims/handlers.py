"""
This module implements specific error handlers for FHI-aims runs. These handlers
try to detect common errors in FHI-aims runs and attempt to fix them on the fly
by modifying the input files.
"""

import logging
import os
from glob import glob
from pathlib import Path

from pyfhiaims.control.control import AimsControl
from pyfhiaims.geometry.geometry import AimsGeometry

from custodian.aims.interpreter import AimsModder
from custodian.aims.io import load_aims_stdout
from custodian.aims.utils import ScfProfile, is_valid_geometry, scf_gap, scf_profile
from custodian.custodian import ErrorHandler
from custodian.utils import backup

logger = logging.getLogger(__name__)

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
    If the run left density matrices (elsi_restart, see ScfMonitorHandler), the next run restarts
    from them depending on the gap of the cycle (see custodian.aims.utils.scf_gap) and on its
    profile:
    - with a gap: first the same settings (a fresh Pulay history), then the ladder, always from the
      density matrices;
    - STALLED without a gap: the same settings from the density matrices, then the ladder from
      scratch;
    - OSCILLATING without a gap: the ladder from scratch.
    The same settings are tried once per density matrices, and never from scratch.
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
            # A cycle stopped by ScfMonitorHandler is corrected for the profile it was stopped for
            # (written in abort_scf), not reclassified with the few iterations FHI-aims did after
            abort = Path(directory, "abort_scf")
            stopped = abort.read_text().strip() if abort.is_file() else ""
            monitored = stopped in {p.value for p in ScfProfile}
            scf_steps = out.get_image(-1).scf
            profile = ScfProfile(stopped) if monitored else scf_profile(scf_steps)
            # With a gap, the half-converged orbitals are worth restarting from, also with new settings
            gapped = scf_gap(scf_steps) > 0.1
            errors += [profile.value, "gap" if gapped else "no gap"]
            limit = int(params.get("sc_iter_limit", 0))
            # Without sc_iter_limit, FHI-aims may stop the SCF cycle early, at iteration 100
            new_settings = {} if limit else {"sc_iter_limit": self.sc_iter_limit}
            # The same settings, restarting from the density matrices with a fresh Pulay history: once,
            # with a gap or on a plateau, and only if the run left them (ScfMonitorHandler on)
            density_matrices = bool(glob("D_spin_*.csc", root_dir=directory))
            output = Path(directory, self.output_filename).read_text(errors="replace") if density_matrices else ""
            same_settings = (
                profile is not ScfProfile.SLOW
                and (gapped or profile is ScfProfile.STALLED)
                and density_matrices
                and "Restarting calculation from saved density matrices" not in output
            )
            # Setting charge_mix_param or spin_mix_param switches off adjust_scf: set both.
            if same_settings:
                logger.info("Restarting from the density matrices with the same settings")
            elif profile is ScfProfile.SLOW and limit:
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
                # New mixing settings restart from the density matrices only with a gap; otherwise from
                # scratch, as the last rungs of VASP's NonConvergingErrorHandler (ICHARG 2)
                keep = gapped and profile is not ScfProfile.SLOW
                stale = set(new_settings) - {"sc_iter_limit"} and density_matrices and not keep
                new_settings = {f"parameters->{key}": val for key, val in new_settings.items()}
                actions.append({"dict": "control.in", "action": {"_set": new_settings}})
                if stale:
                    actions.append({"file": "D_spin_*.csc", "action": {"_file_delete": {"mode": "actual"}}})
            if actions or same_settings:
                if monitored:
                    actions.append({"file": "abort_scf", "action": {"_file_delete": {"mode": "actual"}}})
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


class ScfMonitorHandler(ErrorHandler):
    """
    Monitor the SCF cycle while FHI-aims runs, and stop it when it is not converging.

    Checkpoints come every checkpoint_interval iterations (at iteration checkpoint_interval + 1,
    2 * checkpoint_interval + 1, ..., where "elsi_restart read_and_write checkpoint_interval" writes
    the density matrices). Once the SCF cycle has passed the first checkpoint, every check
    classifies its last checkpoint_interval iterations with custodian.aims.utils.scf_profile. If
    they are not SLOW, the handler writes abort_scf, on which FHI-aims stops at the end of the
    iteration in progress, and UnconvergedErrorHandler corrects the stopped run, deciding whether
    the next run restarts from the density matrices of the last checkpoint. In a relaxation, every
    ionic step is monitored.

    Unlike NonConvergingErrorHandler for VASP, which terminates a run whose last ionic steps all hit
    NELM, it looks inside the SCF cycle and lets FHI-aims stop by itself, like WalltimeHandler with
    STOPCAR, so that the density matrices on disk are complete.

    The control.in settings it relies on are in settings_override, for AimsJob (atomate2 passes
    them). Without them, the next run starts from scratch.
    """

    is_monitor = True
    # FHI-aims stops by itself on abort_scf, at the end of an iteration
    is_terminating = False

    def __init__(
        self,
        output_filename: str = "aims.out",
        checkpoint_interval: int = 150,
        restart: bool = True,
        sc_iter_limit: int = 1000,
    ) -> None:
        """Initialize the handler with the output file to check.

        Args:
            output_filename (str): Filename for the FHI-aims output. Change
                this only if it is different from the default (unlikely).
            checkpoint_interval (int): Number of SCF iterations classified at each check, and
                between two writes of the density matrices. Defaults to 150.
            restart (bool): Whether FHI-aims writes the density matrices, from which the next run
                may restart (see UnconvergedErrorHandler). Defaults to True.
            sc_iter_limit (int): sc_iter_limit set in control.in. Defaults to 1000.
        """
        self.output_filename = output_filename
        self.checkpoint_interval = checkpoint_interval
        self.restart = restart
        self.sc_iter_limit = sc_iter_limit

    @property
    def settings_override(self) -> list[dict]:
        """The control.in settings the handler relies on, for AimsJob(settings_override=...)."""
        settings: dict[str, int | str] = {"parameters->sc_iter_limit": self.sc_iter_limit}
        if self.restart:
            settings["parameters->elsi_restart"] = f"read_and_write {self.checkpoint_interval}"
        return [{"dict": "control.in", "action": {"_set": settings}}]

    def _checkpoint(self, directory: str) -> tuple[list[dict], int]:
        """The SCF iterations of the cycle in progress, and the last checkpoint they passed (0 if none)."""
        out = load_aims_stdout(os.path.join(directory, self.output_filename))
        if out.errors or out.is_finished_ok:  # FHI-aims has stopped: UnconvergedErrorHandler takes over
            return [], 0
        steps = [step for step in out.results["ionic_steps"][-1]["scf_steps"] if "charge_density_change" in step]
        n_checks = (len(steps) - 1) // self.checkpoint_interval
        return steps, n_checks * self.checkpoint_interval + 1 if n_checks > 0 else 0

    def check(self, directory="./") -> bool:
        """Check for error."""
        if os.path.isfile(os.path.join(directory, "abort_scf")):  # already stopping
            return False
        try:
            steps, checkpoint = self._checkpoint(directory)
            # Past the first checkpoint, and not converging over the last checkpoint_interval iterations
            return checkpoint > 0 and scf_profile(steps[-self.checkpoint_interval :]) is not ScfProfile.SLOW
        except Exception:
            return False

    def correct(self, directory="./"):
        """Perform corrections."""
        steps, _ = self._checkpoint(directory)
        profile = scf_profile(steps[-self.checkpoint_interval :])
        # FHI-aims only checks that abort_scf exists: its content tells UnconvergedErrorHandler why
        actions = [{"file": "abort_scf", "action": {"_file_create": {"content": profile.value}}}]
        if self.restart and not glob("D_spin_*.csc", root_dir=directory):
            logger.warning("No density matrices to restart from: pass settings_override to AimsJob")
        AimsModder(directory=directory).apply_actions(actions)
        return {"errors": ["Non-converging SCF", profile.value], "actions": actions}


class RoundChargeErrorHandler(ErrorHandler):
    """
    Check if FHI-aims stopped because the initial charges in geometry.in do not add up to the
    charge in control.in, which it requires within 1e-5.

    Initial charges taken from a previous run, e.g. its Mulliken charges, are rounded: summed
    over many atoms, they can miss the charge by that much. Within charge_tol the mismatch is
    taken as rounding, so the charges are kept and the check is overridden; a larger one is an
    input error and is not corrected.
    """

    is_monitor = False

    def __init__(self, output_filename: str = "aims.out", charge_tol: float = 1e-3) -> None:
        """Initialize the handler with the output file to check.

        Args:
            output_filename (str): Filename for the FHI-aims output. Change
                this only if it is different from the default (unlikely).
            charge_tol (float): Largest mismatch between the sum of the initial charges and
                the charge of the system that is taken as rounding. Defaults to 1e-3.
        """
        self.output_filename = output_filename
        self.charge_tol = charge_tol

    def check(self, directory="./") -> bool:
        """Check for error."""
        msg = "Error: Initial charge and total charge of the system are inconsistent."
        try:
            out = load_aims_stdout(os.path.join(directory, self.output_filename))
            if any(msg in error for error in out.errors):
                return True
        except Exception:
            pass
        return False

    def correct(self, directory="./"):
        """Perform corrections."""
        charge = float(AimsControl.from_file(os.path.join(directory, "control.in")).parameters.get("charge", 0))
        initial_charges = AimsGeometry.from_file(os.path.join(directory, "geometry.in")).initial_charges or []
        actions = []
        if abs(sum(initial_charges) - charge) <= self.charge_tol:
            actions.append(
                {"dict": "control.in", "action": {"_set": {"parameters->override_initial_charge_check": True}}}
            )

        if actions:
            backup(AIMS_BACKUP_FILES, directory=directory)
            AimsModder(directory=directory).apply_actions(actions)
            return {"errors": ["RoundCharge"], "actions": actions}

        # Unfixable error. Just return None for actions.
        return {"errors": ["RoundCharge"], "actions": None}
