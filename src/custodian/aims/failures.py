"""
Report of the FHI-aims runs of a job under custodian, kept when the job fails.

For every FHI-aims run (attempt) in a directory, the report gathers what the handlers saw and
decided: the change of the density at each SCF iteration of the last ionic step, with its gap and
wall time; the SCF settings and the convergence threshold; whether the density started from saved
density matrices; the classification of the cycle (custodian.aims.utils.scf_profile_metrics and
scf_gap); the corrections. build_report reads it all from the files the runs leave, so it works on
any directory, also after the job ended:
- the earlier runs are in the backups of the handlers (error.N.tar.gz), the last one is in the
  directory;
- custodian rewrites custodian.json after every check, so it holds every correction but the one
  being made, which the handler passes as pending.
Only the checks of ScfMonitorHandler that do not stop a run leave no trace in any file: the handler
records them as they happen (record_check). UnconvergedErrorHandler writes the report at every
correction, before its backup, so that it is there however the job ends; AimsJob removes it when
the job succeeds, unless keep_failures_report is set. The report knows nothing of how custodian
was launched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tarfile
import tempfile
from datetime import UTC, datetime
from glob import glob

from monty.serialization import dumpfn, loadfn
from pyfhiaims.control.control import AimsControl
from pyfhiaims.geometry.geometry import AimsGeometry
from pyfhiaims.outputs.stdout import AimsStdout

from custodian.aims.io import load_aims_stdout
from custodian.aims.utils import GAP_THRESHOLD, scf_gap, scf_profile_metrics

logger = logging.getLogger(__name__)

REPORT_FILE = "custodian_failures.json"
SCHEMA_VERSION = 1
# The handlers whose corrections stop a run: the correction of the next handler restarts it.
MONITORS = frozenset({"ScfMonitorHandler"})
# The control.in parameters that shape the SCF cycle.
SCF_SETTINGS = (
    "xc",
    "spin",
    "occupation_type",
    "mixer",
    "n_max_pulay",
    "charge_mix_param",
    "spin_mix_param",
    "prec_mix_param",
    "preconditioner",
    "adjust_scf",
    "adjust_scf_param",
    "sc_iter_limit",
    "sc_accuracy_rho",
    "elsi_restart",
)
RESTARTED = "Restarting calculation from saved density matrices"
# Written by scf_solver.f90 at the end of the iteration in which it found abort_scf
ABORTED = "Abort request found"


def current_attempt(directory: str = "./") -> int:
    """The number of the FHI-aims run in progress: one more than the backups of the earlier ones.

    Args:
        directory: The run directory.

    Returns:
        int: The number of the run, from 1.
    """
    return len(_backups(directory)) + 1


def record_check(directory: str, scf_steps: list[dict], checkpoint: int, window: int, stop: bool) -> None:
    """Add a check of ScfMonitorHandler to the report; a failure is logged, never raised.

    Args:
        directory: The run directory.
        scf_steps: The SCF iterations of the cycle in progress.
        checkpoint: The last checkpoint the cycle passed (0 if none): only past one is the cycle
            classified.
        window: The number of iterations the check classifies (the checkpoint interval).
        stop: Whether the check stops the run.
    """
    try:
        profile = slope = scatter = None
        if checkpoint > 0:
            profile, slope, scatter = scf_profile_metrics(scf_steps[-window:])
        report = _load(directory)
        report.setdefault("checks", []).append(
            {
                "attempt": current_attempt(directory),
                "time": _now(),
                "iteration": len(scf_steps),
                "checkpoint": checkpoint,
                "profile": profile.value if profile else None,
                "slope": slope,
                "scatter": scatter,
                "stop": stop,
            }
        )
        _dump(directory, report)
    except Exception:
        logger.warning(f"Could not record the check in {REPORT_FILE}", exc_info=True)


def write_report(directory: str = "./", pending: dict | None = None, output_filename: str = "aims.out") -> dict | None:
    """Build the report (see build_report) and write it to REPORT_FILE; a failure is logged, never raised.

    Args:
        directory: The run directory.
        pending: The correction being made, not yet in custodian.json.
        output_filename: The FHI-aims output of every run.

    Returns:
        dict | None: The report, or None if it could not be written.
    """
    try:
        report = build_report(directory, pending, output_filename)
        _dump(directory, report)
    except Exception:
        logger.warning(f"Could not write {REPORT_FILE}", exc_info=True)
        return None
    return report


def build_report(directory: str = "./", pending: dict | None = None, output_filename: str = "aims.out") -> dict:
    """Build the report of the FHI-aims runs in a directory from the files they left.

    Args:
        directory: The run directory.
        pending: The correction being made, not yet in custodian.json, as
            {"handler": class name, "errors": [...], "actions": [...] or None}.
        output_filename: The FHI-aims output of every run.

    Returns:
        dict: The structure (geometry.in), the SCF settings before custodian changed them
            (control.in.orig), the status of the last run, one entry per run with its corrections,
            and the checks ScfMonitorHandler recorded.
    """
    corrections = _corrections(directory, pending)
    with tempfile.TemporaryDirectory() as tmp:
        attempts = [
            {
                "attempt": number,
                "source": source,
                **_run(run_dir, output_filename, cached=run_dir == directory),
                "corrections": corrections[number - 1] if number <= len(corrections) else [],
            }
            for number, (source, run_dir) in enumerate(_runs(directory, tmp, output_filename), start=1)
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "directory": os.path.abspath(directory),
        "updated": _now(),
        "structure": _structure(os.path.join(directory, "geometry.in")),
        "input": _settings(os.path.join(directory, "control.in.orig")),
        "status": _status(attempts),
        "attempts": attempts,
        "checks": _load(directory).get("checks", []),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _backups(directory: str) -> list[str]:
    """The backups of the earlier runs (custodian.utils.backup), oldest first."""
    numbered = []
    for path in glob(os.path.join(directory, "error.*.tar*")):
        match = re.fullmatch(r"error\.(\d+)\.tar(\.gz)?", os.path.basename(path))
        if match:
            numbered.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered)]


def _digest(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def _runs(directory: str, tmp: str, output_filename: str) -> list[tuple[str, str]]:
    """The directories of the runs, oldest first: the backups, extracted in tmp, then the directory,
    unless it repeats the last backup (custodian stopped right after a correction).
    """
    runs, last = [], None
    for archive in _backups(directory):
        name = os.path.basename(archive)
        run_dir = os.path.join(tmp, name.split(".tar")[0])
        os.makedirs(run_dir)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and os.path.basename(member.name) in (output_filename, "control.in"):
                    with open(os.path.join(run_dir, os.path.basename(member.name)), "wb") as file:
                        file.write(tar.extractfile(member).read())
        output = os.path.join(run_dir, output_filename)
        if os.path.isfile(output):
            runs.append((name, run_dir))
            last = output
    output = os.path.join(directory, output_filename)
    if os.path.isfile(output) and (last is None or _digest(output) != _digest(last)):
        runs.append((".", directory))
    return runs


def _run(run_dir: str, output_filename: str, cached: bool) -> dict:
    """What a run did: its settings and start, the SCF iterations of its last ionic step, and their
    classification as the handlers made it.

    A run stopped by abort_scf was classified by the check of ScfMonitorHandler that wrote it: over
    the checkpoint interval before the iteration FHI-aims was doing, which it ended before stopping.
    UnconvergedErrorHandler classifies any other run over its whole cycle.
    """
    path = os.path.join(run_dir, output_filename)
    out = load_aims_stdout(path) if cached else AimsStdout(path)
    with open(path, errors="replace") as file:
        text = file.read()
    settings = _settings(os.path.join(run_dir, "control.in"))
    ionic_steps = out.results.get("ionic_steps") or []
    steps = [step for step in ionic_steps[-1]["scf_steps"] if "charge_density_change" in step] if ionic_steps else []
    aborted = ABORTED in text
    interval = re.search(r"write\s+(\d+)", str(settings.get("elsi_restart", "")))
    last = len(steps) - 1 if aborted else len(steps)
    first = max(last - int(interval.group(1)), 0) if aborted and interval else 0
    profile, slope, scatter = scf_profile_metrics(steps[first:last])
    gap = scf_gap(steps)
    if "sc_accuracy_rho" in settings:
        threshold, set_by = float(str(settings["sc_accuracy_rho"]).split()[0]), "control.in"
    else:
        match = re.search(r"chose sc_accuracy_rho\s*=\s*([0-9.Ee+-]+)", text)
        threshold, set_by = (float(match.group(1)), "FHI-aims") if match else (None, None)
    started = re.search(r"Date\s*:\s*(\d{8}),\s*Time\s*:\s*(\d{6})", text)
    return {
        # as FHI-aims prints it, in the time zone of the node
        "started": datetime.strptime("".join(started.groups()), "%Y%m%d%H%M%S").isoformat() if started else None,
        "restarted": RESTARTED in text,
        "settings": settings,
        "sc_accuracy_rho": threshold,
        "sc_accuracy_rho_set_by": set_by,
        "n_ionic_steps": len(ionic_steps),
        "converged": bool(ionic_steps) and bool(out.get_image(-1).converged),
        "aborted": aborted,
        "scf": {
            "charge_density_change": [step["charge_density_change"] for step in steps],
            "spin_density_change": [step.get("spin_density_change") for step in steps]
            if any("spin_density_change" in step for step in steps)
            else None,
            "gap": [step.get("gap") for step in steps],
            "wall_time": [step.get("time", {}).get("total", [None])[-1] for step in steps],
        },
        "classification": {
            # first and last iteration classified, from 1
            "iterations": [first + 1, last],
            "profile": profile.value,
            "slope": slope,
            "scatter": scatter,
            "gap": gap,
            "gapped": gap > GAP_THRESHOLD,
        },
    }


def _settings(path: str) -> dict:
    """The SCF settings of a control.in, as FHI-aims reads them; empty without the file."""
    if not os.path.isfile(path):
        return {}
    parameters = AimsControl.from_file(path).parameters
    return {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in parameters.items()
        if key in SCF_SETTINGS
    }


def _structure(path: str) -> dict:
    """The number of atoms and the composition of a geometry.in; empty without the file."""
    if not os.path.isfile(path):
        return {}
    symbols = AimsGeometry.from_file(path).symbols
    return {
        "n_atoms": len(symbols),
        "composition": {symbol: symbols.count(symbol) for symbol in dict.fromkeys(symbols)},
    }


def _corrections(directory: str, pending: dict | None) -> list[list[dict]]:
    """The corrections of custodian.json and the pending one, grouped by run.

    A correction with actions by a handler that is not a monitor ends its run: custodian then backs
    the run up and starts FHI-aims again.
    """
    entries = []
    log = os.path.join(directory, "custodian.json")
    if os.path.isfile(log):
        with open(log) as file:
            for run in json.load(file):
                for correction in run.get("corrections", []):
                    handler = correction.get("handler")
                    entries.append(
                        {
                            "handler": handler.get("@class") if isinstance(handler, dict) else str(handler),
                            "errors": correction.get("errors"),
                            "actions": correction.get("actions"),
                        }
                    )
    if pending is not None:
        entries.append(pending)
    groups, current = [], []
    for entry in entries:
        current.append(entry)
        if entry["handler"] not in MONITORS and entry["actions"]:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _status(attempts: list[dict]) -> str:
    """How the last run ended, as far as the files tell."""
    if not attempts:
        return "no run"
    corrections = attempts[-1]["corrections"]
    if corrections and corrections[-1]["actions"] is None:
        return "no correction left"
    if corrections:
        return "corrected"
    return "converged" if attempts[-1]["converged"] else "unfinished"


def _load(directory: str) -> dict:
    """The report in the directory, or an empty one."""
    path = os.path.join(directory, REPORT_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        return loadfn(path)
    except Exception:
        logger.warning(f"Could not read {path}: it is written again", exc_info=True)
        return {}


def _dump(directory: str, report: dict) -> None:
    """Write the report atomically: a job killed while writing it keeps the previous one."""
    path = os.path.join(directory, REPORT_FILE)
    report["schema_version"] = SCHEMA_VERSION
    dumpfn(report, f"{path}.tmp", fmt="json", indent=1)
    os.replace(f"{path}.tmp", path)
