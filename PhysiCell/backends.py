"""
Simulation execution backends for the PhysiCell MCP server.

Two implementations behind a single Protocol:

- LocalBackend  : wraps subprocess.Popen, current default, runs on the host
                  the MCP server is running on. Handle format: "local:<pid>".
- SlurmBackend  : wraps sbatch / squeue / sacct / scancel for HPC clusters.
                  Handle format: "slurm:<job_id>" or "slurm:<job_id>_<task>"
                  for array elements.

The backend is selected per-call via an explicit argument, falling back to the
PHYSICELL_BACKEND environment variable, then to "local". This keeps existing
local-only workflows untouched while enabling SLURM dispatch where it makes
sense (cluster head node + interactive allocation, large UQ sweeps, calibration).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol, Literal, Any


# ---------------------------------------------------------------------------
# Data types shared by both backends
# ---------------------------------------------------------------------------

JobState = Literal["pending", "running", "completed", "failed", "stopped", "unknown"]


@dataclass
class Resources:
    """Resource request for a backend submission.

    For LocalBackend most fields are ignored — only `cpus` is used to set
    OMP_NUM_THREADS. SlurmBackend uses every field.
    """
    partition: str = "batch"
    account: str = "ChangLab"
    cpus: int = 16
    mem: str = "32G"
    time: str = "12:00:00"
    array_concurrent: Optional[int] = None  # only used for submit_array


@dataclass
class JobStatus:
    state: JobState
    return_code: Optional[int] = None
    raw: str = ""  # backend-native status string (for display / debugging)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SimBackend(Protocol):
    name: str

    def submit(
        self,
        cmd: list[str],
        cwd: Path,
        log: Path,
        resources: Resources,
        env: Optional[dict] = None,
        job_name: str = "physicell",
    ) -> str:
        """Submit a single simulation. Returns an opaque handle ("local:PID" or "slurm:JOB")."""
        ...

    def submit_array(
        self,
        manifest: Path,
        cwd: Path,
        log_dir: Path,
        resources: Resources,
        env: Optional[dict] = None,
        job_name: str = "physicell-array",
    ) -> tuple[str, int]:
        """Submit an array of simulations. Manifest file: one task per line.
        Returns (array_handle, n_tasks). Array handle format: "slurm:<jobid>" (group)."""
        ...

    def poll(self, handle: str) -> JobStatus:
        """Get current status of a submitted job."""
        ...

    def cancel(self, handle: str) -> None:
        """Cancel a running job."""
        ...


# ---------------------------------------------------------------------------
# LocalBackend: wraps subprocess.Popen
# ---------------------------------------------------------------------------


# Module-level registry of running local processes, keyed by PID.
# Indexed lazily so callers can poll from a different code path than submit.
_local_procs: dict[int, subprocess.Popen] = {}


class LocalBackend:
    name = "local"

    def submit(
        self,
        cmd: list[str],
        cwd: Path,
        log: Path,
        resources: Resources,
        env: Optional[dict] = None,
        job_name: str = "physicell",
    ) -> str:
        log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log, "w")
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        # Default OMP threads to requested cpus when not explicitly set
        run_env.setdefault("OMP_NUM_THREADS", str(resources.cpus))
        try:
            # Accept both list-form and single-string commands. PhysiCell's
            # existing call site uses a single shell string ("./project cfg.xml")
            # so support both.
            if isinstance(cmd, str):
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                    env=run_env,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                    env=run_env,
                )
        finally:
            log_handle.close()  # subprocess inherits the fd

        _local_procs[proc.pid] = proc
        return f"local:{proc.pid}"

    def submit_array(self, *args, **kwargs) -> tuple[str, int]:
        raise NotImplementedError(
            "LocalBackend does not support array submission. "
            "Loop submit() instead, or use SlurmBackend."
        )

    def poll(self, handle: str) -> JobStatus:
        if not handle.startswith("local:"):
            return JobStatus(state="unknown", raw=f"not a local handle: {handle}")
        try:
            pid = int(handle.split(":", 1)[1])
        except (ValueError, IndexError):
            return JobStatus(state="unknown", raw=f"malformed handle: {handle}")

        proc = _local_procs.get(pid)
        if proc is None:
            # Process was submitted by a previous server instance; we cannot
            # recover the Popen object. Best-effort: check /proc on Linux.
            if Path(f"/proc/{pid}").exists():
                return JobStatus(state="running", raw=f"pid {pid} alive (foreign)")
            return JobStatus(state="unknown", raw=f"pid {pid} not tracked")

        rc = proc.poll()
        if rc is None:
            return JobStatus(state="running", raw=f"pid {pid}")
        if rc == 0:
            return JobStatus(state="completed", return_code=0, raw=f"pid {pid} exit 0")
        return JobStatus(state="failed", return_code=rc, raw=f"pid {pid} exit {rc}")

    def cancel(self, handle: str) -> None:
        if not handle.startswith("local:"):
            return
        try:
            pid = int(handle.split(":", 1)[1])
        except (ValueError, IndexError):
            return
        proc = _local_procs.get(pid)
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SlurmBackend: wraps sbatch / squeue / sacct / scancel
# ---------------------------------------------------------------------------


# Map common SLURM job states to our 5-state model.
_SLURM_STATE_MAP = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "REQUEUED": "pending",
    "RESIZING": "running",
    "RUNNING": "running",
    "SUSPENDED": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "CANCELLED": "stopped",
    "DEADLINE": "failed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
    "OUT_OF_MEMORY": "failed",
    "TIMEOUT": "failed",
    "BOOT_FAIL": "failed",
}


def slurm_available() -> bool:
    """Return True if sbatch is on PATH."""
    return shutil.which("sbatch") is not None


def _sbatch_template_path() -> Path:
    """Resolve the bundled physicell.sbatch script path."""
    return Path(__file__).parent / "scripts" / "physicell.sbatch"


class SlurmBackend:
    name = "slurm"

    def __init__(self, sbatch_template: Optional[Path] = None):
        self.template = sbatch_template or _sbatch_template_path()
        if not self.template.exists():
            raise FileNotFoundError(
                f"SLURM sbatch template not found at {self.template}. "
                "This file ships with the MCP server — check installation."
            )

    # ----------------------- single submit -----------------------

    def submit(
        self,
        cmd: list[str],
        cwd: Path,
        log: Path,
        resources: Resources,
        env: Optional[dict] = None,
        job_name: str = "physicell",
    ) -> str:
        """Submit a PhysiCell run as a single SLURM job.

        `cmd` is expected to be of the form ["./project", "config/...xml"]
        (or a string variant). We extract the executable and config and pass
        them as environment variables to the sbatch template, so the template
        is the source of truth for the actual invocation.
        """
        log.parent.mkdir(parents=True, exist_ok=True)
        executable, config_arg = _split_physicell_cmd(cmd)

        export_vars = {
            "PROJECT_DIR": str(cwd),
            "PHYSICELL_EXEC": executable,
            "PHYSICELL_CONFIG": config_arg,
            "OMP_NUM_THREADS": str(resources.cpus),
        }
        if env:
            export_vars.update({k: str(v) for k, v in env.items()})

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--partition={resources.partition}",
            f"--account={resources.account}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.mem}",
            f"--time={resources.time}",
            f"--output={log}",
            f"--error={log}",
            f"--export=ALL,{_format_export(export_vars)}",
            str(self.template),
        ]

        proc = subprocess.run(
            sbatch_cmd, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        # --parsable returns "<jobid>" or "<jobid>;<cluster>"
        job_id = proc.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"sbatch returned unexpected output: {proc.stdout!r}")
        return f"slurm:{job_id}"

    # ----------------------- array submit -----------------------

    def submit_array(
        self,
        manifest: Path,
        cwd: Path,
        log_dir: Path,
        resources: Resources,
        env: Optional[dict] = None,
        job_name: str = "physicell-array",
    ) -> tuple[str, int]:
        """Submit one SLURM array job whose tasks read `manifest`.

        Manifest format: one line per task, with three tab-separated fields:
            <project_dir>\\t<config_path>\\t<extra_env>

        `extra_env` is an optional KEY=VAL[,KEY=VAL...] string, exported into
        the task before invoking the executable. Use it for parameter sweeps.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest = Path(manifest).resolve()
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest}")

        with manifest.open() as f:
            n_tasks = sum(1 for line in f if line.strip())
        if n_tasks == 0:
            raise ValueError(f"Manifest is empty: {manifest}")

        concurrent = resources.array_concurrent or n_tasks
        array_spec = f"0-{n_tasks - 1}"
        if concurrent < n_tasks:
            array_spec += f"%{concurrent}"

        export_vars = {
            "PROJECT_DIR": str(cwd),
            "PHYSICELL_MANIFEST": str(manifest),
            "OMP_NUM_THREADS": str(resources.cpus),
        }
        if env:
            export_vars.update({k: str(v) for k, v in env.items()})

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--partition={resources.partition}",
            f"--account={resources.account}",
            f"--array={array_spec}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.mem}",
            f"--time={resources.time}",
            f"--output={log_dir}/slurm-%A_%a.out",
            f"--error={log_dir}/slurm-%A_%a.out",
            f"--export=ALL,{_format_export(export_vars)}",
            str(self.template),
        ]

        proc = subprocess.run(
            sbatch_cmd, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch (array) failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        job_id = proc.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"sbatch returned unexpected output: {proc.stdout!r}")
        return f"slurm:{job_id}", n_tasks

    # ----------------------- poll -----------------------

    def poll(self, handle: str) -> JobStatus:
        if not handle.startswith("slurm:"):
            return JobStatus(state="unknown", raw=f"not a slurm handle: {handle}")
        job_spec = handle.split(":", 1)[1]  # may be "12345" or "12345_7"
        # Try squeue first (cheap, always-fresh for active jobs)
        sq = subprocess.run(
            ["squeue", "-j", job_spec, "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=15,
        )
        if sq.returncode == 0 and sq.stdout.strip():
            slurm_state = sq.stdout.strip().split("\n")[0]
            return JobStatus(
                state=_SLURM_STATE_MAP.get(slurm_state, "unknown"),
                raw=slurm_state,
            )

        # Fall back to sacct for terminal states (squeue forgets after a few
        # minutes). Use --parsable2 (no trailing pipe) and -X (one row per job
        # rather than one per step).
        sa = subprocess.run(
            ["sacct", "-j", job_spec, "-X", "-n", "--parsable2",
             "-o", "State,ExitCode"],
            capture_output=True, text=True, timeout=15,
        )
        if sa.returncode != 0 or not sa.stdout.strip():
            return JobStatus(state="unknown", raw=f"sacct empty for {job_spec}")

        # First line wins (most recent / array head). For arrays sacct returns
        # multiple rows; we collapse to the worst case.
        lines = [l.strip() for l in sa.stdout.strip().splitlines() if l.strip()]
        states = []
        rcs = []
        for line in lines:
            parts = line.split("|")
            if not parts:
                continue
            slurm_state = parts[0].split()[0]  # "CANCELLED by 12345" -> "CANCELLED"
            states.append(_SLURM_STATE_MAP.get(slurm_state, "unknown"))
            if len(parts) > 1 and parts[1]:
                try:
                    rcs.append(int(parts[1].split(":")[0]))
                except ValueError:
                    pass

        # If any task is still running, report running; else any failure -> failed;
        # else completed.
        if any(s == "running" for s in states):
            return JobStatus(state="running", raw=";".join(states))
        if any(s == "failed" for s in states):
            return JobStatus(
                state="failed",
                return_code=max(rcs) if rcs else None,
                raw=";".join(states),
            )
        if any(s == "stopped" for s in states):
            return JobStatus(state="stopped", raw=";".join(states))
        if all(s == "completed" for s in states):
            return JobStatus(state="completed", return_code=0, raw=";".join(states))
        if any(s == "pending" for s in states):
            return JobStatus(state="pending", raw=";".join(states))
        return JobStatus(state="unknown", raw=";".join(states))

    # ----------------------- cancel -----------------------

    def cancel(self, handle: str) -> None:
        if not handle.startswith("slurm:"):
            return
        job_spec = handle.split(":", 1)[1]
        subprocess.run(["scancel", job_spec], capture_output=True, text=True, timeout=15)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


_BACKEND_CACHE: dict[str, SimBackend] = {}


def get_backend(name: Optional[str] = None) -> SimBackend:
    """Resolve a backend by name. Order: arg > PHYSICELL_BACKEND env var > 'local'."""
    chosen = (name or os.environ.get("PHYSICELL_BACKEND") or "local").lower()
    if chosen in _BACKEND_CACHE:
        return _BACKEND_CACHE[chosen]
    if chosen == "local":
        backend = LocalBackend()
    elif chosen == "slurm":
        if not slurm_available():
            raise RuntimeError(
                "SLURM backend requested but `sbatch` is not on PATH. "
                "Run on a cluster head or login node, or set PHYSICELL_BACKEND=local."
            )
        backend = SlurmBackend()
    else:
        raise ValueError(f"Unknown backend: {chosen!r}. Use 'local' or 'slurm'.")
    _BACKEND_CACHE[chosen] = backend
    return backend


# ---------------------------------------------------------------------------
# Persistence: ~/.physicell-mcp/jobs.jsonl
# ---------------------------------------------------------------------------


def jobs_db_path() -> Path:
    p = Path(os.environ.get("PHYSICELL_MCP_HOME", Path.home() / ".physicell-mcp"))
    p.mkdir(parents=True, exist_ok=True)
    return p / "jobs.jsonl"


def persist_job(record: dict) -> None:
    """Append a job record to the persistent log. Best-effort, never raises."""
    try:
        path = jobs_db_path()
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def load_persisted_jobs() -> list[dict]:
    """Read all persisted job records (in order). Returns [] on missing/invalid."""
    path = jobs_db_path()
    if not path.exists():
        return []
    out = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_physicell_cmd(cmd) -> tuple[str, str]:
    """Pull executable + config arg out of either ["./project", "cfg.xml"] or
    "./project cfg.xml". Return (exec_path, config_path)."""
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    else:
        parts = list(cmd)
    if not parts:
        raise ValueError("Empty command")
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _format_export(d: dict[str, str]) -> str:
    """Format a dict as a SLURM --export-friendly comma-separated list.

    SLURM splits on commas, so values containing commas are not safely escapable
    via this mechanism. We refuse to silently drop them — call sites should
    pass simple paths/integers.
    """
    out = []
    for k, v in d.items():
        v = str(v)
        if "," in v:
            raise ValueError(
                f"--export value for {k} contains comma; SLURM cannot escape it: {v!r}"
            )
        out.append(f"{k}={v}")
    return ",".join(out)
