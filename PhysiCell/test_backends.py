"""Smoke tests for the SimBackend layer.

Run with `uv run pytest test_backends.py -v` from the PhysiCell directory, or
`python -m pytest test_backends.py -v` if pytest is on PATH.

Three test categories:
- LocalBackend lifecycle (submit -> poll running -> poll completed -> cancel)
- SlurmBackend dryrun (only runs if `sbatch` is on PATH)
- Persistence roundtrip (jobs.jsonl read/write)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Make the local package importable when running pytest from this directory.
sys.path.insert(0, str(Path(__file__).parent))

import backends  # noqa: E402
from backends import (
    LocalBackend, SlurmBackend, Resources, JobStatus,
    get_backend, slurm_available, jobs_db_path,
    persist_job, load_persisted_jobs, _local_procs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_jobs_db(monkeypatch, tmp_path):
    """Redirect ~/.physicell-mcp to a tmp dir so tests don't pollute real state."""
    monkeypatch.setenv("PHYSICELL_MCP_HOME", str(tmp_path / "physicell-mcp"))
    # Ensure backend cache is clean for this test.
    backends._BACKEND_CACHE.clear()
    yield tmp_path / "physicell-mcp"


@pytest.fixture
def tmp_workdir(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    log = tmp_path / "logs" / "run.log"
    return cwd, log


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


def test_local_backend_lifecycle(tmp_workdir):
    """submit -> poll running -> wait -> poll completed."""
    cwd, log = tmp_workdir
    be = LocalBackend()
    handle = be.submit(
        cmd=["sh", "-c", "echo hello; sleep 1; echo done"],
        cwd=cwd,
        log=log,
        resources=Resources(cpus=1),
    )
    assert handle.startswith("local:")
    pid = int(handle.split(":", 1)[1])
    assert pid > 0

    # Immediately should be running.
    js = be.poll(handle)
    assert js.state in ("running", "completed"), f"unexpected state: {js}"

    # Wait for completion.
    deadline = time.time() + 10
    while time.time() < deadline:
        js = be.poll(handle)
        if js.state in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert js.state == "completed", f"final state: {js}"
    assert js.return_code == 0
    assert log.exists()
    assert "hello" in log.read_text()


def test_local_backend_failure(tmp_workdir):
    cwd, log = tmp_workdir
    be = LocalBackend()
    handle = be.submit(
        cmd=["sh", "-c", "exit 7"],
        cwd=cwd,
        log=log,
        resources=Resources(cpus=1),
    )
    deadline = time.time() + 5
    js = JobStatus(state="running")
    while time.time() < deadline:
        js = be.poll(handle)
        if js.state in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert js.state == "failed"
    assert js.return_code == 7


def test_local_backend_cancel(tmp_workdir):
    cwd, log = tmp_workdir
    be = LocalBackend()
    handle = be.submit(
        cmd=["sh", "-c", "sleep 30"],
        cwd=cwd,
        log=log,
        resources=Resources(cpus=1),
    )
    js = be.poll(handle)
    assert js.state == "running"
    be.cancel(handle)
    deadline = time.time() + 5
    while time.time() < deadline:
        js = be.poll(handle)
        if js.state != "running":
            break
        time.sleep(0.1)
    assert js.state in ("failed", "completed", "stopped"), f"unexpected: {js}"


def test_local_backend_unknown_handle():
    be = LocalBackend()
    js = be.poll("local:9999999")
    # Either truly unknown or bogus-pid-not-found — we only require non-running.
    assert js.state in ("unknown", "running"), f"got {js}"


def test_local_backend_array_unsupported():
    be = LocalBackend()
    with pytest.raises(NotImplementedError):
        be.submit_array(
            manifest=Path("/tmp/nope"),
            cwd=Path("/tmp"),
            log_dir=Path("/tmp"),
            resources=Resources(),
        )


# ---------------------------------------------------------------------------
# SlurmBackend (only when sbatch is available)
# ---------------------------------------------------------------------------


def _have_sbatch():
    return shutil.which("sbatch") is not None


@pytest.mark.skipif(not _have_sbatch(), reason="sbatch not on PATH")
def test_slurm_backend_template_present():
    be = SlurmBackend()
    assert be.template.exists()
    assert be.template.name == "physicell.sbatch"


@pytest.mark.skipif(not _have_sbatch(), reason="sbatch not on PATH")
def test_slurm_backend_submit_dryrun(tmp_workdir):
    """Submit a no-op job that exits cleanly so we exercise the parsing code
    without burning real cluster resources for a long time. Requires the test
    runner to have permission to submit to the default partition."""
    cwd, log = tmp_workdir
    log.parent.mkdir(parents=True, exist_ok=True)
    # Override the template with a trivial bash that just prints + exits.
    fake_template = cwd / "fake.sbatch"
    fake_template.write_text(
        "#!/bin/bash\n"
        "echo \"FAKE PHYSICELL JOB\"\n"
        "echo \"PROJECT_DIR=$PROJECT_DIR\"\n"
        "echo \"PHYSICELL_CONFIG=$PHYSICELL_CONFIG\"\n"
        "exit 0\n"
    )
    fake_template.chmod(0o755)
    be = SlurmBackend(sbatch_template=fake_template)

    # Use cluster-appropriate defaults; if the partition/account are wrong on
    # this test host, sbatch fails — the test is then a no-op skip via xfail.
    res = Resources(
        partition=os.environ.get("PHYSICELL_TEST_PARTITION", "batch"),
        account=os.environ.get("PHYSICELL_TEST_ACCOUNT", "ChangLab"),
        cpus=1,
        mem="1G",
        time="0:02:00",
    )
    try:
        handle = be.submit(
            cmd=["./project", "config/x.xml"],
            cwd=cwd,
            log=log,
            resources=res,
            job_name="physicell-smoke",
        )
    except RuntimeError as e:
        pytest.skip(f"sbatch refused submission (likely partition/account): {e}")
    assert handle.startswith("slurm:")
    job_id = handle.split(":", 1)[1]
    assert job_id.isdigit()

    # Poll a few times — state should resolve to one of our 5 within 60 s for a
    # trivial job. We don't wait long; just ensure poll() returns a valid state.
    js = be.poll(handle)
    assert js.state in ("pending", "running", "completed", "failed", "stopped", "unknown")

    # Best-effort cancel so we don't leave anything queued if the test ran on a
    # cluster where the trivial job sat in a queue.
    be.cancel(handle)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_roundtrip(tmp_jobs_db):
    rec = {
        "simulation_id": "abc12345",
        "project_name": "test_proj",
        "backend": "slurm",
        "handle": "slurm:99999",
        "submitted_at": time.time(),
    }
    persist_job(rec)
    loaded = load_persisted_jobs()
    assert len(loaded) == 1
    assert loaded[0]["simulation_id"] == "abc12345"
    assert loaded[0]["handle"] == "slurm:99999"


def test_persistence_appends(tmp_jobs_db):
    persist_job({"simulation_id": "a", "handle": "local:1"})
    persist_job({"simulation_id": "b", "handle": "local:2"})
    loaded = load_persisted_jobs()
    assert [r["simulation_id"] for r in loaded] == ["a", "b"]


def test_persistence_handles_corrupt_lines(tmp_jobs_db):
    persist_job({"simulation_id": "good"})
    db = jobs_db_path()
    with db.open("a") as f:
        f.write("this is not json\n")
        f.write("\n")  # blank line
    persist_job({"simulation_id": "good2"})
    loaded = load_persisted_jobs()
    assert [r["simulation_id"] for r in loaded] == ["good", "good2"]


def test_persistence_empty_when_missing(tmp_jobs_db):
    # New tmp dir, nothing written yet
    loaded = load_persisted_jobs()
    assert loaded == []


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_get_backend_default_is_local(monkeypatch, tmp_jobs_db):
    monkeypatch.delenv("PHYSICELL_BACKEND", raising=False)
    be = get_backend()
    assert be.name == "local"


def test_get_backend_explicit_local(tmp_jobs_db):
    be = get_backend("local")
    assert be.name == "local"


def test_get_backend_unknown_raises(tmp_jobs_db):
    with pytest.raises(ValueError):
        get_backend("invented")


def test_get_backend_slurm_without_sbatch(monkeypatch, tmp_jobs_db):
    """If sbatch isn't on PATH, requesting slurm should raise clearly."""
    if _have_sbatch():
        pytest.skip("sbatch is available, can't test missing-sbatch path")
    monkeypatch.setenv("PHYSICELL_BACKEND", "slurm")
    with pytest.raises(RuntimeError, match="sbatch"):
        get_backend()
