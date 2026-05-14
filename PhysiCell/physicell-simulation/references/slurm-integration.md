# SLURM / HPC integration for the PhysiCell MCP

This reference documents the SLURM-aware execution path. The MCP server still defaults to local subprocess execution — nothing here is required for laptop/single-host operation. Read this when running on a cluster.

## When to switch backends

The decision table from `physicell-simulation/SKILL.md §1.5`, expanded with rationale:

| Workload | Why local works / fails | Recommended path |
|---|---|---|
| 1 sim, <2 h, on a workstation | Local Popen is fine | `run_simulation` |
| 1 sim, on a cluster head node | Head nodes ban heavy compute | `submit_simulation_slurm` |
| 2–9 sims | A small loop with `submit_simulation_slurm` is fine; array overhead isn't worth it | loop `submit_simulation_slurm` |
| ≥10 sims, parameter sweeps, replicate batches | A SLURM array gets you fairshare and one job-id to monitor | `submit_simulation_array_slurm` |
| Single sim >2 h | Risk of head-node terminations or local crashes; SLURM walltime is explicit | `submit_simulation_slurm` |
| UQ with `samples × replicates ≥ 50` | Concurrent-futures pool saturates one node | `configure_uq_slurm` then UQ tool |
| BO / ABC calibration on a cluster | Calibrations are long; stay-on-head-node is unsafe | `configure_uq_slurm` then `run_*_calibration` |

The MCP server does not auto-pick — the agent picks based on the workload. Default is local.

## Cluster defaults (OHSU ARC)

| Setting | Value | Notes |
|---|---|---|
| Account | `ChangLab` | Required for billing |
| Partition (default) | `batch` | Up to 1 d 12 h walltime |
| Partition (long) | `gpu`, `a100` | 14 d walltime; only when GPUs needed |
| Partition (interactive) | `interactive` | 7 d walltime; use for `srun --pty bash` sessions |
| Scratch | `/arc/scratch1/$USER` | Use for I/O-heavy output |
| Default cpus | 16 (single sim), 8 (array element) | OMP scales linearly per sim |
| Default mem | 32 G (single sim), 16 G (array element) | PhysiCell sweeps tend to be memory-light |

Override per call when needed (`submit_simulation_slurm(cpus=32, mem='64G', time_limit='1-00:00:00')`).

## Choosing a backend

### From the environment

```bash
export PHYSICELL_BACKEND=slurm   # makes run_simulation itself dispatch to SLURM
export PHYSICELL_BACKEND=local   # explicit local (default if unset)
```

This is the right toggle for a long-running session on the cluster: you set it once when you start `claude` inside an `srun --partition=interactive` allocation, and from then on all `run_simulation` calls go through SLURM with conservative defaults.

### Per call

Every SLURM-specific tool exposes resource arguments (`partition`, `account`, `cpus`, `mem`, `time_limit`). Use these when one specific run needs more (or less) than the defaults.

### For UQ

Set the slurm config once on the session:

```python
configure_uq_slurm(
    partition="batch",
    account="ChangLab",
    cpus=16,
    mem="64G",
    time_limit="1-00:00:00",
    array_concurrent=32,
)
```

Then run `run_sensitivity_analysis(...)`, `run_bayesian_calibration(...)`, or `run_abc_calibration(...)` as usual. The UQ driver runs inside the SLURM job, not on the head node.

To revert to local in-process execution for the rest of the session, set `session.uq_context.slurm_config = None` (or restart the session).

## The `ARCH=native` rebuild gotcha

PhysiCell's Makefile uses `ARCH := native`, which bakes the build host's microarchitecture into the binary. Building on a login node and then running on a different compute-node CPU family can SIGILL.

The bundled sbatch template (`scripts/physicell.sbatch`) handles this with a sentinel file:

- On the first SLURM job for a project, `make -j` runs to rebuild on the actual compute node, then `.built_on_compute` is touched.
- Subsequent runs skip rebuild as long as that sentinel exists.
- If you change compute partitions and SIGILL, delete `.built_on_compute` in the project dir to force a rebuild.

`compile_physicell_project()` (which the agent calls during normal model-building) still builds on the login node; the SLURM rebuild is purely a safety net.

## Recovering state after a server restart

The MCP server persists every submission to `~/.physicell-mcp/jobs.jsonl` (append-only). On startup, a daemon thread reads this file and reconciles SLURM-backed jobs against `squeue` / `sacct`. Local-backend jobs are skipped (their PIDs are stale).

If you restart your Claude session and `list_simulations` is missing jobs you submitted earlier, call `list_slurm_jobs()`. It queries SLURM and re-registers any in-flight jobs into `running_simulations`, after which `get_simulation_status(sim_id)` works again.

The persistence file is safe to delete if it grows unwieldy — you only lose recovery for jobs already in terminal state.

## Watching array jobs

Array submissions write per-element logs to a logs directory next to the manifest. Tail them all:

```bash
tail -f /path/to/array_dir/logs/slurm-<JOBID>_*.out
```

Watch SLURM:

```bash
squeue -u $USER                 # all your jobs
squeue -j <JOBID>               # one array (one row per element state)
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS  # post-mortem
```

## Cancelling jobs

`stop_simulation(sim_id)` calls `scancel` on SLURM-backed sims and `terminate()`/`kill()` on local sims. For an entire array, look up any element's handle (e.g. `slurm:12345_3`) and call `stop_simulation` — `scancel` accepts the array job ID prefix and cancels the whole group.

## Local-only operation (laptops, single hosts)

If you don't have SLURM, **everything still works.** The local backend is the default; none of the SLURM tools are imported until first use. Specifically:

- `run_simulation`, `get_simulation_status`, `list_simulations`, `stop_simulation` all use `LocalBackend` by default — same behavior as before SLURM support landed.
- `submit_simulation_slurm`, `submit_simulation_array_slurm`, `list_slurm_jobs`, and `configure_uq_slurm` return a clear error message if `sbatch` is not on PATH instead of crashing.
- `~/.physicell-mcp/jobs.jsonl` is still written on local submissions, but reconciliation is a no-op (PIDs are stale across restarts).
- The skill (`physicell-simulation/SKILL.md`) directs the agent to local tools whenever the workload doesn't warrant a cluster.

There is no requirement to install or configure SLURM if you only run on a single machine. The split-backend design is intentional: the same skill, the same prompts, the same tool calls work everywhere, and the right execution path is chosen based on the host environment and workload size.

## Common pitfalls

1. **Submitting from the head node without resources.** `submit_simulation_slurm` defaults are conservative; for big sweeps, override `time_limit` upward. A job killed at walltime mid-write can corrupt the UQ database.
2. **Forgetting `configure_uq_slurm` before a calibration.** The local UQ driver will silently saturate the head node's CPUs. The skill (`§1.5`) instructs the agent to set this; if you're driving the MCP directly, remember.
3. **Array `array_concurrent` set too high for ABC.** The pyABC SMC sampler segfaults under filesystem contention beyond 16–32 concurrent processes. Default of 32 is a tested ceiling.
4. **`PHYSICELL_BACKEND=slurm` + missing `sbatch`.** `get_backend("slurm")` raises clearly; the affected tool call surfaces the error. Either set `local` or run on a SLURM-aware host.
5. **Mixed local + SLURM tracking.** Both end up in `running_simulations`; `list_simulations` shows everything. `list_slurm_jobs` filters to SLURM only, which is usually what you want when reasoning about cluster fairshare.

## Files involved

- `mcp-biomodelling-servers/PhysiCell/backends.py` — `SimBackend` protocol, `LocalBackend`, `SlurmBackend`, persistence helpers.
- `mcp-biomodelling-servers/PhysiCell/scripts/physicell.sbatch` — the universal launcher script invoked by `SlurmBackend`.
- `mcp-biomodelling-servers/PhysiCell/server.py` — `run_simulation`, `submit_simulation_slurm`, `submit_simulation_array_slurm`, `list_slurm_jobs`, `configure_uq_slurm`, and the UQ dispatch wrapper `_dispatch_uq_to_slurm`.
- `mcp-biomodelling-servers/PhysiCell/session_manager.py` — `UQContext.slurm_config` field.
- `~/.physicell-mcp/jobs.jsonl` — persistent submission log (override location with `PHYSICELL_MCP_HOME`).
