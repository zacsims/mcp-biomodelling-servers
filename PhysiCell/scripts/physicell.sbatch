#!/bin/bash
# physicell.sbatch — universal launcher for PhysiCell jobs under SLURM.
#
# This script is invoked by the PhysiCell MCP server's SlurmBackend. SBATCH
# resource directives are passed via the sbatch command line (--partition,
# --account, --cpus-per-task, --mem, --time, --output, etc.), not embedded
# here, so the same template handles single jobs, array jobs, UQ sweeps, etc.
#
# It supports two modes, controlled by which env vars are present:
#
#   1. Single-job mode  : PHYSICELL_EXEC + PHYSICELL_CONFIG are set.
#                          Runs one PhysiCell instance.
#
#   2. Array mode       : PHYSICELL_MANIFEST is set + SLURM_ARRAY_TASK_ID
#                          is provided by SLURM. Reads the Nth line of the
#                          manifest (0-indexed) to pick the project dir,
#                          config file, and per-task env overrides.
#
# Manifest lines (tab-separated):
#     <project_dir>\t<config_path>\t<extra_env>
# Where <extra_env> is optional KEY=VAL[;KEY=VAL...]; (semicolons separate,
# avoiding conflict with SLURM's comma parsing).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${PHYSICELL_PROJECT_DIR:-}}"
EXEC="${PHYSICELL_EXEC:-./project}"
CONFIG="${PHYSICELL_CONFIG:-config/PhysiCell_settings.xml}"

# Array mode: parse the manifest line for this task.
if [[ -n "${PHYSICELL_MANIFEST:-}" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: PHYSICELL_MANIFEST set but SLURM_ARRAY_TASK_ID is empty" >&2
        exit 2
    fi
    LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
    LINE="$(sed -n "${LINE_NO}p" "$PHYSICELL_MANIFEST" || true)"
    if [[ -z "$LINE" ]]; then
        echo "ERROR: manifest $PHYSICELL_MANIFEST has no line $LINE_NO" >&2
        exit 2
    fi
    IFS=$'\t' read -r M_PROJ M_CFG M_ENV <<< "$LINE"
    PROJECT_DIR="${M_PROJ:-$PROJECT_DIR}"
    CONFIG="${M_CFG:-$CONFIG}"
    if [[ -n "${M_ENV:-}" ]]; then
        # Apply ;-separated KEY=VAL pairs
        IFS=';' read -ra _kv_pairs <<< "$M_ENV"
        for kv in "${_kv_pairs[@]}"; do
            [[ -z "$kv" ]] && continue
            export "$kv"
        done
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: PROJECT_DIR is empty (set via --export=PROJECT_DIR=... or manifest)" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: PROJECT_DIR does not exist: $PROJECT_DIR" >&2
    exit 2
fi

# Best-effort module load. Don't fail if modules aren't available (e.g. running
# inside a conda env that already provides the toolchain).
if command -v module >/dev/null 2>&1; then
    module load gcc/14.2.0 2>/dev/null || true
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

cd "$PROJECT_DIR"

# Rebuild guard: PhysiCell's Makefile uses ARCH=native, which bakes the build
# host's microarchitecture into the binary. A login-node build can SIGILL on a
# different compute-node CPU family. The .built_on_compute marker is touched
# after a successful in-job build so we only rebuild once per (host family).
if [[ ! -x "$EXEC" ]] || [[ ! -f .built_on_compute ]]; then
    echo "[$(date)] Building $EXEC in $PROJECT_DIR ..."
    make -j "$OMP_NUM_THREADS"
    touch .built_on_compute
fi

echo "[$(date)] Host=$(hostname) Job=${SLURM_JOB_ID:-local}"
echo "[$(date)] Array=${SLURM_ARRAY_TASK_ID:-n/a} OMP=${OMP_NUM_THREADS}"
echo "[$(date)] Exec=$EXEC Config=$CONFIG"

if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --cpu-bind=cores "$EXEC" "$CONFIG"
else
    "$EXEC" "$CONFIG"
fi

echo "[$(date)] Done."
