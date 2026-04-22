#!/usr/bin/env python3
"""Autoregressive calibration driver for the spatial-recapitulation workflow.

Fires one `claude -p` invocation per calibration iteration, each starting
with a fresh context consisting of:
  1. The blind-constraint block (verbatim, first)
  2. The manifest and panel spec (stable across iterations)
  3. A compressed digest of iterations older than the recent window
  4. The last N iterations' records and delta reports (recent window)
  5. The current registry snapshot

State is exchanged between iterations via files in the model artifact
directory. The driver validates each iteration's iter_record.json against
the schema, audits the constraints_acknowledged block, updates the digest
incrementally as records fall off the window, and rebuilds the digest from
a sampled archive every --digest-refresh-every iterations to correct drift.

Stops when any of:
  - Every metric in both tiers passes its tolerance
  - --max-iters is reached
  - --stall-patience iterations show no improvement and no registry changes

Phase 0–3 of the workflow (manifest, panel, target fingerprint, initial
registry) must already be complete; the driver will refuse to start otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schemas" / "iter_record.schema.json"
HOOK_PATH = HERE / "hooks" / "forbid_target_source_paths.py"
ITERATION_PROMPT_TEMPLATE = HERE / "prompts" / "iteration_prompt_template.md"
DIGEST_UPDATE_TEMPLATE = HERE / "prompts" / "digest_update_prompt_template.md"
DIGEST_REBUILD_TEMPLATE = HERE / "prompts" / "digest_rebuild_prompt_template.md"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class DriverConfig:
    model_dir: Path
    max_iters: int
    tier1_tolerance: float
    tier2_tolerance: float
    per_metric_tolerance: dict[str, float]
    window_size: int
    digest_refresh_every: int
    stall_patience: int
    claude_bin: str
    calibration_model: str | None  # passed to claude -p via --model if set
    dry_run: bool


@dataclass
class StopReason:
    kind: str           # "converged" | "max_iters" | "stalled" | "error"
    detail: str


# --------------------------------------------------------------------------- #
# Model-dir layout helpers
# --------------------------------------------------------------------------- #


def manifest_path(model_dir: Path) -> Path:
    return model_dir / "manifest.json"


def digest_path(model_dir: Path) -> Path:
    return model_dir / "digest.md"


def driver_log_path(model_dir: Path) -> Path:
    return model_dir / "driver.log"


def registry_log_path(model_dir: Path) -> Path:
    return model_dir / "registry.jsonl"


def iter_dir(model_dir: Path, n: int) -> Path:
    return model_dir / f"iter_{n}"


def iter_record_file(model_dir: Path, n: int) -> Path:
    return iter_dir(model_dir, n) / "iter_record.json"


def panel_spec_path(model_dir: Path) -> Path:
    """Return the highest-versioned panel_spec_v*.json file."""
    candidates = sorted(
        model_dir.glob("panel_spec_v*.json"),
        key=lambda p: _parse_panel_version(p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No panel_spec_v*.json found in {model_dir}")
    return candidates[-1]


def _parse_panel_version(name: str) -> int:
    m = re.search(r"panel_spec_v(\d+)\.json", name)
    return int(m.group(1)) if m else -1


def latest_completed_iter(model_dir: Path) -> int:
    """Highest iter_<n>/ with a valid iter_record.json; -1 if only iter_0 (setup)."""
    best = -1
    for p in model_dir.glob("iter_*"):
        if not p.is_dir():
            continue
        m = re.match(r"iter_(\d+)", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if iter_record_file(model_dir, n).exists():
            best = max(best, n)
    return best


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def log(model_dir: Path, msg: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(driver_log_path(model_dir), "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# Manifest and forbidden-paths derivation
# --------------------------------------------------------------------------- #


def load_manifest(model_dir: Path) -> dict:
    path = manifest_path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Phase 0 (manifest) must complete before "
            "the autoregressive driver can run."
        )
    with open(path) as fh:
        manifest = json.load(fh)
    out_folder = manifest.get("target_output_folder")
    if not out_folder:
        raise ValueError(f"manifest.json missing required 'target_output_folder'")
    manifest["target_output_folder"] = str(Path(out_folder).resolve())
    if "target_project_root" in manifest and manifest["target_project_root"]:
        manifest["target_project_root"] = str(
            Path(manifest["target_project_root"]).resolve()
        )
    else:
        manifest["target_project_root"] = str(
            Path(manifest["target_output_folder"]).parent
        )
    return manifest


def forbidden_paths(manifest: dict) -> list[str]:
    root = manifest["target_project_root"]
    out = manifest["target_output_folder"]
    return [
        f"{root}/**  (everything under the target project root)",
        f"  EXCEPT {out}/**  (the output folder — this is the ONLY permitted read location under the target project)",
        f"{root}/*.xml (settings.xml, PhysiKit.xml, etc.)",
        f"{root}/config/**",
        f"{root}/custom_modules/**",
        f"{root}/*.cpp, {root}/*.h, {root}/Makefile",
        f"{root}/**/rules.csv, {root}/**/cells.csv (outside the output folder)",
    ]


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def load_schema() -> dict:
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


def validate_iter_record(record: dict, schema: dict) -> list[str]:
    """Return a list of error strings; empty list means valid."""
    if _HAS_JSONSCHEMA:
        validator = jsonschema.Draft202012Validator(schema)
        return [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(record)]
    # Minimal hand-rolled fallback covering the required fields we rely on.
    errs: list[str] = []
    for key in schema.get("required", []):
        if key not in record:
            errs.append(f"missing required field: {key}")
    ca = record.get("constraints_acknowledged") or {}
    for key in (
        "blind_constraint_version",
        "rule",
        "allowed_target_paths",
        "forbidden_target_paths",
        "files_read_this_iter",
        "self_attestation",
    ):
        if key not in ca:
            errs.append(f"constraints_acknowledged missing: {key}")
    if ca.get("self_attestation") != (
        "I did not read any file in forbidden_target_paths this iteration."
    ):
        errs.append("self_attestation text does not match required verbatim string")
    return errs


def audit_constraints(record: dict, manifest: dict) -> list[str]:
    """Cross-check files_read_this_iter against the forbidden path roots."""
    errs: list[str] = []
    ca = record.get("constraints_acknowledged") or {}
    files = ca.get("files_read_this_iter") or []
    project_root = manifest["target_project_root"]
    out_folder = manifest["target_output_folder"]
    for raw in files:
        try:
            candidate = os.path.realpath(os.path.expanduser(str(raw)))
        except Exception:
            continue
        if _is_under(candidate, out_folder):
            continue
        if _is_under(candidate, project_root):
            errs.append(
                f"files_read_this_iter lists '{raw}' (resolves to "
                f"'{candidate}') which is inside the target project root "
                f"but NOT under the permitted output folder."
            )
    return errs


def _is_under(candidate: str, root: str) -> bool:
    try:
        Path(candidate).relative_to(root)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _render(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def assemble_iteration_prompt(
    cfg: DriverConfig, manifest: dict, iter_n: int
) -> str:
    model_dir = cfg.model_dir
    window_start = max(0, iter_n - cfg.window_size)
    window_end = iter_n - 1
    digest_end = window_start - 1  # digest covers [0, window_start - 1]

    window_blocks: list[str] = []
    for n in range(window_start, window_end + 1):
        rec = _read(iter_record_file(model_dir, n))
        delta_report = _read(iter_dir(model_dir, n) / f"delta_report_iter_{n}.md")
        window_blocks.append(
            f"### Iteration {n} — iter_record.json\n\n```json\n{rec}\n```\n\n"
            f"### Iteration {n} — delta_report_iter_{n}.md\n\n{delta_report}\n"
        )
    window_content = "\n".join(window_blocks) if window_blocks else "(no prior iterations in window)"

    digest_content = _read(digest_path(model_dir)) or "(digest empty — no iterations have fallen off the window yet)"

    previous_registry = iter_dir(model_dir, max(0, iter_n - 1)) / f"registry_iter_{max(0, iter_n - 1)}.json"
    previous_delta_csv = iter_dir(model_dir, max(0, iter_n - 1)) / f"delta_iter_{max(0, iter_n - 1)}.csv"

    panel_spec = panel_spec_path(model_dir)
    target_fp = model_dir / "target" / "target_fingerprint.csv"

    template = ITERATION_PROMPT_TEMPLATE.read_text()
    values = {
        "ITER": str(iter_n),
        "ALLOWED_TARGET_PATHS": manifest["target_output_folder"],
        "FORBIDDEN_TARGET_PATHS": "\n    ".join(forbidden_paths(manifest)),
        "MANIFEST_PATH": str(manifest_path(model_dir)),
        "MANIFEST_JSON": json.dumps(manifest, indent=2),
        "PANEL_SPEC_PATH": str(panel_spec),
        "TARGET_FINGERPRINT_PATH": str(target_fp),
        "ITER_RECORD_SCHEMA_PATH": str(SCHEMA_PATH),
        "DIGEST_END_ITER": str(digest_end) if digest_end >= 0 else "(none yet)",
        "DIGEST_CONTENT": digest_content,
        "WINDOW_START": str(window_start),
        "WINDOW_END": str(window_end) if window_end >= 0 else "(none)",
        "WINDOW_SIZE": str(cfg.window_size),
        "WINDOW_CONTENT": window_content,
        "CURRENT_REGISTRY_PATH": str(previous_registry),
        "PREVIOUS_DELTA_CSV_PATH": str(previous_delta_csv),
        "ITER_DIR": str(iter_dir(model_dir, iter_n)),
    }
    return _render(template, values)


def assemble_digest_update_prompt(
    cfg: DriverConfig, falling_off_iter: int
) -> str:
    model_dir = cfg.model_dir
    prev_digest = _read(digest_path(model_dir))
    falling_record = _read(iter_record_file(model_dir, falling_off_iter))
    falling_delta = _read(
        iter_dir(model_dir, falling_off_iter)
        / f"delta_report_iter_{falling_off_iter}.md"
    )
    template = DIGEST_UPDATE_TEMPLATE.read_text()
    values = {
        "FALLING_OFF_ITER": str(falling_off_iter),
        "PREVIOUS_DIGEST": prev_digest or "(empty)",
        "FALLING_OFF_ITER_RECORD": falling_record or "(missing)",
        "FALLING_OFF_DELTA_REPORT": falling_delta or "(missing)",
    }
    return _render(template, values)


def assemble_digest_rebuild_prompt(
    cfg: DriverConfig, next_iter: int
) -> str:
    """Sample the archive (every iteration outside the current window) and
    construct the rebuild prompt."""
    model_dir = cfg.model_dir
    window_start = max(0, next_iter - cfg.window_size)
    archived = [n for n in range(0, window_start)]
    excluded = [n for n in range(window_start, next_iter)]

    # Keep all archive records inline if under ~50; else sample every Kth.
    sample = archived if len(archived) <= 50 else archived[:: max(1, len(archived) // 50)]
    blocks: list[str] = []
    for n in sample:
        rec = _read(iter_record_file(model_dir, n))
        if rec:
            blocks.append(
                f"### Iteration {n} — iter_record.json\n\n```json\n{rec}\n```"
            )
    template = DIGEST_REBUILD_TEMPLATE.read_text()
    values = {
        "ARCHIVE_SAMPLE": "\n\n".join(blocks) if blocks else "(empty archive)",
        "WINDOW_ITERS_EXCLUDED": ", ".join(str(i) for i in excluded) or "(none)",
    }
    return _render(template, values)


# --------------------------------------------------------------------------- #
# Claude invocation
# --------------------------------------------------------------------------- #


def _build_claude_cmd(cfg: DriverConfig, settings_path: Path) -> list[str]:
    cmd = [cfg.claude_bin, "-p", "--settings", str(settings_path)]
    if cfg.calibration_model:
        cmd.extend(["--model", cfg.calibration_model])
    return cmd


def _write_session_settings(model_dir: Path) -> Path:
    """Write a per-session .claude/settings.json that registers the hook."""
    settings_dir = model_dir / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / "settings.json"
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Read|Edit|Write|Grep|Glob|Bash|NotebookEdit",
                    "hooks": [
                        {
                            "type": "command",
                            "command": str(HOOK_PATH),
                        }
                    ],
                }
            ]
        }
    }
    settings_file.write_text(json.dumps(settings, indent=2))
    return settings_file


def run_claude(
    cfg: DriverConfig,
    prompt: str,
    manifest_abs_path: Path,
    cwd: Path,
    purpose: str,
) -> tuple[int, str, str]:
    """Fire a claude -p invocation. Returns (returncode, stdout, stderr)."""
    settings_path = _write_session_settings(cfg.model_dir)
    cmd = _build_claude_cmd(cfg, settings_path)
    env = os.environ.copy()
    env["CALIBRATION_MANIFEST"] = str(manifest_abs_path)

    if cfg.dry_run:
        preview = prompt[:400].replace("\n", " ")
        log(cfg.model_dir, f"[dry-run] skipping claude -p ({purpose}); prompt preview: {preview!r}…")
        return 0, "", ""

    log(cfg.model_dir, f"spawning claude -p ({purpose}): {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            env=env,
            timeout=None,
        )
    except FileNotFoundError:
        return 127, "", f"claude binary not found at '{cfg.claude_bin}'"
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------- #
# Stopping criteria
# --------------------------------------------------------------------------- #


def _effective_tolerance(
    cfg: DriverConfig, tier: str, metric: str, default_in_record: float
) -> float:
    if metric in cfg.per_metric_tolerance:
        return cfg.per_metric_tolerance[metric]
    return cfg.tier1_tolerance if tier == "tier1" else cfg.tier2_tolerance


def check_converged(cfg: DriverConfig, record: dict) -> bool:
    """Strict convergence: every listed failure must be within driver
    tolerances (in case the agent used looser tolerances internally)."""
    for tier in ("tier1", "tier2"):
        for fail in (record.get(tier, {}).get("failures") or []):
            metric = fail.get("metric", "")
            err = abs(float(fail.get("err", float("inf"))))
            tol = _effective_tolerance(cfg, tier, metric, float(fail.get("tolerance", 0)))
            if err > tol:
                return False
    return True


def is_improvement(prev: dict | None, curr: dict) -> bool:
    """Improvement = more Tier 1 passes, or same Tier 1 passes + more Tier 2 passes."""
    if prev is None:
        return True
    p1 = prev.get("tier1", {}).get("pass_count", 0)
    c1 = curr.get("tier1", {}).get("pass_count", 0)
    if c1 > p1:
        return True
    if c1 < p1:
        return False
    return curr.get("tier2", {}).get("pass_count", 0) > prev.get("tier2", {}).get("pass_count", 0)


# --------------------------------------------------------------------------- #
# Main driver loop
# --------------------------------------------------------------------------- #


def run_iteration(cfg: DriverConfig, manifest: dict, iter_n: int, schema: dict) -> dict:
    iter_dir(cfg.model_dir, iter_n).mkdir(parents=True, exist_ok=True)
    prompt = assemble_iteration_prompt(cfg, manifest, iter_n)
    prompt_dump = iter_dir(cfg.model_dir, iter_n) / "prompt.md"
    prompt_dump.write_text(prompt)

    rc, stdout, stderr = run_claude(
        cfg, prompt, manifest_path(cfg.model_dir).resolve(),
        cwd=cfg.model_dir, purpose=f"iteration {iter_n}",
    )
    (iter_dir(cfg.model_dir, iter_n) / "agent_stdout.log").write_text(stdout or "")
    (iter_dir(cfg.model_dir, iter_n) / "agent_stderr.log").write_text(stderr or "")

    if cfg.dry_run:
        # In dry-run, no agent ran → no iter_record to validate.
        # Return a minimal synthetic record so the stopping-criteria logic
        # short-circuits cleanly at the top of the next loop tick.
        return {
            "iter": iter_n,
            "tier1": {"pass_count": 0, "fail_count": 0, "failures": []},
            "tier2": {"pass_count": 0, "fail_count": 0, "failures": []},
            "registry_changes": [],
            "flags": ["dry_run"],
        }

    if rc != 0:
        raise RuntimeError(
            f"claude -p exited with code {rc} on iter {iter_n}. "
            f"stderr: {stderr[:500]}"
        )

    rec_file = iter_record_file(cfg.model_dir, iter_n)
    if not rec_file.exists():
        raise RuntimeError(f"iter {iter_n}: agent did not write {rec_file}")
    with open(rec_file) as fh:
        record = json.load(fh)

    errs = validate_iter_record(record, schema)
    if errs:
        raise RuntimeError(
            f"iter {iter_n} iter_record.json failed schema validation:\n  "
            + "\n  ".join(errs)
        )
    errs = audit_constraints(record, manifest)
    if errs:
        raise RuntimeError(
            f"iter {iter_n} constraints audit failed:\n  " + "\n  ".join(errs)
        )
    return record


def maybe_update_digest(cfg: DriverConfig, manifest: dict, next_iter: int) -> None:
    """Fold the iteration that just fell off the window into the digest.

    At next_iter=M, window is [M-N, M-1]. The record that just fell off is
    iter (M - N - 1) — i.e. it was in the window when iter M-1 started but is
    not in the window when iter M starts. Skip when nothing has fallen off
    yet (M < N + 1) since the digest has no responsibility for in-window
    iterations.
    """
    falling = next_iter - cfg.window_size - 1
    if falling < 0:
        return
    if not iter_record_file(cfg.model_dir, falling).exists():
        return
    prompt = assemble_digest_update_prompt(cfg, falling)
    (cfg.model_dir / f".digest_update_prompt_iter_{falling}.md").write_text(prompt)
    rc, stdout, stderr = run_claude(
        cfg, prompt, manifest_path(cfg.model_dir).resolve(),
        cwd=cfg.model_dir, purpose=f"digest update (fold iter {falling})",
    )
    if rc == 0 and stdout.strip():
        digest_path(cfg.model_dir).write_text(stdout)
        log(cfg.model_dir, f"digest updated (folded iter {falling})")
    else:
        log(cfg.model_dir, f"digest update FAILED rc={rc}, keeping previous digest. stderr: {stderr[:200]}")


def maybe_rebuild_digest(cfg: DriverConfig, manifest: dict, next_iter: int) -> None:
    """Full digest rebuild from archive every cfg.digest_refresh_every iters."""
    if cfg.digest_refresh_every <= 0:
        return
    if next_iter == 0 or next_iter % cfg.digest_refresh_every != 0:
        return
    prompt = assemble_digest_rebuild_prompt(cfg, next_iter)
    (cfg.model_dir / f".digest_rebuild_prompt_iter_{next_iter}.md").write_text(prompt)
    rc, stdout, stderr = run_claude(
        cfg, prompt, manifest_path(cfg.model_dir).resolve(),
        cwd=cfg.model_dir, purpose=f"digest rebuild at iter {next_iter}",
    )
    if rc == 0 and stdout.strip():
        # Keep the pre-rebuild digest as a backup for diffing.
        bk = cfg.model_dir / f"digest.before_rebuild_at_iter_{next_iter}.md"
        if digest_path(cfg.model_dir).exists():
            shutil.copy2(digest_path(cfg.model_dir), bk)
        digest_path(cfg.model_dir).write_text(stdout)
        log(cfg.model_dir, f"digest rebuilt at iter {next_iter} (backup: {bk.name})")
    else:
        log(cfg.model_dir, f"digest rebuild FAILED rc={rc}, keeping previous digest. stderr: {stderr[:200]}")


def drive(cfg: DriverConfig) -> StopReason:
    manifest = load_manifest(cfg.model_dir)
    schema = load_schema()

    last_completed = latest_completed_iter(cfg.model_dir)
    # Phase 0–3 must have produced at least iter_0/registry_iter_0.json.
    if not (iter_dir(cfg.model_dir, 0) / "registry_iter_0.json").exists():
        raise RuntimeError(
            f"iter_0/registry_iter_0.json missing in {cfg.model_dir}. "
            "Phase 3 (initial parameter registry) must complete before the "
            "autoregressive driver can run."
        )

    next_iter = (last_completed + 1) if last_completed >= 0 else 1
    # If iter_0 has a registry but no iter_record (the conventional Phase 3
    # output), start at iter 1.
    if next_iter == 0:
        next_iter = 1

    log(cfg.model_dir, f"driver starting; last completed iter={last_completed}; next iter={next_iter}")
    log(cfg.model_dir, f"config: N={cfg.window_size}, digest_refresh_every={cfg.digest_refresh_every}, stall_patience={cfg.stall_patience}, max_iters={cfg.max_iters}")

    prev_record: dict | None = None
    if last_completed >= 0:
        with open(iter_record_file(cfg.model_dir, last_completed)) as fh:
            prev_record = json.load(fh)

    stall_count = 0
    while True:
        if next_iter > cfg.max_iters:
            return StopReason("max_iters", f"reached --max-iters={cfg.max_iters}")

        # Full rebuild FIRST when cadence hits, so the iteration reads a
        # freshly corrected digest.
        maybe_rebuild_digest(cfg, manifest, next_iter)
        # Then roll the most recent fallen-off record in.
        maybe_update_digest(cfg, manifest, next_iter)

        record = run_iteration(cfg, manifest, next_iter, schema)
        log(
            cfg.model_dir,
            f"iter {next_iter} done: tier1 pass={record['tier1']['pass_count']}/"
            f"fail={record['tier1']['fail_count']}, tier2 pass={record['tier2']['pass_count']}/"
            f"fail={record['tier2']['fail_count']}, flags={record.get('flags', [])}",
        )

        if cfg.dry_run:
            # Dry-run never converges on its own — it only stops at --max-iters
            # or user interrupt. Skip the improvement/stall bookkeeping too.
            prev_record = record
            next_iter += 1
            continue

        if check_converged(cfg, record):
            return StopReason(
                "converged",
                f"all tier 1 + tier 2 metrics within tolerance at iter {next_iter}",
            )

        improved = is_improvement(prev_record, record)
        registry_changes = record.get("registry_changes") or []
        if improved or registry_changes:
            stall_count = 0
        else:
            stall_count += 1
            log(cfg.model_dir, f"no improvement and no registry changes (stall {stall_count}/{cfg.stall_patience})")
            if stall_count >= cfg.stall_patience:
                return StopReason(
                    "stalled",
                    f"{cfg.stall_patience} consecutive iterations with no improvement and no registry changes",
                )

        prev_record = record
        next_iter += 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_per_metric(entries: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in entries:
        if "=" not in e:
            raise argparse.ArgumentTypeError(
                f"--per-metric-tolerance expects metric=value, got '{e}'"
            )
        k, v = e.split("=", 1)
        out[k.strip()] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autoregressive calibration driver for spatial recapitulation.",
    )
    p.add_argument(
        "--model-dir", required=True, type=Path,
        help="Path to artifacts/<session_id>/<model_id>/ (must contain manifest.json and iter_0/).",
    )
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--tier1-tolerance", type=float, default=0.10)
    p.add_argument("--tier2-tolerance", type=float, default=0.20)
    p.add_argument(
        "--per-metric-tolerance", action="append", default=[],
        help="Override tolerance for a specific metric: metric_name=value. Repeatable.",
    )
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--digest-refresh-every", type=int, default=10)
    p.add_argument("--stall-patience", type=int, default=2)
    p.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    p.add_argument(
        "--calibration-model", default=None,
        help="Model id to pass to 'claude -p --model'. Default: harness default.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Assemble prompts and write them to disk, but skip claude -p invocations.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = DriverConfig(
        model_dir=args.model_dir.resolve(),
        max_iters=args.max_iters,
        tier1_tolerance=args.tier1_tolerance,
        tier2_tolerance=args.tier2_tolerance,
        per_metric_tolerance=_parse_per_metric(args.per_metric_tolerance),
        window_size=args.window_size,
        digest_refresh_every=args.digest_refresh_every,
        stall_patience=args.stall_patience,
        claude_bin=args.claude_bin,
        calibration_model=args.calibration_model,
        dry_run=args.dry_run,
    )
    if not cfg.model_dir.is_dir():
        print(f"--model-dir does not exist or is not a directory: {cfg.model_dir}", file=sys.stderr)
        return 2

    try:
        reason = drive(cfg)
    except KeyboardInterrupt:
        log(cfg.model_dir, "driver interrupted by user")
        return 130
    except Exception as e:
        log(cfg.model_dir, f"driver aborted: {e}")
        raise

    log(cfg.model_dir, f"STOP ({reason.kind}): {reason.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
