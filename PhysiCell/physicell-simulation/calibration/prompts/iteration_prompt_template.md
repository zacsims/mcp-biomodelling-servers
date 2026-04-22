# Calibration iteration {{ITER}} — autoregressive driver

## BLIND CONSTRAINT — READ FIRST, MUST RESTATE IN OUTPUT

You are calibrating a PhysiCell simulation to recapitulate a target whose
ground-truth configuration you are NOT allowed to see. You may read ONLY
files under the target's OUTPUT folder:

    {{ALLOWED_TARGET_PATHS}}

You MUST NOT read, cat, grep, glob, open, or otherwise inspect any file
under the following paths, EVEN IF ACCESSIBLE:

    {{FORBIDDEN_TARGET_PATHS}}

Forbidden files explicitly include (but are not limited to):
  - settings.xml, PhysiCell_settings.xml, PhysiKit.xml
  - rules.csv, cells.csv, *.csv outside the output folder
  - custom.cpp, main.cpp, Makefile, custom_modules/**
  - Any XML, CSV, C++, or header file that is under the target project
    root but NOT under the output folder above

A mechanical hook will reject such reads; in addition, your iter_record.json
MUST list every file you read in `constraints_acknowledged.files_read_this_iter`
and MUST include the self_attestation line verbatim:

    "I did not read any file in forbidden_target_paths this iteration."

If you need information that a forbidden file would give you, INFER it from
output files only. That is the point of the exercise.

---

## Fixed setup (do not modify)

- **Manifest** (`{{MANIFEST_PATH}}`):

```json
{{MANIFEST_JSON}}
```

- **Panel spec** (`{{PANEL_SPEC_PATH}}`): the same fingerprint panel applied
  to the target. Use it verbatim for this iteration's recap.

- **Target fingerprint** (read-only, allowed):
    `{{TARGET_FINGERPRINT_PATH}}`

- **Iteration schema** (your iter_record.json MUST validate against this):
    `{{ITER_RECORD_SCHEMA_PATH}}`

---

## Compressed history (iterations 0 .. {{DIGEST_END_ITER}})

{{DIGEST_CONTENT}}

---

## Recent window (iterations {{WINDOW_START}} .. {{WINDOW_END}})

The last {{WINDOW_SIZE}} iterations' machine-readable records and
human-readable delta reports follow. Treat them as ground truth about
what has already been tried — do not redo experiments you see here.

{{WINDOW_CONTENT}}

---

## Current state

- Active registry (snapshot from previous iteration):
    `{{CURRENT_REGISTRY_PATH}}`
- Previous iteration's delta CSV:
    `{{PREVIOUS_DELTA_CSV_PATH}}`
- Target output folder (allowed to read):
    `{{ALLOWED_TARGET_PATHS}}`
- Your working artifact directory for this iteration:
    `{{ITER_DIR}}`

---

## Your task

Run calibration iteration {{ITER}} per Phase 4 and Phase 5 of the
spatial-recapitulation workflow. Concretely:

1. Read the compressed history, recent window, and current registry.
2. Propose a parameter adjustment (or registry change) that targets a
   failing cell of the previous iteration's scorecard. Prefer a single
   coordinated adjustment per iteration so the cause of any improvement
   is legible. Ground values in the Translation Table and (if available)
   literature.
3. Apply the adjustment to the PhysiCell project via the appropriate
   `mcp__PhysiCell__` tools. Run the simulation.
4. Apply the same panel to the recap output, compute the scorecard,
   compute the delta vs target.
5. Write the standard Phase 4 artifacts to `{{ITER_DIR}}`:
     - `registry_iter_{{ITER}}.json`
     - `recap_fingerprint_iter_{{ITER}}.csv`
     - `delta_iter_{{ITER}}.csv`
     - `delta_report_iter_{{ITER}}.md`
6. Write `iter_record.json` in `{{ITER_DIR}}` validating against
   `{{ITER_RECORD_SCHEMA_PATH}}`. The `constraints_acknowledged` block
   is required and will be audited.
7. Append any registry changes to `registry.jsonl` at the model root.
8. Stop. Do not plan the next iteration — the driver handles looping.

## Convergence check (driver-side, informational here)

The driver will stop the loop when BOTH tiers pass under their tolerances
OR when `--max-iters` is reached OR when `--stall-patience` iterations
show no improvement and no registry changes. If you believe the run has
converged this iteration, set `flags: ["converged"]` in iter_record.json.
The driver will confirm deterministically from the scorecard.
