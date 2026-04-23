---
name: blind-recap
description: >
  Rebuild a PhysiCell model that reproduces a target simulation's behaviour using only
  the target's output folder — no peeking at config/, cells.csv, cell_rules.csv, or
  custom_modules/. Codifies the fingerprinter + evidence + builder + calibrator team
  pattern, the handoff artifact schema, scoring conventions, and the known infrastructure
  pitfalls. Use when the user asks to "recapitulate", "rediscover", or "blind-recover"
  a PhysiCell model from its output alone, or to benchmark the modelling pipeline
  against a held-out target.
compatibility: Requires PhysiCell MCP + spatialtissuepy MCP + LiteratureValidation MCP
metadata:
  author: simsz
  version: "1.0"
---

# Blind recapitulation skill

## Purpose

Given a PhysiCell simulation's output folder as the only source of truth about the target's dynamics, rebuild a model whose behaviour matches the target. The value of the exercise is discipline: it tests whether the modelling pipeline (fingerprint → literature → build → calibrate → score) recovers a known mechanism from data alone.

## When to use

- User references `blind-recap`, blind recapitulation, or an explicit "no peeking at the target's config" framing.
- User points to a PhysiCell user_project and asks to reproduce it without reading the canonical rules.
- Benchmark / regression runs where you want to measure how well the pipeline recovers a model from output.

## Do not use for

- Ordinary model building from a new biological scenario — use the `physicell-simulation` skill directly.
- Reproducing a published paper whose equations and parameters are in the paper — that's not blind; read the paper.

## Allowed vs forbidden inputs

| Allowed | Forbidden |
|---|---|
| Target's `output_NNN/` directories (any of them) | Target's `config/` |
| SVG snapshots, MAT snapshots, XML frames, initial.svg / final.svg | Target's `cells.csv`, `cell_rules.csv` |
| Published literature on the biological scenario | Target's `custom_modules/` source (leaks rules) |
| The target's biological scenario description (given by the user) | Target's `main.cpp` / `Makefile` / generated `project` binary |
| Derived spatial / temporal metrics on allowed data | Any file that records the ground-truth parameters |

The principle: treat the target as if it were experimental data. You can measure it but not interrogate its source.

## Team pattern

Spawn four agents on a shared team under a `hypoxia-recap` / `<topic>-recap` team name. Each agent has a scoped role and produces named artifacts in the shared workspace (default `artifacts/<team>/`).

| Agent | Role | Key outputs |
|---|---|---|
| `spatial-analysis` (fingerprinter) | Extract QoIs and a scoring pipeline | `target_fingerprint.json`, `score_spec.json`, `target_summary.md`, per-candidate `score_*.json` |
| `literature-rule-validator` (evidence) | Literature-backed rule priors with citations | `proposed_rules.json`, `literature_brief.md`, stored rule justifications |
| `model-constructor` (builder) | Compose PhysiCell model from fingerprint + priors | `candidate_v{N}.xml`, `candidate_v{N}_cell_rules.csv`, `candidate_v{N}_manifest.json`, `builder_handoff.json` |
| `parameter-calibration` (fitter) | ABC calibration + confirmation sims | `calibration_final.json`, `calibration_result_v{N}.json`, `calibrated_output_v{N}/` |

Optional fifth agent: `uq` for pre-calibration sensitivity screening when the UQ parameter set is large (>5 params with wide priors).

## Handoff artifact schemas

Minimal viable schemas each agent should write for the next agent downstream.

### `target_fingerprint.json` (fingerprinter → builder, fitter)

```json
{
  "domain_bbox": [[xmin, ymin, zmin], [xmax, ymax, zmax]],
  "cell_types": ["tumor", "motile tumor"],
  "timesteps": [0, 60, 120, ...],
  "trajectories": { "tumor": [...], "motile tumor": [...] },
  "spatial_metrics": {
    "1200": { "ripley_h": {...}, "voronoi_ratio": ..., "assortativity": ..., "hull_area_ratio": ... }
  },
  "qois": [
    { "name": "motile_frac_at_t7200", "value": 0.113, "weight": 1.2, "note": "quasi-steady fraction" },
    ...
  ]
}
```

6–10 QoIs is a good target. Mix scalar terminal values (easy to hit) with ratio / shape signatures (oscillation peak/trough, hull expansion ratio) to avoid over-fitting to endpoint counts.

### `score_spec.json` (fingerprinter → everyone)

```json
{
  "metrics": ["weighted_nrmse_over_qois"],
  "bands": { "excellent": 0.15, "good": 0.30, "acceptable": 0.50, "poor": 1.0 },
  "penalty": "per_qoi = min(1, |cand - target| / |target|)",
  "aggregate": "sum(weight * penalty) / sum(weight)"
}
```

Clamped penalty is important: without it, a single catastrophic QoI dominates the aggregate and masks progress elsewhere.

### `proposed_rules.json` (evidence → builder)

List of rule dicts; each includes cell_type, signal, direction, behavior, half_max, hill_power, base_value (i.e., XML default), saturation_value, confidence label, justification paragraph, and citation list. See `physicell-simulation/references/rules-and-hill-functions.md` for rule semantics — the base_value and saturation_value fields must be set with knowledge of which end is the "firing end".

### `builder_handoff.json` (builder → fitter)

```json
{
  "session_name": "...",
  "xml_path": "artifacts/<team>/candidate_v1.xml",
  "uncertain_parameters": [
    { "path": "...", "prior_range": [lo, hi], "prior_median": ..., "unit": "...", "prior_type": "uniform|log-uniform", "reason": "..." }
  ],
  "fixed_parameters": [...],
  "notes": "..."
}
```

`prior_type` matters — log-uniform is correct for parameters spanning >1 decade (uptake rates, switch rates). Uniform wastes particles at the low end.

### `calibration_final.json` (fitter → builder, fingerprinter, team-lead)

Must include: `method`, `map_parameters`, `posterior_summary` (median, p05, p95, map per param), `candidate_output_path`, `method_notes` (prior widths, replicate count, number of rounds, distance function). Flag any params whose posterior hugs a prior bound — those are identifiability failures, not fits.

## Workflow

1. **Team setup.** `TeamCreate` with a clear team name. Spawn all four agents in parallel with the team_name parameter so they share the task list. Create top-level tasks (fingerprint, build, calibrate, score, report) and wire dependencies via `addBlockedBy`.

2. **Fingerprint first, everything else blocked.** The fingerprinter's output unlocks everyone else. While it runs, the evidence agent can start its literature searches (not strictly dependent on the fingerprint).

3. **Build from fingerprint + priors.** The builder waits for both `target_fingerprint.json` and `proposed_rules.json`. It composes a candidate, exports v1, and writes `builder_handoff.json` with the UQ parameter list (ranked by prior width × expected leverage).

4. **Bracket the target before full ABC.** Run a baseline (all priors at median) and at least one "extreme" sim (highest-leverage param at prior boundary). Confirm the resulting scores bracket the target. If they don't, the prior or the model structure is wrong — fix before spending calibration budget.

5. **Truncate for calibration.** Re-extract target QoIs at an earlier horizon (e.g., `target_fingerprint_t2000.json`) and score at the truncated horizon. This cuts per-particle sim time and still discriminates the dominant QoIs.

6. **Two-stage ABC.** Stage 1: 50 Latin-hypercube particles at truncated horizon, 8 parallel workers, rejection-ABC (not pyABC SMC — see pitfalls). Stage 2: top-20 by stage-1 distance re-run at full horizon with joint distance. Best joint particle = MAP.

7. **Apply + re-export.** Builder bakes MAP into `candidate_vN.xml`. Fingerprinter scores at both horizons. If acceptable (< 0.50 per score_spec bands), ship. Otherwise iterate: widen priors, add rules (e.g., hypoxic-apoptosis to contain runaway proliferation), or add spatial QoIs to the distance.

8. **Final report.** Team-lead writes `REPORT.md` pulling: scoring matrix, MAP parameters, mechanistic milestones, residual misses, and deliverable inventory.

## Known pitfalls (read before starting)

1. **`uq_physicell.pc_model._set_xml_element_value` XPath-setter bug.** Terminal `[@name='X']` selectors corrupt the `name` attribute instead of writing to element text. Silently breaks any calibration that substitutes a transformation_rate or similar attribute-named element. Dry-run one substitution and inspect the XML diff before launching.

2. **pyABC ABC-SMC multicore sampler is unreliable here.** Segfault cascade >16 processes (filesystem contention); replicate-retry loop that never accepts particles for heterogeneous runtimes. Use simple rejection ABC unless you have a specific reason not to.

3. **PhysiCell rule direction semantics.** `decreases` ≠ "max at low signal"; see `physicell-simulation/references/rules-and-hill-functions.md`. The firing rate lives at the signal end where the rule is active — for hypoxia-induced rules that's the **XML default**, with `saturation_value = 0`. Getting this backwards produces a rule firing at normoxia instead of hypoxia and looks like a calibration failure.

4. **literature-rule-validator runs 30–90 min per audit.** Do not treat silence in that window as a stall; do not fall back to team-lead-authored priors before the validator has delivered. See `feedback_literature_validator_latency.md`.

5. **Teammate MCP-tool-schema stall at spawn.** MCP-heavy teammates can go silent for up to ~1 h loading deferred tool schemas on their first turn. If a newly-spawned teammate is silent past ~5 min with `in_progress` status and no artifact writes and the agent has never produced output before, consider asking for a "tools loaded" ack.

6. **Substrate fields may not be persisted in output.** If `has_substrate=false` in the target's MultiCellDS export, you have position-based QoIs only — no O₂ field to read directly. Infer the substrate operating point from spatial cell-type patterns (motile cells at rim → low O₂ at rim).

7. **Output folder may be named `output_001/` not `output/`.** Don't assume. List the target's directory structure before loading.

## Acceptance criteria

Score below 0.50 aggregate at full horizon is the minimum threshold to declare the recap "demonstrated." A single-QoI-within-1% headline match (e.g., terminal quasi-steady phenotype fraction) plus qualitative mechanism recovery (oscillation shape, spatial segregation, hypoxic dip) is stronger evidence than a lower aggregate alone — an aggregate hides whether the mechanism is right or a few lucky QoIs pulled the score down. Both should be reported.

## References

- `physicell-simulation/references/rules-and-hill-functions.md` — PhysiCell rule semantics, UQ knob placement.
- Agent definitions under `~/.claude/agents/` — `{spatial-analysis, literature-rule-validator, model-constructor, parameter-calibration, uq}.md`.
- Project memories under `~/.claude/projects/.../memory/` — team coordination, literature-validator latency, uq_physicell XPath bug.
