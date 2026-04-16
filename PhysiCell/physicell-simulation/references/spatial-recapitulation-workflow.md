# Spatial Recapitulation Workflow

Reverse-engineer a PhysiCell simulation from spatial statistics alone, without access to the original configuration, rules, or parameters.

## Prerequisites

- A completed PhysiCell simulation output folder (the "target")
- Both MCP servers connected: PhysiCell MCP (`mcp__PhysiCell__`) and spatialtissuepy MCP (`mcp__spatialtissuepy__`)
- **Blind constraint**: the agent sees ONLY the output folder path — no XML config, no rules CSV, no parameter files

## How to read this workflow

The workflow is framed as a systems-engineering V-model: requirements (target fingerprint) → architecture (structural choices) → implementation (parameter values) → verification (recap fingerprint vs target). Each phase produces a machine-readable artifact that is preserved across iterations so the full decision history is auditable.

Two properties matter:

- **Flexible per-simulation** — the user drives *what* is measured and *how tightly*. The workflow accommodates tumors, neural development, granulomas, and any other ABM use case because the panel of statistics is assembled from the user's biological context.
- **Systematic within a session** — once the panel is chosen, it is applied identically to the target and to every recap iteration. Scoring is a per-metric pass/fail grid, not a free-form judgment, so convergence decisions are reproducible.

---

## Preamble — Session setup (interactive)

Before any tool calls, the agent interviews the user against a fixed manifest schema and writes `manifest.json` to the model's artifact directory. No free-form manifests — runs must be both captured and standardized.

### Manifest schema

```json
{
  "model_id": "recap_tb_granuloma_2026-04-16_run1",
  "biological_context": "TB granuloma in lung tissue; macrophage-centric",
  "relationships_of_interest": ["infiltration", "boundary_formation", "co_aggregation"],
  "morphological_features_of_interest": ["clusters", "boundaries", "layered_organization"],
  "tolerances": {
    "mode": "global",
    "global_pct": 10,
    "population_pct": 20,
    "per_metric_overrides": {}
  },
  "timestep_policy": "all",
  "target_output_folder": "/abs/path/to/target/output"
}
```

Field notes:

- `relationships_of_interest` — attraction, repulsion, infiltration, boundary formation, hub cells, co-aggregation, rare-cell localization, etc. Drives which metrics go into the fingerprint panel.
- `morphological_features_of_interest` — clusters, boundaries, gradients, isolated niches, invasive fronts, layered organization. Drives whether any exploration metrics are attached (see Phase 0).
- `tolerances.mode` — `global` | `per_metric` | `agent_decides`. Default population tolerance is 2× the global spatial tolerance; override via `per_metric_overrides`.
- `timestep_policy` — `all` is the default and strongly recommended. Subsampling risks missing transient events, early dynamics, and leading/lagging indicators. Users who pass a subset take responsibility for that choice.
- `model_id` — the name of the per-model artifact directory. Use something descriptive and unique per run so multiple passes at the same biology don't collide.

If the user declines any field, the agent proposes a default and the user confirms or edits. The final manifest is written before any analysis begins.

---

## Phase 0 — Build the Spatial Fingerprint Panel

The panel is a reusable specification: the same panel is applied to the target in Phase 1 and to every recap iteration in Phase 4. Build it once, version it, serialize it.

```python
summary_create_panel(session_id=SID, panel_key="fingerprint",
                     name="SpatialFingerprint_v1")

# Add metrics block-by-block — the default skeleton below is used when the
# manifest's relationships_of_interest and morphological_features_of_interest
# don't indicate anything more specific.
```

### Default fingerprint skeleton

Oriented to quantify cell–cell relationships and their morphological footprint.

**Cell–cell relationship block** (primary; always included)

```python
for a, b in ordered_pairs(cell_types):
    summary_add_metric(SID, "fingerprint", "colocalization_quotient",
                       params={"type_a": a, "type_b": b, "radius": R_CLQ})
for a, b in unordered_pairs(cell_types):
    summary_add_metric(SID, "fingerprint", "cross_k",
                       params={"type_a": a, "type_b": b, "max_radius": R_MAX})
summary_add_metric(SID, "fingerprint", "type_assortativity",
                   params={"graph": "proximity", "radius": R_NET})
summary_add_metric(SID, "fingerprint", "attribute_mixing_matrix",
                   params={"graph": "proximity", "radius": R_NET})
```

**Clustering / dispersion block** (primary; always included)

```python
for ct in cell_types + [None]:
    summary_add_metric(SID, "fingerprint", "ripleys_h",
                       params={"cell_type": ct, "radii": RADII})
summary_add_metric(SID, "fingerprint", "nearest_neighbor_g")
summary_add_metric(SID, "fingerprint", "clark_evans_index")   # if available
```

**Morphology block** (primary; always included)

```python
summary_add_metric(SID, "fingerprint", "getis_ord_gi_star",
                   params={"radius": R_GI})
summary_add_metric(SID, "fingerprint", "spatial_density",
                   params={"radius": R_DENS})           # radial profile
for ct in cell_types:
    summary_add_metric(SID, "fingerprint", "convex_hull",
                       params={"cell_type": ct})
```

**Neighborhood-structure block** (primary; always included)

```python
summary_add_metric(SID, "fingerprint", "clustering_coefficient",
                   params={"graph": "proximity", "radius": R_NET})
summary_add_metric(SID, "fingerprint", "degree_assortativity",
                   params={"graph": "proximity", "radius": R_NET})
```

**Marker block** (conditional — only if the target has markers)

```python
for marker in markers:
    summary_add_metric(SID, "fingerprint", "morans_i",
                       params={"marker": marker, "radius": R_MARKER})
    summary_add_metric(SID, "fingerprint", "mark_correlation",
                       params={"marker": marker})
```

**Population block** (secondary-tier constraint — see Phase 4)

```python
summary_add_metric(SID, "fingerprint", "cell_counts")
summary_add_metric(SID, "fingerprint", "cell_proportions")
```

Population metrics *must* match across the trajectory, but the constraint is weaker than the spatial ones. They are scored in Tier 2 with a looser tolerance (default 2× spatial). A recap that matches spatially but drifts in population is a diagnostic signal — right geometry, wrong mechanism — **not** a passing fit.

### Parameter-selection heuristics (for radii / thresholds)

Derive from `data_get_bounds()` and typical cell diameters. Default recipe:

- `RADII = [5, 10, 20, 50, 100, 0.2 * min_bound_span]` (multi-scale for Ripley's H).
- `R_CLQ ≈ 2 × mean_cell_radius` (short-range type association).
- `R_DENS = R_NET = 0.1 × min_bound_span` (neighborhood size).
- `R_GI = R_DENS` (hotspot radius consistent with density).
- `R_MAX = 0.3 × min_bound_span` (cross-K upper limit).
- `R_MARKER = R_NET`.

Document the chosen values in `panel_spec_v1.json` so reruns are reproducible.

### Exploration panel (conditional, cost-gated)

Attached **only** when the manifest's `morphological_features_of_interest` names a feature a specific exploratory method uniquely addresses — keeps runs cheap.

| User named feature | Attach |
|---|---|
| niches / recurring microenvironment motifs | `lda_fit` + `lda_topic_spatial_consistency` |
| loops / holes / isolated regions / topological structure | `topology_run_mapper` with an appropriate filter (density, distance-to-type, etc.) |
| gradients / layered organization | `morans_i` on a positional coordinate, plus radial density profile |

Exploration outputs inform structural design decisions in Phase 2. They never enter the convergence grid in Phase 4.

### Persistence

```python
panel_spec = summary_to_dict(SID, "fingerprint")
# write to artifacts/<session_id>/<model_id>/panel_spec_v1.json
```

If the panel is modified mid-session (e.g., a metric proved uninformative), bump the version to `panel_spec_v2.json` and keep both. Version is recorded in every downstream artifact so deltas are always comparable.

---

## Phase 1 — Target Characterization (requirements extraction)

```python
synthetic_load_physicell_simulation(session_id=SID_TGT,
                                    output_folder=manifest.target_output_folder)
timesteps = (synthetic_list_physicell_timesteps(session_id=SID_TGT)
             if manifest.timestep_policy == "all"
             else manifest.timestep_policy)

synthetic_summarize_simulation(SID_TGT, panel_key="fingerprint",
                               timestep_indices=timesteps)
df_target = summary_multi_sample_to_dataframe(SID_TGT, "fingerprint")
# write df_target to artifacts/.../<model_id>/target/target_fingerprint.csv
```

The resulting `target_fingerprint.csv` is the requirements specification: rows = every timestep, columns = every fingerprint metric. An illustrative shape:

| Time (h) | Ripley's H(100), tumor | CLQ(tumor→immune) | Density (cells/μm²) | Mean NND (μm) | Assortativity |
| -------- | ---------------------- | ----------------- | ------------------- | ------------- | ------------- |
| 0        | 0.0                    | 1.02              | 0.0004              | 42.1          | +0.05         |
| 24       | +12.8                  | 1.18              | 0.0011              | 28.7          | +0.19         |
| 48       | +28.4                  | 1.31              | 0.0021              | 19.2          | +0.33         |
| 72       | +45.2                  | 1.45              | 0.0032              | 14.3          | +0.45         |

(Real scorecards will have many more columns — one per fingerprint metric — and a row per timestep.)

If an exploration panel is attached, its outputs go to `target/target_exploration/` and feed Phase 2 (not Phase 4).

---

## Phase 2 — Architectural Design (structural choices)

Structural decisions — which cell types are present, which substrates to include, which rule families to wire — are driven by the target's relationship inventory (from the CLQ / assortativity / mixing-matrix blocks) and morphology (from the Gi\*, density profile, and any exploration findings).

> The Translation Table below is a **hypothesis generator**, not a causal specification. Agent-based models have emergent behavior that does not always trace cleanly to one input parameter; the same spatial pattern can arise from different mechanisms. Treat each row as a **first guess** to test via recapitulation, not a guarantee.

### Translation Table

| Statistic | Pattern | What It Implies | PhysiCell Action |
|---|---|---|---|
| **Population counts** | N cells, doubling time | Proliferation rate | `set_cycle_transition_rate(rate)` where rate ≈ ln(2)/(doubling_min) |
| **Cell proportions** | Stable or shifting ratio | Relative growth/death balance | Adjust cycle and death rates between types |
| **Ripley's H > 0** | Clustering | Low dispersal, cohesive growth | Low `motility_speed` (0.1-0.5), high adhesion |
| **Ripley's H < 0** | Dispersion / regularity | Contact inhibition or repulsion | Add pressure→decreases→cycle entry rule, increase motility |
| **H peak radius** | Cluster size ≈ r μm | Spatial scale of aggregation | Match initial placement radius; check O₂ diffusion length |
| **CLQ(A,B) > 1** | A and B co-locate | Attraction between types | A secretes substrate S; B chemotaxes toward S |
| **CLQ(A,B) < 1** | A and B avoid | Repulsion or competition | Separate chemotaxis targets; resource competition |
| **CLQ(A,B) ≈ 1** | No spatial association | No direct interaction | Omit cross-type chemotaxis rules |
| **CLQ asymmetric** | CLQ(A,B) ≠ CLQ(B,A) | One type seeks the other | The type with CLQ > 1 is the "seeker" — give it chemotaxis |
| **Central hotspot** | Gi* cluster in center | Centripetal growth | O₂ Dirichlet boundaries + necrosis rule + pressure rule |
| **Peripheral hotspots** | Gi* at edges | Invasive front | High motility + outward chemotaxis |
| **Multiple hotspots** | Several Gi* clusters | Multi-focal growth | Multiple `place_initial_cells()` calls |
| **Single hotspot** | One Gi* cluster | Cohesive mass | Single `place_initial_cells(pattern="random_disc")` |
| **Mean NND ≈ 17 μm** | Cells touching | Dense packing | Default volume (~2494 μm³), high adhesion |
| **Mean NND >> 30 μm** | Sparse | High motility or low density | Increase motility; check domain size vs cell count |
| **G function steep** | Most NNDs small | Contact monolayer | Regular packing from contact inhibition |
| **Mixing diagonal-heavy** | Homophily (>70% same-type) | Segregated tissue | Strong same-type adhesion via `configure_cell_mechanics()` |
| **Mixing off-diagonal** | Well-mixed | No differential adhesion | Default mechanics, shared substrates |
| **Assortativity > 0.3** | Like-attracts-like | Type-specific domains | Differential adhesion between types |
| **Assortativity < 0** | Heterophily | Cross-type infiltration | Type B chemotaxes toward substrate from type A |
| **Assortativity ≈ 0** | Random mixing | No type-specific spatial rules | Default mechanics |
| **High clustering coeff** | Triangular clusters (>0.3) | Tight local packing | Low motility, strong adhesion |
| **Moran's I > 0** | Marker autocorrelation | Substrate gradients | Low diffusion coefficient, high uptake |

### Model Building Sequence

1. **Domain**: Set `create_simulation_domain()` to match spatial bounds from `data_get_bounds()`.
2. **Cell types**: One `add_single_cell_type()` per unique type found in the target.
3. **Substrates**: Inferred from spatial patterns — central necrotic core → oxygen with Dirichlet boundaries; type colocalization (CLQ > 1) → chemokine with secretion/chemotaxis pairing; density gradients → substrate with uptake creating local depletion.
4. **Motility / mechanics**: From Ripley's H and NND — high H (clustering) → low speed (0.1–0.5 μm/min); low H (dispersion) → high speed (1.0–5.0 μm/min).
5. **Rules**: From colocalization and mixing — CLQ > 1 between types → chemotaxis rule connecting them; central density gradient → oxygen-decreases-necrosis + pressure-decreases-cycle-entry; type segregation → differential adhesion.
6. **Initial conditions**: From hotspot pattern and early-timestep spatial extent — single compact cluster → `random_disc`; scattered → `random_rectangle`; ring → `annular`.
7. **Literature grounding**: Use `search_literature()` from the LiteratureValidation MCP to ground parameter values before adding rules.

Record every structural choice in the initial parameter registry (Phase 3) with the hypothesis that justified it — this makes the decision log auditable later.

---

## Phase 3 — Initial Parameter Registry

Before running anything, the agent writes `registry_iter_0.json` to the model directory. Each entry records the initial guess, its allowed range, its intended role, and the hypothesis that led to the value.

```json
[
  {
    "name": "tumor.motility_speed",
    "value": 0.3,
    "range": [0.05, 2.0],
    "role": "motility",
    "source": "TranslationTable:HighH→lowSpeed"
  },
  {
    "name": "immune.chemotaxis_strength",
    "value": 1.0,
    "range": [0.0, 5.0],
    "role": "chemotaxis",
    "source": "TranslationTable:CLQ(immune,tumor)>1→chemotaxis"
  }
]
```

The registry is **mutable** during calibration. The agent may add a parameter that wasn't originally registered (because a stat refuses to converge), remove one that's stuck at a boundary or produces non-biological output, or split one (e.g., apply chemotaxis only to cell type B). Registry changes are expected — calibration frequently reveals that the initial parameter set was incomplete, and that freedom is part of the design.

### Change log

A single rolling file `registry.jsonl` at the model directory root. One JSON line per change:

```json
{"iter": 3, "timestamp": "2026-04-16T14:02:11Z", "change_type": "add", "parameter": "immune.chemotaxis_strength", "old_value": null, "new_value": 2.5, "reason": "CLQ(immune,tumor) still < 1 after 2 iter; adding chemotaxis toward tumor-secreted substrate"}
{"iter": 4, "timestamp": "2026-04-16T14:18:42Z", "change_type": "modify", "parameter": "tumor.motility_speed", "old_value": 0.3, "new_value": 0.15, "reason": "Ripley's H dropped below target; reduce dispersal"}
```

`registry.jsonl` is the what/when/why history. `registry_iter_<n>.json` files are snapshots of the active parameter set at each iteration — together they form the full audit trail.

---

## Phase 4 — Simulate and Verify

Run the PhysiCell project with the current parameter registry. Then apply the **same panel** to the recap output:

```python
synthetic_load_physicell_simulation(session_id=SID_RCP,
                                    output_folder=recap_output_folder)
synthetic_summarize_simulation(SID_RCP, panel_key="fingerprint",
                               timestep_indices=same_indices_as_target)
df_recap = summary_multi_sample_to_dataframe(SID_RCP, "fingerprint")
```

### Scoring (per metric, per timestep, no weighting)

```text
For each (metric m, timestep t):
    T = target_fingerprint[t, m]
    R = recap_fingerprint[t, m]

    if m is scalar-valued (density, counts, NND, proportion):
        err[t,m] = |R − T| / (|T| + ε)               # relative error

    if m is directional (H, CLQ, assortativity, Moran's I):
        sign_match[t,m] = (sign(R) == sign(T))
        err[t,m]        = |R − T| / (|T| + ε)        # reported alongside sign

    if m is vector-valued (Ripley's H(r) curve, mixing matrix):
        err[t,m] = ||R − T||_2 / ||T||_2             # normalized L2

    if m is categorical (hotspot count, # components):
        err[t,m] = (R == T)                          # boolean
```

No weighting, no aggregated loss, no scalar objective. The scorecard is the full per-cell grid.

### Two-tier convergence grid

- **Tier 1 — spatial metrics** (cell–cell relationship, clustering, morphology, neighborhood-structure, marker blocks). *Every* `err[t,m]` must be within its metric's tolerance.
- **Tier 2 — population metrics** (counts, proportions). *Every* `err[t,m]` must be within its (looser) tolerance. Default population tolerance is 2× the global spatial tolerance unless the manifest overrides; e.g. global 10 % → spatial 10 %, population 20 %.

**Full convergence** = Tier 1 passes **AND** Tier 2 passes.

**Partial states** (reported explicitly, never accepted as a fit):

- Tier 1 passes, Tier 2 fails → *"spatially recapitulated, population drift."* Right geometry, wrong mechanism. Something in the dynamics (cycle/death rates, differential fitness across types) needs adjustment even though the spatial organization already matches.
- Tier 2 passes, Tier 1 fails → *"counts right, organization wrong."* Keep calibrating spatial-driving parameters (motility, adhesion, chemotaxis rules, initial placement).

### Iteration artifacts

Each iteration writes to its own `iter_<n>/` folder:

- `recap_fingerprint_iter_<n>.csv` — the recap scorecard.
- `delta_iter_<n>.csv` — same shape as the scorecard; entries are `err` values.
- `delta_report_iter_<n>.md` — human-readable summary highlighting the failing cells of the grid and, for each, suggesting the most plausible registry knob (informed by the Translation Table hypotheses).
- `registry_iter_<n>.json` — snapshot of the active parameter set.

---

## Phase 5 — Calibration Loop

Modular section — intended to be replaced wholesale by the forthcoming dedicated UQ MCP server, which will expose each fingerprint metric as a first-class QoI and run calibration end-to-end. Keep the interface between "fingerprint panel" and "calibration tool" clean so the swap is local to this phase.

### Short-term plan (agent-on-the-fly)

1. **Coordinate adjustments** — for each failing metric, propose the parameter most likely to move it (from the Translation Table or prior iteration deltas). Apply one adjustment at a time when possible, so the cause of each improvement is legible.
2. **Local search** — if single-parameter moves stall, run small factorial sweeps (3–5 values per parameter over 2–3 parameters) and pick the best configuration by counting how many metrics pass under each.
3. **DoE + sensitivity** — if mid-scale local search stalls, run a Latin Hypercube + per-metric Sobol indices using `uq-physicell`. Because current UQ accepts only scalar QoIs, the agent must pick specific cells of the fingerprint grid to surface as QoIs (e.g., `ripleys_h_tumor_r100_t_final`).
4. **Bayesian / ABC calibration** — only when the search space is small and the QoIs are well-chosen. Use `run_bayesian_calibration` or `run_abc_calibration`.
5. **Parameter-registry changes** — each iteration may add/remove registry entries when the active set is insufficient or when a tuned parameter produces non-biological output (Phase 3 mechanics).

### Stopping criteria

Stop when either condition holds:

- Every metric in both tiers passes its tolerance, **or**
- 2 consecutive iterations produce no improvement *and* no new parameters are being registered.

---

## Phase 6 — Reporting & Audit

Session ends with `final_report.md` in the model directory containing:

- Manifest echo (biological context, relationships of interest, morphology features of interest, tolerances, timestep policy).
- Panel spec version used (`panel_spec_v<n>.json`).
- Convergence status — the full per-metric pass/fail grid with Tier 1 and Tier 2 clearly separated.
- Final parameter registry (`registry_iter_<final>.json`).
- Parameter registry history — the whole `registry.jsonl` with diff reasons.
- Exploratory findings that influenced structural decisions.
- Population consistency check — did the population dynamics match, as expected from a good spatial fit? If not, flag that the fit is spatially good but dynamically off (the model may have the right static geometry for the wrong mechanistic reasons).

### Artifact directory structure

```text
artifacts/<session_id>/<model_id>/
    manifest.json                  # standardized schema, filled at session start
    panel_spec_v1.json
    [panel_spec_v2.json ...]
    registry.jsonl                 # single rolling change log (what/when/why)
    target/
        target_fingerprint.csv
        target_exploration/...     # present only if user named a relevant feature
    iter_0/
        registry_iter_0.json       # snapshot of active parameters at this iter
        recap_fingerprint_iter_0.csv
        delta_iter_0.csv
        delta_report_iter_0.md
    iter_1/
        ...
    final_report.md
```

Each iteration folder is self-contained — re-running later, or onboarding another agent, can begin from any checkpoint.

---

## Tips

1. **Spatial first, population second.** A good spatial match usually brings population dynamics along for the ride. When it doesn't, that's a diagnostic signal, not a passing fit.
2. **The Translation Table is a hypothesis generator.** Emergent ABM behavior doesn't always trace cleanly to one input parameter. Treat each row as a first guess to test, not a rule.
3. **Expect the parameter registry to grow.** Calibration regularly reveals that the initial parameter set is incomplete. Adding a parameter with a documented reason is better than flailing on the existing ones.
4. **Every timestep matters.** Subsampling risks missing transient events, early dynamics, and leading/lagging indicators. Use `timestep_policy="all"` unless you have a specific reason not to.
5. **Also analyze the target's first timestep** (`timestep_index=0`) — it's the strongest clue about initial conditions.
6. **Use visualizations for sanity checks**: `viz_plot_cell_types`, `viz_plot_ripleys_curve`, `viz_plot_colocalization_heatmap`, `viz_plot_hotspot_map`.
7. **Ground parameters in literature** via `search_literature()` from the LiteratureValidation MCP before adding rules.
8. **When in doubt, re-read the manifest.** If calibration feels aimless, the tolerances or the relationships-of-interest may be under-specified — revisit the preamble with the user.
