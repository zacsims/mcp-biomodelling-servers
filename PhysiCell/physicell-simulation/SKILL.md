---
name: physicell-simulation
description: >
  Build, configure, and run PhysiCell multicellular simulations using the PhysiCell MCP server.
  Use when the user asks to model tumors, cells, tissues, or multicellular biological systems.
  Covers simulation setup, cell rules (Hill functions), initial conditions, PhysiBoSS boolean
  networks, UQ/calibration, and literature validation. Prevents common configuration mistakes
  like the "from 0 towards 0" rule bug.
compatibility: Requires PhysiCell MCP server connection
metadata:
  author: simsz
  version: "2.0"
---

# PhysiCell Simulation Skill

## Non-Negotiable Rules

These rules are enforced by server-side validation. Violating them will cause tool calls to fail.

1. **Never edit generated files.** Do NOT use Edit, Write, or Bash to modify XML configs, `cell_rules.csv`, or `cells.csv`. These are managed by the MCP server. Direct edits will be overwritten and desync session state. Fix problems by calling the appropriate MCP tool, then re-export.

2. **All MCP tools must be called via MCP tool calling** (prefix `mcp__PhysiCell__`). Do NOT run them via Bash, subprocess, or npx.

3. **Never extract parameter values from raw sources.** Do NOT read papers, abstracts, or web results and manually pick out numbers for half_max, hill_power, rates, etc. Use literature search tools or the LiteratureValidation MCP (if available) for evidence-based parameter guidance.

4. **Literature validation is optional.** If the LiteratureValidation MCP is available, use it to validate rules against published literature. If it is not available, proceed with export directly after adding rules. The validation tools (`get_rules_for_validation`, `store_validation_results`, `get_validation_report`) remain available for when validation is desired.

## 1. Tool Ordering

Follow this sequence. Do not skip steps.

```
create_session
  → analyze_biological_scenario
  → create_simulation_domain
  → add_single_substrate              (repeat per substrate)
  → add_single_cell_type              (repeat per cell type)
  → configure_cell_parameters         (repeat per cell type)
  → set_substrate_interaction         (repeat per cell×substrate pair)
  → [setter tools as needed]          (see Section 2 — set nonzero defaults BEFORE rules)
  → add_single_cell_rule              (repeat per rule — see Section 2)
  → place_initial_cells               (repeat per placement)
  → [LITERATURE VALIDATION]           (optional — see Section 7)
  → export_xml_configuration
  → export_cell_rules_csv
  → export_cells_csv                  (if cells were placed)
  → create_physicell_project
  → compile_physicell_project
  → run_simulation
  → get_simulation_status             (poll every 60s until done)
  → get_simulation_analysis_overview   (first look at results)
  → get_population_timeseries          (growth curves)
  → get_timestep_summary              (drill into specific timepoints)
  → get_substrate_summary             (substrate concentrations)
  → get_cell_data                     (detailed cell attributes)
  → generate_analysis_plot            (save visualizations)
  → generate_simulation_gif           (spatial animation)
```

## 2. Hill Function Rules

### How rules work

`add_single_cell_rule` writes a CSV row that PhysiCell reads to compute behavior via a Hill function:

```
                                      signal^n
behavior = base_value + (saturation - base_value) × ─────────────────
                                      half_max^n + signal^n
```

- **base_value** = `min_signal` parameter (default: 0). Behavior value when signal is absent.
- **saturation** = XML default rate for that behavior. Comes from cell type config, NOT from the rule tool.
- **half_max** = signal level at 50% effect.
- **hill_power** (n) = steepness: 1=gradual, 4=moderate, 8=switch-like.

`direction = "increases"`: base_value (low signal) → saturation (high signal).
`direction = "decreases"`: saturation (low signal) → base_value (high signal).

### The "from 0 towards 0" bug

If base_value = 0 AND the XML default is also 0:

```
behavior = 0 + (0 - 0) × Hill(signal) = 0    ← ALWAYS ZERO, rule does nothing
```

No error is raised. The simulation runs but the rule has no effect.

**Prevention:** Before adding any rule, ensure the target behavior has a nonzero XML default. Use the appropriate setter tool:

| Behavior | Default | Setter tool |
|---|---|---|
| cycle entry | Depends on cycle model | Usually safe as-is |
| apoptosis / necrosis | 0 unless set | `configure_cell_parameters` (`apoptosis_rate`, `necrosis_rate`) |
| migration speed / persistence | 0 unless set | `configure_cell_parameters` (`motility_speed`, `persistence_time`) |
| X secretion / X uptake | 0 | `set_substrate_interaction()` |
| transition to X | 0 | `set_cell_transformation_rate()` |
| attack X / phagocytose X / fuse to X | 0 | `set_cell_interaction()` |
| phagocytose dead cells | 0 | `configure_cell_interactions()` |
| attachment / detachment rate | 0 | `configure_cell_mechanics()` |
| damage / damage repair rate | 0 | `configure_cell_integrity()` |
| exit from cycle phase N / custom | 0 | No setter — use nonzero `min_signal` |

The server validates rules and rejects "from 0 towards 0" with guidance on which setter to use.

### Setter examples

```python
# Death rates, motility:
configure_cell_parameters(cell_type="tumor", apoptosis_rate=0.001, necrosis_rate=0.00277, motility_speed=0.5)

# Secretion/uptake:
set_substrate_interaction(cell_type="tumor", substrate="chemokine", secretion_rate=0.1)

# Cell transformation:
set_cell_transformation_rate(cell_type="tumor", target_cell_type="motile_tumor", rate=0.001)

# Per-target interactions:
set_cell_interaction(cell_type="macrophage", target_cell_type="tumor", interaction_type="attack", rate=0.1)

# Dead-cell phagocytosis:
configure_cell_interactions(cell_type="macrophage", apoptotic_phagocytosis_rate=0.01, necrotic_phagocytosis_rate=0.01)

# Mechanics:
configure_cell_mechanics(cell_type="tumor", attachment_rate=0.01, detachment_rate=0.001)

# Integrity:
configure_cell_integrity(cell_type="tumor", damage_rate=0.001, damage_repair_rate=0.0001)
```

### Worked example: Oxygen → Necrosis

Goal: Low oxygen increases necrosis rate.

```python
# 1. Cell type has necrosis_rate=0.0001 from configure_cell_parameters → XML default = 0.0001 ✓
# 2. Rule: oxygen DECREASES necrosis (high O2 = less necrosis)
add_single_cell_rule(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",     # high oxygen → LESS necrosis
    behavior="necrosis",
    min_signal=0.0,            # base_value: necrosis at high O2
    max_signal=38,
    half_max=5.0,              # O2 level at 50% effect
    hill_power=4
)
# Result: necrosis = 0 at high O2, rises toward 0.0001 as O2 drops ✓
```

## 3. Common Mistakes

| # | Mistake | Fix |
|---|---------|-----|
| 1 | "From 0 towards 0" rule | Set nonzero XML default BEFORE adding rule |
| 2 | Dirichlet not enabled | Set `dirichlet_enabled=True` in `add_single_substrate` |
| 3 | Wrong cycle model | Check `get_available_cycle_models()` |
| 4 | Cells outside domain | Placement must fit within `create_simulation_domain` extents |
| 5 | Rule references missing substrate | Add substrate BEFORE rules |
| 6 | Missing `export_cells_csv` | Call before `create_physicell_project()` |
| 7 | No substrate interactions set | Call `set_substrate_interaction()` for each cell×substrate |
| 8 | `increases`/`decreases` swapped | "increases" = more signal → more behavior |
| 9 | Not polling simulation status | Call `get_simulation_status()` after `run_simulation()` |
| 10 | Forgot `compile_physicell_project` | Must compile before running |
| 11 | Reading binary output files directly | Use `get_simulation_analysis_overview()` or `get_cell_data()` — never `Read` .mat/.xml output files |

## 4. Substrate Defaults

| Substrate | Diffusion (μm²/min) | Decay (1/min) | Initial | Dirichlet | Units |
|-----------|---------------------|---------------|---------|-----------|-------|
| oxygen | 100000 | 0.1 | 38.0 | 38.0 (enabled) | mmHg |
| glucose | 30000 | 0.0025 | 16.9 | 16.9 | mM |
| chemokine | 50000 | 0.01 | 0.0 | — | dimensionless |
| drug | 100000 | 0.01-0.1 | 0.0 | varies | μM |
| VEGF | 20000 | 0.01 | 0.0 | — | dimensionless |

For oxygen, **always enable Dirichlet boundaries**:
```python
add_single_substrate("oxygen", 100000, 0.1, 38.0, dirichlet_enabled=True, dirichlet_value=38.0, units="mmHg")
```

## 5. Cell Placement Patterns

| Pattern | Use case | Key parameters |
|---------|----------|----------------|
| `random_disc` | Tumor mass, circular colony | center_x, center_y, radius, num_cells |
| `random_rectangle` | Tissue layer | x_min, x_max, y_min, y_max, num_cells |
| `single` | Individual cell | center_x, center_y |
| `grid` | Uniform monolayer | x_min, x_max, y_min, y_max, spacing |
| `annular` | Ring (e.g., immune cells around tumor) | center_x, center_y, radius, inner_radius, num_cells |

A 1000×1000 domain spans -500 to +500 per axis. Ensure placements fit within domain bounds.

## 6. Substrate Interactions

For every cell type × substrate pair, set uptake/secretion:

```python
set_substrate_interaction(cell_type="tumor", substrate="oxygen", uptake_rate=10.0)
set_substrate_interaction(cell_type="tumor", substrate="chemokine", secretion_rate=0.1)
```

Typical oxygen uptake: 10.0 (tumor), 5.0 (normal). Glucose: 0.5-2.0. Drug: 0.01-0.1.

## 7. Literature Validation (Optional)

### When to validate

Validate when the user explicitly requests literature validation or when the LiteratureValidation MCP server is available and the user wants evidence-based parameter guidance. Validation is not required for export.

### Prohibited tools and actions

| Do NOT use | Why | Use instead |
|---|---|---|
| `get_full_text_article` | Dumps 50KB+ into context | `validate_rule()` via LiteratureValidation MCP |
| `get_article_metadata` | Manual parameter extraction is unreliable | `validate_rule()` via LiteratureValidation MCP |
| WebSearch/WebFetch for parameter values | Web snippets lack rigor | `validate_rule()` via LiteratureValidation MCP |
| Task tool for literature work | Subagents lack MCP access | Do all literature work in main conversation |

### Workflow

**Phase 1 — Validate every rule (no exceptions)**

1. `get_rules_for_validation()` — get the full rule list from your PhysiCell session
2. `validate_rules_batch(name, rules)` — validate ALL rules from step 1. Do NOT skip any. Include `signal_units` when known (e.g., "mmHg" for oxygen). Edison automatically searches 150M+ papers — no paper collection management needed.
3. `get_validation_summary(name)` — review support levels

**Phase 2 — Store and report (mandatory)**

4. `store_validation_results(validations)` — persist results. Each dict needs `cell_type`, `signal`, `direction`, `behavior`, and optionally `collection_name`. The server reads PaperQA answer files directly from disk — you do NOT need to pass `raw_paperqa_answer`. VERDICT and DIRECTION are extracted server-side from the authoritative files written by `validate_rule()`. Direction mismatches are auto-flagged as `contradictory`. Store ALL rules, not a subset.
5. `get_validation_report()` — generates the formal report. **You MUST call this.** Do NOT substitute your own summary.

**Phase 3 — Revise contradictions (mandatory if any exist)**

6. For each `contradictory` rule:
    - **Direction mismatches** (literature says the opposite direction): Change the `direction` parameter in `add_single_cell_rule()`. The server detects these automatically from PaperQA's `DIRECTION:` verdict.
    - **Other contradictions**: Adjust half_max, hill_power, or base value based on the PaperQA evidence
    - Re-validate with `validate_rule()` or `validate_rules_batch()`
    - `store_validation_results()` to update
    - `get_validation_report()` to confirm no contradictions remain
7. For `unsupported` rules: review evidence and apply any suggested adjustments. These do not block export but should be noted.
8. Re-export configuration after all revisions.

### Signal units

Pass `signal_units` when validating to prevent unit confusion:
- **oxygen** → mmHg (half_max=3.75 means 3.75 mmHg ≈ 0.49% O₂)
- **glucose** → mM
- **pressure** → dimensionless (0-1)
- **drugs** → μM

**Support levels:** strong, moderate, weak, unsupported (from PaperQA VERDICT). `contradictory` is auto-assigned by the server when a direction mismatch is detected.

## 8. Post-Simulation Analysis

After `run_simulation()`:
1. Poll `get_simulation_status()` every 60 seconds until complete/failed
2. `get_simulation_analysis_overview()` — one-shot executive summary (start here)
3. `get_population_timeseries()` — cell counts over time, growth rates, doubling times
4. `get_timestep_summary(timestep=N)` — drill into a specific timepoint
5. `get_substrate_summary()` — substrate concentration statistics with gradients
6. `get_cell_data(cell_type="X", columns="col1,col2")` — filtered cell attributes
7. `generate_analysis_plot(plot_type="population_timeseries")` — save plots to disk
8. `generate_simulation_gif()` — spatial animation from SVG snapshots

**Do NOT** use `Read` to open binary output files (.mat, .xml output, .gif). These will crash with buffer overflow. Always use the analysis tools above.

If rules seem inactive, check `detailed_rules.txt` for "from X towards Y" values.

## 9. Reference Files

- **`references/rules-and-hill-functions.md`** — Full Hill function math and worked examples
- **`references/parameter-reference.md`** — Typical biological parameter values by cell type
- **`references/troubleshooting.md`** — Error diagnosis by symptom
- **`references/uq-calibration-workflow.md`** — Sensitivity analysis and calibration
- **`references/physiboss-integration.md`** — Boolean network (MaBoSS/PhysiBoSS) integration
- **`references/literature-validation.md`** — Edison PaperQA3 API integration details

## 10. Quick Start Template

Basic tumor growth simulation (no literature validation):

```
1. create_session()
2. analyze_biological_scenario("Tumor growth with oxygen-dependent proliferation and necrosis")
3. create_simulation_domain(domain_x=1000, domain_y=1000, domain_z=20, max_time=4320)
4. add_single_substrate("oxygen", 100000, 0.1, 38.0, dirichlet_enabled=True, dirichlet_value=38.0, units="mmHg")
5. add_single_cell_type("tumor")
6. configure_cell_parameters(cell_type="tumor", volume_total=2500, motility_speed=0.5, apoptosis_rate=5.31667e-5, necrosis_rate=0.00277)
7. set_substrate_interaction(cell_type="tumor", substrate="oxygen", uptake_rate=10.0)
8. add_single_cell_rule(cell_type="tumor", signal="oxygen", direction="decreases", behavior="necrosis", min_signal=0, half_max=3.75, hill_power=8)
9. add_single_cell_rule(cell_type="tumor", signal="pressure", direction="decreases", behavior="cycle entry", min_signal=0, half_max=1.0, hill_power=4)
10. place_initial_cells(cell_type="tumor", pattern="random_disc", num_cells=200, radius=100)
11. export_xml_configuration()
12. export_cell_rules_csv()
13. export_cells_csv()
14. create_physicell_project("tumor_growth")
15. compile_physicell_project("tumor_growth")
16. run_simulation("tumor_growth")
17. get_simulation_status(simulation_id)   # poll until complete
18. get_simulation_analysis_overview(simulation_id)
19. get_population_timeseries(simulation_id)
20. generate_analysis_plot(simulation_id, plot_type="population_timeseries")
21. generate_simulation_gif(simulation_id)
```
