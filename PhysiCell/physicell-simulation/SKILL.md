---
name: physicell-simulation
description: >
  Build, configure, and run PhysiCell multicellular simulations using the PhysiCell MCP server.
  Use when the user asks to model tumors, cells, tissues, or multicellular biological systems.
  Covers simulation setup, cell rules (Hill functions), initial conditions, PhysiBoSS boolean
  networks, UQ/calibration, and literature-informed model building. Prevents common configuration
  mistakes like the "from 0 towards 0" rule bug.
compatibility: Requires PhysiCell MCP server connection
metadata:
  author: simsz
  version: "3.0"
---

# PhysiCell Simulation Skill

## Non-Negotiable Rules

These rules are enforced by server-side validation. Violating them will cause tool calls to fail.

1. **Never edit generated files.** Do NOT use Edit, Write, or Bash to modify XML configs, `cell_rules.csv`, or `cells.csv`. These are managed by the MCP server. Direct edits will be overwritten and desync session state. Fix problems by calling the appropriate MCP tool, then re-export.

2. **All MCP tools must be called via MCP tool calling** (prefix `mcp__PhysiCell__`). Do NOT run them via Bash, subprocess, or npx.

3. **Never extract parameter values from raw sources.** Do NOT read papers, abstracts, or web results and manually pick out numbers for half_max, hill_power, rates, etc. Use `search_literature()` from the LiteratureValidation MCP for evidence-based parameter guidance.

4. **Always search literature before adding rules.** Before choosing half_max, hill_power, or rate values, call `search_literature()` to get evidence-based guidance. Do NOT rely on training data or memorized citations — always make the actual tool call. After adding each rule, record the evidence with `store_rule_justification()`. If `search_literature()` fails with a connection error, you may proceed without it, but you must attempt the call first.

## 1. Tool Ordering

Follow this sequence. Do not skip steps. Literature search and justifications happen *alongside* rule creation, not as a separate phase.

```
create_session
  → analyze_biological_scenario
  → create_simulation_domain
  → add_single_substrate              (repeat per substrate)
  → add_single_cell_type              (repeat per cell type)
  → configure_cell_parameters         (repeat per cell type)
  → set_substrate_interaction         (repeat per cell×substrate pair)
  → [setter tools as needed]          (see Section 2 — set nonzero defaults BEFORE rules)
  → search_literature                 (call BEFORE adding rules — get evidence for parameter values)
  → add_single_cell_rule              (repeat per rule — see Section 2)
  → store_rule_justification          (record evidence AFTER each rule)
  → place_initial_cells               (repeat per placement)
  → get_rule_justifications           (generate evidence report before export)
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
| cycle entry / exit from cycle phase N | 0 unless set | `set_cycle_transition_rate()` |
| apoptosis / necrosis | 0 unless set | `configure_cell_parameters` (`apoptosis_rate`, `necrosis_rate`) |
| migration speed / persistence / bias | 0 unless set | `configure_cell_parameters` (`motility_speed`, `persistence_time`, `migration_bias`) |
| X secretion / X uptake | 0 | `set_substrate_interaction()` |
| transition to X | 0 | `set_cell_transformation_rate()` |
| attack X / phagocytose X / fuse to X | 0 | `set_cell_interaction()` |
| phagocytose dead cells | 0 | `configure_cell_interactions()` |
| attachment / detachment rate | 0 | `configure_cell_mechanics()` |
| damage / damage repair rate | 0 | `configure_cell_integrity()` |
| chemotactic response to X | 0 | `set_advanced_chemotaxis()` |
| custom behaviors | 0 | No setter — use nonzero `min_signal` |

The server validates rules and rejects "from 0 towards 0" with guidance on which setter to use.

### Setter examples

```python
# Cycle entry / proliferation rate (0.00072 ≈ 24h doubling):
set_cycle_transition_rate(cell_type="tumor", rate=0.00072)
# Go-or-grow: motile cells proliferate slower:
set_cycle_transition_rate(cell_type="motile_tumor", rate=0.0002)

# Death rates, motility (only provided params are changed — omitted params keep current values):
configure_cell_parameters(cell_type="tumor", apoptosis_rate=0.001, necrosis_rate=0.00277, motility_speed=0.5)
# Migration bias (0=random, 1=fully directed):
configure_cell_parameters(cell_type="motile_tumor", migration_bias=0.85)

# Secretion/uptake (secretion_target and net_export_rate also available):
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

# Basic chemotaxis (single substrate, attraction):
set_chemotaxis(cell_type="motile", substrate="oxygen", enabled=True, direction=1)

# Advanced chemotaxis (per-substrate sensitivity — REQUIRED before "chemotactic response to X" rules):
set_advanced_chemotaxis(cell_type="motile", substrate="oxygen", sensitivity=0.5, enabled=True)
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

## 7. Literature-Informed Model Building

### Default behavior

**Always search literature before choosing parameter values for rules.** This is the standard workflow, not an optional step. For every rule you add, you should:
1. Call `search_literature()` to find evidence for the parameter values
2. Use the returned evidence to choose half_max, hill_power, and rate values
3. Call `store_rule_justification()` to record your evidence basis

Do NOT skip literature search because you "already know" the answer from training data. Your training data may be outdated or wrong. The tool call takes seconds and returns current, cited evidence.

If `search_literature()` fails with a connection error (MCP server unavailable), you may fall back to the parameter reference tables. But you must **attempt the call first** — do not pre-decide to skip it.

### Workflow

Search literature *during* model building, not as a post-hoc validation step.

**1. Search for evidence before/while adding rules**

Use `search_literature()` from the LiteratureValidation MCP to ask any biological question:

```python
# Ask about mechanisms
search_literature("What is the effect of oxygen on tumor cell necrosis?")

# Ask about parameter values
search_literature("What oxygen concentration causes half-maximal necrosis in solid tumors?")

# Ask about cell behaviors
search_literature("What is a typical migration speed for breast cancer cells in vitro?")
```

Edison automatically searches 150M+ papers and returns answers with citations. Results are cached.

**2. Add rules informed by literature**

Use the evidence to choose parameter values, then add the rule:

```python
add_single_cell_rule(
    cell_type="tumor", signal="oxygen", direction="decreases",
    behavior="necrosis", half_max=3.75, hill_power=8
)
```

**3. Store justification for each rule**

After adding a rule, record why it's biologically valid:

```python
store_rule_justification(
    cell_type="tumor", signal="oxygen", direction="decreases", behavior="necrosis",
    justification="Tumor cells undergo necrosis below ~5 mmHg O2. Half-max of 3.75 mmHg is consistent with published hypoxia thresholds.",
    key_citations="Vaupel 2004, Grimes 2014"
)
```

**4. Generate justification report**

After all rules are added:

```python
get_rule_justifications()
```

This produces a report showing each rule's evidence basis, flags unjustified rules, and exports to `rule_justifications.md`.

### Tips

- **Always call the tool** — do not skip `search_literature()` because you think you know the answer
- Ask specific questions (e.g., "What is the EC50 for oxygen-dependent necrosis?") rather than vague ones
- PubMed and bioRxiv MCPs are fine for discovery and browsing; Edison (`search_literature`) is best for deep evidence search with citations
- The Task tool cannot access MCP tools — do all literature work in the main conversation
- One `search_literature()` call per biological question — batch related questions into separate calls

## 8. Post-Simulation Analysis

All analysis tools accept either `simulation_id` (from `run_simulation()`) or `output_folder` (direct path). They use pcdl (PhysiCell Data Loader) to read binary `.mat` output files.

After `run_simulation()`:
1. Poll `get_simulation_status()` every 60 seconds until complete/failed
2. `get_simulation_analysis_overview()` — one-shot executive summary (start here). Shows cell counts at first/last timestep, spatial extent, substrate stats, and suggests next tools.
3. `get_population_timeseries()` — cell counts over time, growth rates, doubling times. Subsamples if >200 timesteps.
4. `get_timestep_summary(timestep=N)` — drill into a specific timepoint (0-based index, -1 for latest)
5. `get_substrate_summary(timestep=N)` — substrate concentration statistics with center-vs-edge gradients
6. `get_cell_data(cell_type="X", columns="col1,col2", sort_by="total_volume", max_rows=50)` — filtered cell attributes with statistics
7. `generate_analysis_plot(plot_type=T)` — save plots to disk. Types: `population_timeseries`, `cell_scatter`, `substrate_contour`
8. `generate_simulation_gif()` — spatial animation from SVG snapshots

**Do NOT** use `Read` to open binary output files (.mat, .xml output, .gif). Always use the analysis tools above.

If rules seem inactive, check `detailed_rules.txt` for "from X towards Y" values.

## 9. Reference Files

- **`references/rules-and-hill-functions.md`** — Full Hill function math and worked examples
- **`references/parameter-reference.md`** — Typical biological parameter values by cell type
- **`references/troubleshooting.md`** — Error diagnosis by symptom
- **`references/uq-calibration-workflow.md`** — Sensitivity analysis and calibration
- **`references/physiboss-integration.md`** — Boolean network (MaBoSS/PhysiBoSS) integration
- **`references/literature-search.md`** — Edison PaperQA3 literature search details

## 10. Quick Start Template

Basic tumor growth simulation:

```
1. create_session()
2. analyze_biological_scenario("Tumor growth with oxygen-dependent proliferation and necrosis")
3. create_simulation_domain(domain_x=1000, domain_y=1000, domain_z=20, max_time=4320)
4. add_single_substrate("oxygen", 100000, 0.1, 38.0, dirichlet_enabled=True, dirichlet_value=38.0, units="mmHg")
5. add_single_cell_type("tumor")
6. configure_cell_parameters(cell_type="tumor", volume_total=2500, motility_speed=0.5, apoptosis_rate=5.31667e-5, necrosis_rate=0.00277)
7. set_substrate_interaction(cell_type="tumor", substrate="oxygen", uptake_rate=10.0)
8. add_single_cell_rule(cell_type="tumor", signal="oxygen", direction="decreases", behavior="necrosis", min_signal=0, half_max=3.75, hill_power=8)
9. store_rule_justification(cell_type="tumor", signal="oxygen", direction="decreases", behavior="necrosis", justification="Hypoxia-induced necrosis threshold ~5 mmHg", key_citations="Vaupel 2004")
10. add_single_cell_rule(cell_type="tumor", signal="pressure", direction="decreases", behavior="cycle entry", min_signal=0, half_max=1.0, hill_power=4)
11. store_rule_justification(cell_type="tumor", signal="pressure", direction="decreases", behavior="cycle entry", justification="Contact inhibition of proliferation", key_citations="Helmlinger 1997")
12. place_initial_cells(cell_type="tumor", pattern="random_disc", num_cells=200, radius=100)
13. get_rule_justifications()
14. export_xml_configuration()
15. export_cell_rules_csv()
16. export_cells_csv()
17. create_physicell_project("tumor_growth")
18. compile_physicell_project("tumor_growth")
19. run_simulation("tumor_growth")
20. get_simulation_status(simulation_id)   # poll until complete
21. get_simulation_analysis_overview(simulation_id)
22. get_population_timeseries(simulation_id)
23. generate_analysis_plot(simulation_id, plot_type="population_timeseries")
24. generate_simulation_gif(simulation_id)
```
