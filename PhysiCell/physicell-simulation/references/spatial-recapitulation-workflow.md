# Spatial Recapitulation Workflow

Reverse-engineer a PhysiCell simulation from spatial statistics alone, without access to the original configuration, rules, or parameters.

## Prerequisites

- A completed PhysiCell simulation output folder (the "target")
- Both MCP servers connected: PhysiCell MCP (`mcp__PhysiCell__`) and spatialtissuepy MCP (`mcp__spatialtissuepy__`)
- **Blind constraint**: the agent sees ONLY the output folder path — no XML config, no rules CSV, no parameter files

## Phase 1: Characterize the Target

Extract a "spatial fingerprint" from the target simulation using spatialtissuepy.

### Step 1: Load simulation and inspect

```python
# Load the full simulation (all timesteps)
synthetic_load_physicell_simulation(session_id=SID, output_folder="/path/to/target/output")

# Convert final timestep to spatial data for analysis
synthetic_timestep_to_spatial_data(session_id=SID, timestep_index=-1, data_key="final")

# Basic inventory
data_get_cell_types(session_id=SID, data_key="final")
data_get_cell_counts(session_id=SID, data_key="final")
data_get_bounds(session_id=SID, data_key="final")
```

### Step 2: Population dynamics (growth curves)

```python
synthetic_cell_count_trajectory(session_id=SID)
synthetic_type_proportions_trajectory(session_id=SID)
```

Record: initial counts, final counts, doubling times, proportion trends.

### Step 3: Spatial statistics

Compute each for the final timestep. Repeat per cell type where noted.

```python
# Clustering — run per cell type AND for all cells
statistics_ripleys_h(session_id=SID, data_key="final", cell_type="tumor", max_radius=200)
statistics_ripleys_h(session_id=SID, data_key="final", max_radius=200)

# Colocalization — run for every ordered pair of cell types
statistics_colocalization_quotient(session_id=SID, data_key="final",
    type_a="tumor", type_b="immune", radius=50)

# Density — global and per type
spatial_density(session_id=SID, data_key="final", radius=50)
spatial_density(session_id=SID, data_key="final", radius=50, cell_type="tumor")

# Nearest-neighbor G function
statistics_nearest_neighbor_g(session_id=SID, data_key="final")

# Hotspot detection
statistics_getis_ord_gi_star(session_id=SID, data_key="final", radius=50)
```

### Step 4: Network analysis

```python
network_build_proximity_graph(session_id=SID, data_key="final", radius=30)
network_attribute_mixing_matrix(session_id=SID, data_key="final")
network_type_assortativity(session_id=SID, data_key="final")
network_clustering_coefficient(session_id=SID)
```

### Step 5: Record the fingerprint

Organize all values into a structured scorecard:

```
Target Fingerprint:
  Population:     {tumor: 1423, immune: 200, total: 1623}
  Growth:         200 → 1623 over 72h, doubling ≈ 21h
  Ripley's H(100): {tumor: +45.2, immune: -3.1, all: +38.7}
  H peak radius:  tumor ≈ 80 μm
  CLQ:            {(tumor,immune): 0.72, (immune,tumor): 1.45}
  Mean density:   0.0032 cells/μm²
  Mean NND:       14.3 μm
  Gi* hotspots:   1 central
  Mixing matrix:  85% same-type edges
  Assortativity:  +0.45
  Clustering coeff: 0.31
```

## Phase 2: Translate Statistics to Model

Use the translation table below to choose PhysiCell parameters. Work top-to-bottom — population dynamics first, spatial organization second.

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
2. **Cell types**: One `add_single_cell_type()` per unique type found.
3. **Population dynamics**: Calculate cycle rate from growth curve. Set death rates to match observed plateau/decline. This is the highest priority — get the numbers right first.
4. **Substrates**: Infer from spatial patterns:
   - Central necrotic core → oxygen with Dirichlet boundaries
   - Type colocalization (CLQ > 1) → chemokine with secretion/chemotaxis pairing
   - Density gradients → substrate with uptake creating local depletion
5. **Motility**: From Ripley's H and NND:
   - High H (clustering) → low speed (0.1-0.5 μm/min)
   - Low H (dispersion) → high speed (1.0-5.0 μm/min)
6. **Rules**: From colocalization and mixing:
   - CLQ > 1 between types → chemotaxis rule connecting them
   - Central density gradient → oxygen-decreases-necrosis + pressure-decreases-cycle-entry
   - Type segregation → differential adhesion
7. **Initial conditions**: From hotspot pattern and early-timestep spatial extent:
   - Single compact cluster → `random_disc`
   - Scattered → `random_rectangle`
   - Ring → `annular`
8. **Search literature** for parameter values before adding rules (standard workflow).

## Phase 3: Compare and Iterate

After running the recapitulation simulation, compute the same spatial fingerprint and compare.

### Comparison procedure

```python
# Load recapitulation output into a NEW spatialtissuepy session
synthetic_load_physicell_simulation(session_id=SID2, output_folder="/path/to/recap/output")
synthetic_timestep_to_spatial_data(session_id=SID2, timestep_index=-1, data_key="final")

# Compute the same statistics as Phase 1
# Compare each metric to the target fingerprint
```

### Adjustment decision table

| Mismatch | Direction | Adjustment |
|---|---|---|
| Population count | Too high (>20% over) | Decrease cycle rate or increase death rate |
| Population count | Too low (>20% under) | Increase cycle rate or decrease death rate |
| Ripley's H | Too high (over-clustered) | Increase `motility_speed` or decrease adhesion |
| Ripley's H | Too low (under-clustered) | Decrease `motility_speed` or increase adhesion |
| CLQ direction | Target >1 but got <1 | Add or strengthen chemotaxis rules between types |
| CLQ direction | Target <1 but got >1 | Remove cross-type chemotaxis; add competition |
| Type assortativity | Off by >0.15 | Adjust differential adhesion or chemotaxis |
| Mean NND | Off by >20% | Adjust cell volume, motility, or domain size |
| Hotspot count | Wrong number | Adjust number of initial placement sites |

### Convergence criteria

Accept the recapitulation when:
- Population counts within 15% of target for all cell types
- Ripley's H sign matches for all types and magnitude within 20%
- CLQ direction (>1 or <1) matches for all type pairs
- Type assortativity within 0.15 of target

## UQ Fallback

If manual tuning plateaus after 3+ iterations without convergence:

1. **Setup**: `setup_uq_analysis()` on the recapitulation project
2. **Define parameters**: Use the parameters you've been manually adjusting as UQ variables with bounds centered on current values
3. **Define QoIs**: Use cell count targets from the target fingerprint (`cell_count:tumor`, `live_cells`, etc.)
4. **Sensitivity analysis**: `run_sensitivity_analysis(method="Sobol")` to identify which parameters matter most
5. **Calibration**: `run_bayesian_calibration()` or `run_abc_calibration()` to optimize population dynamics
6. **Spatial check**: After each calibration result, manually compute spatial statistics to verify spatial convergence — UQ currently only optimizes scalar QoIs (cell counts), not spatial patterns

This hybrid approach lets UQ handle population dynamics while the agent handles spatial pattern matching.

## Tips

1. Start with the simplest model that could produce the target pattern — minimal cell types, one substrate
2. Add complexity only when statistics demand it (CLQ ≠ 1 means you need a connecting mechanism)
3. Population dynamics first, spatial organization second — if the cell counts are wrong, the spatial stats will be meaningless
4. Use visualizations for sanity checks: `viz_plot_cell_types`, `viz_plot_ripleys_curve`, `viz_plot_colocalization_heatmap`
5. Also convert the target's first timestep (`timestep_index=0`) to get clues about initial conditions
6. When in doubt about parameter values, use `search_literature()` from LiteratureValidation MCP
