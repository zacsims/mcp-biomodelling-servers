---
name: spatial-analysis
description: >
  Run spatial statistics panel analysis on PhysiCell simulation output using the
  SpatialTissue MCP server. Use after PhysiCell simulation completes to quantify
  spatial organisation, neighbourhood composition, microenvironment topics (LDA),
  cell-graph network structure, and topological landscape (Mapper).
  Covers Ripley's K, Clark–Evans, diversity indices, LDA, betweenness/assortativity,
  and the full metric catalogue.
compatibility: Requires SpatialTissue MCP server connection
metadata:
  author: simsz
  version: "1.0"
---

# SpatialTissue Spatial Analysis Skill

## Non-Negotiable Rules

1. **Never pass dead cells to LDA.** `run_spatial_lda` automatically sets `include_dead_cells=False`. Do not try to override this.

2. **Call `run_panel` after building the panel.** Always call `view_panel` to confirm configuration before calling `run_panel`.

3. **Use MCP tool calling.** All tools must be called via `mcp__SpatialTissue__<tool>`. Never invoke Python directly via Bash or subprocess.

4. **output_folder is the PhysiCell output directory.** It must contain `output*.xml` files. Obtain it from `mcp__PhysiCell__get_simulation_status()`.

5. **Preset panels are a fast start.** If the user wants custom metrics, `load_preset_panel` first then `add_metric`/`remove_metric`.

---

## 1. Overview

SpatialTissue wraps the `spatialtissuepy` Python library as a 10-tool MCP server. It operates on PhysiCell simulation output and answers questions like:

- *Are tumour cells clustered or dispersed?* → Ripley's K, Clark–Evans index
- *What immune cells surround tumour cells?* → neighbourhood composition / entropy
- *Are there recurring microenvironmental patterns?* → Spatial LDA topics
- *Are cell types mixing or segregating?* → type assortativity, homophily ratio
- *What is the topological landscape?* → Mapper (TDA)

---

## 2. Workflow

```
create_session                         ← start a session (auto-creates if omitted)
  ↓
load_preset_panel(preset='basic')      ← fast start with 6 core metrics
  OR
add_metric (repeat)                    ← build custom panel metric by metric
  ↓
view_panel                             ← confirm configuration
  ↓
run_panel(output_folder=...)           ← run all metrics × all timesteps → CSV + plot
  ↓
(optional) run_spatial_lda(...)        ← discover microenvironment topics
(optional) run_network_analysis(...)   ← cell-graph structure
(optional) run_spatial_mapper(...)     ← topological landscape (single timestep)
```

---

## 3. Preset Panels

| Preset | Metrics included |
|--------|-----------------|
| `basic` | cell_counts, cell_proportions, cell_density, mean_nearest_neighbor_distance, clark_evans_index, shannon_diversity |
| `spatial` | cell_counts, mean_nearest_neighbor_distance, clark_evans_index, ripleys_k (r=25,50,100,200), l_function, g_function_summary, convex_hull_metrics |
| `neighborhood` | cell_counts, cell_proportions, mean_neighborhood_entropy, mean_neighborhood_composition, neighborhood_homogeneity, interaction_strength_matrix |
| `comprehensive` | All of the above plus simpson_diversity, spatial_extent, centroid |

---

## 4. Panel Metrics Catalogue

All metrics below can be added with `add_metric(metric_type=..., params_json=...)`.

### Population

| Metric | Key params | What it measures |
|--------|-----------|-----------------|
| `cell_counts` | — | Raw count per cell type |
| `cell_proportions` | — | Fraction 0–1 per cell type |
| `cell_type_ratio` | `numerator`, `denominator` | Ratio of two cell type counts |
| `cell_density` | — | Cells per unit area (µm²) |
| `shannon_diversity` | — | Shannon entropy of cell type distribution |
| `simpson_diversity` | — | Simpson diversity index |
| `spatial_extent` | — | Bounding box dimensions (µm) |
| `centroid` | — | Population spatial centroid (x, y) |

### Spatial

| Metric | Key params | What it measures |
|--------|-----------|-----------------|
| `mean_nearest_neighbor_distance` | — | Mean distance to nearest cell of any type |
| `cross_type_nnd` | `type_a`, `type_b` | Mean nearest-neighbour distance between two types |
| `clark_evans_index` | — | Aggregation index: <1 = clustered, 1 = random, >1 = dispersed |
| `ripleys_k` | `radii=[25,50,100]` | Ripley's K at each radius (>1 = clustering) |
| `l_function` | `radii=[25,50,100]` | L(r)-r: deviation from CSR (positive = clustering) |
| `g_function_summary` | `radii=[10,25,50]` | G-function: empirical nearest-neighbour CDF |
| `spatial_autocorrelation` | — | Moran's I: spatial autocorrelation of cell density |
| `convex_hull_metrics` | — | Area, perimeter, solidity of cell population convex hull |

### Neighbourhood

| Metric | Key params | What it measures |
|--------|-----------|-----------------|
| `mean_neighborhood_entropy` | `radius=50` | Mean Shannon entropy of cell-type neighbourhoods |
| `mean_neighborhood_composition` | `radius=50` | Mean count of each cell type in neighbourhoods |
| `neighborhood_homogeneity` | `radius=50` | Fraction of neighbours with same type |
| `colocalization_score` | `type_a`, `type_b`, `radius=50` | Co-localization between two cell types |
| `mixing_score` | `radius=50` | Global cell type mixing index |
| `interaction_strength_matrix` | `radius=50` | Pairwise interaction strengths (N_types × N_types) |
| `border_contact_score` | `type_a`, `type_b`, `radius=50` | Contact fraction at type boundaries |
| `infiltration_score` | `infiltrating_type`, `host_type`, `radius=50` | Infiltration depth of one type into another |

---

## 5. Parameter Reference

### `ripleys_k` / `l_function`
```json
{"radii": [25, 50, 100, 200]}
```
- `radii`: list of radii in µm to evaluate. More radii = more columns in output.
- Typical PhysiCell scales: 10–500 µm.

### `g_function_summary`
```json
{"radii": [10, 25, 50]}
```

### `mean_neighborhood_entropy` / `mean_neighborhood_composition` / `neighborhood_homogeneity` / `mixing_score`
```json
{"radius": 50}
```
- `radius`: neighbourhood search radius in µm. Match to typical cell diameter × 3–5.

### `colocalization_score` / `border_contact_score`
```json
{"type_a": "cancer", "type_b": "CD8_T_cell", "radius": 50}
```
- `type_a`, `type_b`: exact cell type names as defined in PhysiCell config.

### `interaction_strength_matrix`
```json
{"radius": 50}
```
- Returns one column per type-pair: `interaction_cancer_CD8_T_cell`, etc.

### `cell_type_ratio`
```json
{"numerator": "CD8_T_cell", "denominator": "cancer"}
```

### `cross_type_nnd`
```json
{"type_a": "cancer", "type_b": "CD8_T_cell"}
```

### `infiltration_score`
```json
{"infiltrating_type": "CD8_T_cell", "host_type": "cancer", "radius": 50}
```

---

## 6. Complex Analyses

### 6a. Spatial LDA — Microenvironment Topic Discovery

**Tool:** `run_spatial_lda(output_folder, n_topics, neighborhood_radius, timestep_indices)`

**When to use:** Discover whether recurring spatial cell-type patterns exist across the tissue.

**n_topics guidance:**
- 2–3: simple tumour/stroma/immune separation
- 5–8: moderate complexity (recommended starting point)
- 10+: fine-grained tissue compartments; risk of over-fitting for small simulations

**Interpretation:**
- Topics are probability distributions over cell types in local neighbourhoods
- High topic coherence = consistent spatial pattern
- Topic weights per cell show spatial localisation

**Example:**
```
run_spatial_lda(
  output_folder="/sim/output",
  n_topics=5,
  neighborhood_radius=50.0,
  timestep_indices="[0, 5, 10, 15, 20]"   # or null for all
)
```

### 6b. Network Analysis — Cell Graph Metrics

**Tool:** `run_network_analysis(output_folder, metrics, graph_method, radius, timestep_indices)`

**Graph construction methods:**
| method | edges connect | use when |
|--------|--------------|----------|
| `proximity` | all cells within `radius` µm | general; most common |
| `knn` | k nearest neighbours | uniform density; long-range |
| `delaunay` | Delaunay triangulation | geometry-driven; no radius needed |

**Available metrics:**
| metric | what it reveals |
|--------|----------------|
| `degree` | Mean number of cell-cell contacts |
| `betweenness` | Communication bridges between tissue regions |
| `closeness` | Information propagation speed |
| `clustering` | Local clustering coefficient (triangle motifs) |
| `assortativity` | Whether similar cell types connect preferentially |
| `homophily` | Fraction of same-type neighbours |

**Example:**
```
run_network_analysis(
  output_folder="/sim/output",
  metrics='["betweenness","assortativity","homophily"]',
  graph_method="proximity",
  radius=30.0
)
```

**Tip:** `assortativity > 0` = cell types segregate; `assortativity < 0` = cell types mix.

### 6c. Topological Mapper — Landscape Analysis

**Tool:** `run_spatial_mapper(output_folder, filter_fn, n_intervals, overlap, timestep)`

**filter_fn options:**
| value | lens function | reveals |
|-------|--------------|---------|
| `density` | local cell density | high/low density regions |
| `pca` | first principal component of coordinates | global gradient |
| `eccentricity` | graph eccentricity | core vs peripheral cells |
| `distance_to_type:cancer` | distance to a specific cell type | invasion front structure |

**n_intervals / overlap guidance:**
- `n_intervals=10, overlap=0.5` — good default
- Increase `n_intervals` for finer resolution (more nodes)
- Increase `overlap` (0.3–0.7) for smoother connectivity

**Interpretation:**
- Isolated Mapper components = disconnected tissue regions
- Loops in the graph = cyclic spatial patterns (e.g. ring-like invasion fronts)
- Long branches = gradient structures (e.g. hypoxic core → oxygenated periphery)

**Example:**
```
run_spatial_mapper(
  output_folder="/sim/output",
  filter_fn="distance_to_type:cancer",
  n_intervals=12,
  overlap=0.5,
  timestep=-1    # last timestep
)
```

---

## 7. Reading Results

### Panel CSV (`panel_<timestamp>.csv`)
- Rows = timesteps (indexed by simulation time in minutes/hours)
- Columns = metric output names
- Multi-output metrics (ripleys_k with 3 radii) produce 3 columns

### Inline summary table
The `run_panel` tool returns a table of first→last timestep values with trend arrows:
- ↑ = metric grew >5% relative
- ↓ = metric dropped >5% relative
- ~ = stable

**Reading `clark_evans_index`:**
- < 1.0: cells are clustered (common in tumours)
- = 1.0: random spatial distribution
- > 1.0: cells are uniformly dispersed

**Reading `shannon_diversity`:**
- Higher = more even mix of cell types
- Lower = one cell type dominates
- Useful for tracking immune infiltration over time

**Reading `ripleys_k`:**
- Values > theoretical CSR line = spatial clustering at that radius
- Compare across timesteps to track clump formation/dissolution

### Network CSVs
- `network_metrics_<timestamp>.csv`: time × (degree, betweenness per type, assortativity, homophily, ...)

### LDA CSVs
- `lda_topics_<timestamp>.csv`: topic summary matrix

---

## 8. Integration with PhysiCell

The `output_folder` for all SpatialTissue tools is the PhysiCell simulation output directory.

```python
# 1. Get the output folder from PhysiCell server
status = mcp__PhysiCell__get_simulation_status()
# Look for: "output_folder: /path/to/project/output"

# 2. Create a spatial analysis session
mcp__SpatialTissue__create_session()

# 3. Load a preset and run
mcp__SpatialTissue__load_preset_panel(preset="neighborhood")
mcp__SpatialTissue__run_panel(output_folder="/path/to/project/output")
```

Alternatively, `create_physicell_project()` prints the project directory; append `/output` to get the output folder.

---

## 9. Error Recovery

| Error | Fix |
|-------|-----|
| `spatialtissuepy not installed` | `pip install spatialtissuepy[viz,network]` |
| `Metric not found in registry` | Check spelling against the catalogue table above |
| `No timesteps selected` | Pass valid integer indices or `null` for all |
| `output_folder not found` | Verify the simulation completed: `get_simulation_status()` |
| `Panel is empty` | Call `load_preset_panel` or `add_metric` before `run_panel` |
| `distance_to_type: cell type missing` | Check exact cell type name via `get_cell_data()` |
