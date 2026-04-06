# pcdl Analysis Reference

pcdl (PhysiCell Data Loader) reads PhysiCell binary output and exposes it as pandas DataFrames. All MCP analysis tools use pcdl internally — you never call it directly.

## Output Files

PhysiCell writes one set of files per saved timestep:

| File pattern | Contents |
|---|---|
| `output00000000.xml` | Timestep index — points to the .mat files below |
| `output00000000_cells.mat` | Cell agent state (positions, volumes, phases, custom vars) |
| `output00000000_microenvironment0.mat` | Substrate concentrations on the mesh |

pcdl's `TimeStep` class reads one XML index file, then lazy-loads `.mat` files as requested.

## TimeStep API

Methods used by the MCP analysis tools:

| Method | Returns | Loaded when |
|---|---|---|
| `get_cell_df()` | `DataFrame` — one row per cell agent | `microenv=False` (default) |
| `get_time()` | `float` — simulation time in minutes | always |
| `get_substrate_list()` | `list[str]` — substrate names | `microenv=True` |
| `get_conc_df()` | `DataFrame` — one row per mesh voxel | `microenv=True` |
| `plot_scatter(focus)` | `matplotlib.Figure` — spatial cell scatter | `microenv=False` |
| `plot_contour(focus)` | `matplotlib.Figure` — substrate heatmap | `microenv=True` |

Constructor pattern used everywhere:
```python
ts = pcdl.TimeStep(xml_name, output_path=str(folder),
                   microenv=False, graph=False, verbose=False)
```

Set `microenv=True` only when substrate data is needed (saves memory).

## Cell DataFrame Columns

`get_cell_df()` returns a DataFrame with these key columns:

### Always present

| Column | Type | Description |
|---|---|---|
| `cell_type` | str | Cell type name (e.g., "tumor", "macrophage") |
| `current_phase` | str | Cell cycle / death phase string (see below) |
| `position_x` | float | X position (um) |
| `position_y` | float | Y position (um) |
| `position_z` | float | Z position (um) |
| `total_volume` | float | Cell volume (um^3) |
| `pressure` | float | Mechanical pressure from neighbors |

### Common optional columns

| Column | Type | When present |
|---|---|---|
| `total_attack_time` | float | When immune interactions are configured |
| `damage` | float | When cell integrity is configured |
| `oxygen` | float | Internalized substrate (per substrate name) |
| Custom variable names | float | Any custom variables defined in the model |

The full column list depends on the model. Use `get_timestep_summary()` to see available columns for a specific simulation.

## Live vs Dead Phase Classification

The analysis tools classify cells as live or dead using exact phase string matching:

### Live phases
```
Ki67_negative, Ki67_positive, S, G0G1, G0, G1,
G1a, G1b, G1c, G2, G2M, M, live
```

### Dead phases
```
apoptotic, necrotic_swelling, necrotic_lysed, debris
```

Any phase not in the live set is counted as dead. The `get_timestep_summary()` tool further breaks dead cells into "Apoptotic" (phase = `apoptotic`) and "Necrotic" (phase contains `necrotic` or `debris`).

## Substrate DataFrame Columns

`get_conc_df()` returns a DataFrame with one row per mesh voxel:

| Column | Type | Description |
|---|---|---|
| `mesh_center_m` | float | Voxel center X coordinate (um) |
| `mesh_center_n` | float | Voxel center Y coordinate (um) |
| *substrate_name* | float | Concentration for each substrate (one column per substrate) |

For example, a simulation with oxygen and chemokine substrates produces columns: `mesh_center_m`, `mesh_center_n`, `oxygen`, `chemokine`.

## Tool-to-pcdl Mapping

Which analysis tool calls which pcdl methods:

| MCP Tool | pcdl Methods | microenv |
|---|---|---|
| `get_simulation_analysis_overview` | `get_cell_df`, `get_time`, `get_substrate_list`, `get_conc_df` | First/mid/last: False; Final: True |
| `get_timestep_summary` | `get_cell_df`, `get_time` | False |
| `get_population_timeseries` | `get_cell_df`, `get_time` (per timestep) | False |
| `get_substrate_summary` | `get_time`, `get_substrate_list`, `get_conc_df` | True |
| `get_cell_data` | `get_cell_df`, `get_time` | False |
| `generate_analysis_plot` (population_timeseries) | `get_cell_df`, `get_time` (per timestep) | False |
| `generate_analysis_plot` (cell_scatter) | `get_time`, `plot_scatter` | False |
| `generate_analysis_plot` (substrate_contour) | `get_time`, `get_substrate_list`, `plot_contour` | True |

## Typical Analysis Workflow

```
1. get_simulation_analysis_overview()    # Executive summary — start here
2. get_population_timeseries()           # Growth curves across all timesteps
3. get_timestep_summary(timestep=-1)     # Drill into final timepoint
4. get_substrate_summary(timestep=-1)    # Substrate gradients
5. get_cell_data(cell_type="tumor",      # Filtered cell attributes
     columns="total_volume,pressure")
6. generate_analysis_plot(               # Save visualization
     plot_type="population_timeseries")
```

## Example Output: Population Timeseries

From a tumor growth simulation with oxygen-dependent necrosis (72h, 200 initial cells):

| Time (min) | Total | Live | Dead | tumor |
|---|---|---|---|---|
| 0 | 200 | 200 | 0 | 200 |
| 720 | 312 | 305 | 7 | 312 |
| 1440 | 498 | 471 | 27 | 498 |
| 2160 | 753 | 684 | 69 | 753 |
| 2880 | 1089 | 937 | 152 | 1089 |
| 3600 | 1502 | 1198 | 304 | 1502 |
| 4320 | 1945 | 1423 | 522 | 1945 |

Growth analysis: tumor 200 -> 1945, growth rate = 0.0327/hr, doubling time ~ 21.2 hr
