# Troubleshooting Guide

Diagnose PhysiCell simulation problems by symptom.

## No Output Files

**Symptom**: Simulation appears to run but no output files are generated.

**Possible causes**:
1. **Compilation failed**: Check `compile_physicell_project()` output for errors
2. **Simulation crashed immediately**: Check `get_simulation_status()` for error messages
3. **Output directory wrong**: PhysiCell writes to `./output/` relative to the project directory
4. **No initial cells**: If `place_initial_cells()` wasn't called AND `number_of_cells=0` in XML, there are no cells to simulate

**Fix**:
- Verify compilation succeeded
- Check simulation status
- Ensure cells were placed or `number_of_cells > 0` in the XML
- Use `get_simulation_output_files()` to check what was actually generated

## Cells Don't Move

**Symptom**: Cells remain stationary throughout the simulation.

**Possible causes**:
1. **Motility speed = 0**: Check `configure_cell_parameters(motility_speed=...)` was called
2. **Persistence time = 0**: Cells re-randomize direction every step and go nowhere effectively
3. **No chemotactic signal**: If relying on chemotaxis, ensure substrate exists and is secreted
4. **Motility not enabled**: Some cell type templates disable motility by default

**Fix**:
```python
configure_cell_parameters(cell_type="...", motility_speed=0.5, persistence_time=5.0)
```

## Cells Don't Die

**Symptom**: Cell population only grows, no death observed.

**Possible causes**:
1. **Death rates = 0**: Check `apoptosis_rate` and `necrosis_rate` in `configure_cell_parameters()`
2. **Rules have no effect**: "From 0 towards 0" bug (see below)
3. **Substrate conditions don't trigger death**: Oxygen stays high, no hypoxia
4. **Dirichlet boundaries too generous**: Constant oxygen supply prevents necrosis

**Fix**:
- Set nonzero death rates: `configure_cell_parameters(apoptosis_rate=5.31667e-5, necrosis_rate=0.00277)`
- Check rules in `detailed_rules.txt` (see below)
- Consider reducing domain size or Dirichlet values to create resource depletion

## Rules Have No Effect (The "From 0 Towards 0" Bug)

**Symptom**: Rules are defined but cells don't respond to signals. The simulation runs as if no rules were set.

**Diagnosis**: Check `detailed_rules.txt` in the output directory for lines like:
```
tumor: oxygen decreases necrosis from 0.0000 towards 0.0000
```
The "from 0.0000 towards 0.0000" indicates both base_value and saturation are zero.

**Root cause**: The saturation value comes from the XML default rate for the target behavior. If that default is 0 (because it wasn't configured), the rule interpolates between 0 and 0.

**Fix**:
1. Identify the target behavior
2. Set a nonzero XML default BEFORE adding the rule:
   - For death rates: `configure_cell_parameters(apoptosis_rate=X, necrosis_rate=Y)`
   - For motility: `configure_cell_parameters(motility_speed=X)`
   - For secretion: `set_substrate_interaction(secretion_rate=X)`
3. Re-export XML and rules CSV
4. Re-create and re-compile the project

See `rules-and-hill-functions.md` for the full deep dive.

## Simulation Hangs or Runs Extremely Slowly

**Symptom**: Simulation doesn't produce output for a very long time.

**Possible causes**:
1. **Domain too large**: Large domains with many voxels consume memory and CPU
2. **Too many cells**: Exponential growth without death can create millions of cells
3. **3D simulation**: 3D is orders of magnitude slower than 2D
4. **Small dx**: dx < 10 creates very fine meshes

**Fix**:
- Start with small domains (500-1000 μm) for testing
- Use 2D (domain_z = 20, dx = 20) unless 3D is specifically needed
- Add death mechanisms to prevent unlimited growth
- Check `get_simulation_status()` for progress

## cells.csv Not Found

**Symptom**: Project creation succeeds but simulation has no initial cells.

**Possible causes**:
1. **Forgot `export_cells_csv()`**: Must be called after `place_initial_cells()` but before `create_physicell_project()`
2. **Called in wrong order**: CSV must be exported before project creation copies files

**Fix**:
```
place_initial_cells(...)
export_xml_configuration()     # exports XML
export_cell_rules_csv()        # exports rules
export_cells_csv()             # exports cells.csv ← don't forget this!
create_physicell_project(...)  # copies all files to project
```

## Compilation Fails

**Symptom**: `compile_physicell_project()` returns errors.

**Possible causes**:
1. **PhysiCell not installed**: The base PhysiCell source must be available
2. **Missing dependencies**: g++, make, or other build tools not installed
3. **Makefile paths wrong**: Project template may have incorrect paths

**Fix**:
- Verify PhysiCell is installed and compilable independently
- Check that g++ and make are available: `g++ --version`, `make --version`
- Look at the compilation error output for specific issues

## Substrate Doesn't Diffuse Properly

**Symptom**: Substrate concentration stays uniform or doesn't change.

**Possible causes**:
1. **Diffusion coefficient too high**: Substrate equilibrates instantly
2. **No sources or sinks**: No cells consuming/producing the substrate
3. **Dirichlet boundaries overriding everything**: Boundaries reset concentration every step
4. **Decay rate too high or too low**: Substrate disappears too fast or persists too long

**Fix**:
- Check substrate parameters match the reference values in `parameter-reference.md`
- Ensure `set_substrate_interaction()` was called for at least one cell type
- For localized gradients, disable Dirichlet boundaries or set them only on specific sides

## Cells Cluster Unrealistically

**Symptom**: Cells form tight unrealistic clusters or separate completely.

**Possible causes**:
1. **Cell-cell adhesion too strong**: Cells stick together permanently
2. **Cell-cell repulsion too weak**: Cells overlap unrealistically
3. **Volume too large**: Cells are bigger than expected, causing crowding
4. **Domain too small**: Cells fill the space and can't spread

**Fix**:
- Adjust cell volume: `configure_cell_parameters(volume_total=2500)`
- Ensure domain is large enough for the cell population
- Check mechanics parameters (adhesion/repulsion are set to reasonable defaults)

## GIF Generation Fails

**Symptom**: `generate_simulation_gif()` returns an error.

**Possible causes**:
1. **No SVG files**: Simulation didn't produce SVG snapshots
2. **ImageMagick not installed**: GIF generation requires ImageMagick's `convert`
3. **Simulation didn't complete**: Check `get_simulation_status()` first

**Fix**:
- Ensure simulation completed: `get_simulation_status(simulation_id)`
- Check output files: `get_simulation_output_files(simulation_id, file_type="svg")`
- Install ImageMagick if needed: `brew install imagemagick` (macOS)

## Analysis Tools Return Errors

**Symptom**: `get_simulation_analysis_overview()`, `get_cell_data()`, or other analysis tools fail.

**Possible causes**:
1. **pcdl not installed**: The `pcdl` package (PhysiCell Data Loader) is required for all analysis tools
2. **Simulation not complete**: Analysis tools need `.mat` output files which are written during simulation
3. **No output files**: Simulation crashed before writing any output
4. **Wrong simulation_id or output_folder**: Path doesn't point to valid PhysiCell output

**Fix**:
- Install pcdl: `pip install pcdl`
- Check simulation status first: `get_simulation_status(simulation_id)`
- Verify output files exist: `get_simulation_output_files(simulation_id)`
- For `get_cell_data()`: use `columns` parameter to select specific fields, `cell_type` to filter by type
- For `generate_analysis_plot()`: valid plot types are `population_timeseries`, `cell_scatter`, `substrate_contour`

## Rules Reference Nonexistent Substrate or Cell Type

**Symptom**: Rule is added without error but has no effect because the signal or behavior references something that doesn't exist in the XML.

**Fix**:
- Always add substrates BEFORE rules
- Always add cell types BEFORE rules
- Use `list_all_available_signals()` and `list_all_available_behaviors()` to verify available options
- Check spelling matches exactly (case-sensitive)
