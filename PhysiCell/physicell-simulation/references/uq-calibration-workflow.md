# UQ and Calibration Workflow

Uncertainty Quantification (UQ) tools enable sensitivity analysis and parameter calibration for PhysiCell simulations.

## Prerequisites

Before starting UQ analysis, you must have:
1. A fully configured simulation (domain, substrates, cell types, rules)
2. Exported XML and rules CSV
3. Created and compiled a PhysiCell project

Optional dependencies:
- `torch` + `botorch` — required for Bayesian Optimization calibration
- `pyabc` — required for ABC-SMC calibration
- `uq-physicell >= 1.2.4` — core UQ framework

## Workflow Overview

```
setup_uq_analysis
  → get_uq_parameter_suggestions        (auto-detect tunable parameters)
  → define_uq_parameters                (select and bound parameters)
  → define_quantities_of_interest        (what to measure from output)
  → [BRANCH A] run_sensitivity_analysis  (which params matter most)
  → [BRANCH B] provide_experimental_data + run_bayesian_calibration
  → [BRANCH C] provide_experimental_data + run_abc_calibration
  → get_calibration_results / get_sensitivity_results
  → apply_calibrated_parameters          (update model with best-fit)
```

## Step 1: Setup

```python
setup_uq_analysis(
    project_name="my_project",     # auto-detected if omitted
    num_replicates=3,              # simulations per parameter set
    num_workers=4                  # parallel workers
)
```

This detects the compiled project, XML config, and rules CSV from the session.

## Step 2: Parameter Suggestions

```python
get_uq_parameter_suggestions()
```

Returns auto-detected parameters from both XML and rules CSV, with reference values and suggested bounds.

## Step 3: Define Parameters

Two parameter types:

### XML Parameters (cell properties)

```python
define_uq_parameters(parameters=[
    {
        "name": "tumor_apoptosis_rate",
        "type": "xml",
        "xpath": ".//cell_definition[@name='tumor']//death//model[@code='100']//death_rate",
        "ref_value": 5.31667e-5,
        "lower_bound": 1e-6,
        "upper_bound": 1e-3
    }
])
```

### Rules Parameters (Hill function coefficients)

Rule keys use the format: `"cell_type,signal,direction,behavior,field"`

Where `field` is one of:
- `saturation` — behavior value at signal saturation (CSV column 5 equivalent)
- `half_max` — signal level at 50% effect (CSV column 6)
- `hill_power` — Hill coefficient / steepness (CSV column 7)

```python
define_uq_parameters(parameters=[
    {
        "name": "tumor_oxygen_necrosis_halfmax",
        "type": "rules",
        "rule_key": "tumor,oxygen,decreases,necrosis,half_max",
        "ref_value": 3.75,
        "lower_bound": 1.0,
        "upper_bound": 10.0
    },
    {
        "name": "tumor_oxygen_necrosis_hillpower",
        "type": "rules",
        "rule_key": "tumor,oxygen,decreases,necrosis,hill_power",
        "ref_value": 8,
        "lower_bound": 1,
        "upper_bound": 16
    }
])
```

## Step 4: Quantities of Interest

Define what to measure from simulation outputs:

```python
define_quantities_of_interest(qois=[
    {"name": "live_tumor_cells", "function": "cell_count:tumor"},
    {"name": "dead_cells", "function": "dead_cells"},
    {"name": "total_live", "function": "live_cells"}
])
```

Built-in functions:
- `live_cells` — total live cell count
- `dead_cells` — total dead cell count
- `cell_count:<type>` — count of specific cell type (e.g., `cell_count:tumor`)

Custom lambda expressions on the cell DataFrame are also supported.

## Sensitivity Analysis

### Methods

| Method | Description | Num Samples | Best for |
|--------|-------------|:-----------:|----------|
| `Sobol` | Variance-based global SA | 64-256 | Comprehensive, gold standard |
| `LHS` | Latin Hypercube Sampling | 32-128 | Fast screening |
| `OAT` | One-at-a-Time | 10-50 | Quick local SA |
| `Fast` | Fourier Amplitude SA | 64-256 | Alternative to Sobol |

### Running SA

```python
run_sensitivity_analysis(
    method="Sobol",
    num_samples=64,            # higher = more accurate, slower
    num_workers=4,
    parallel_method="inter-process"
)
```

This runs in the background. Check progress:

```python
get_sensitivity_results(run_id="...")  # returns status or results
```

### Interpreting SA Results

Results include:
- **S1 (first-order indices)**: Direct effect of each parameter
- **ST (total-order indices)**: Direct + interaction effects
- **Parameter rankings**: Which parameters have the most influence

Parameters with high ST and low S1 have significant interaction effects.

## Calibration: Bayesian Optimization

For finding best-fit parameters given experimental data.

### Providing Experimental Data

```python
provide_experimental_data(
    csv_path="/path/to/data.csv",
    column_mapping={
        "live_tumor_cells": "Tumor Count",   # QoI name → CSV column
        "dead_cells": "Dead Count"
    },
    time_column="time"
)
```

CSV format: time column + one column per QoI.

### Running BO

```python
run_bayesian_calibration(
    num_initial_samples=10,    # random exploration
    num_iterations=50,         # optimization iterations
    max_workers=4,
    distance_metric="sum_squared_differences"
)
```

Distance metrics: `sum_squared_differences` (default), `manhattan`, `chebyshev`

### Retrieving Results

```python
get_calibration_status()   # check if done
get_calibration_results()  # best-fit parameters + Pareto front
```

## Calibration: ABC-SMC

Approximate Bayesian Computation with Sequential Monte Carlo. Returns full posterior distributions (uncertainty estimates), not just point estimates.

```python
run_abc_calibration(
    max_populations=8,
    max_simulations=500,
    min_population_size=30,
    max_population_size=100,
    num_workers=4,
    prior_bounds={                    # optional, uses parameter bounds if omitted
        "param_name": {"lower": 0.1, "upper": 2.0}
    },
    fixed_params={                    # optional, fix some parameters
        "other_param": 0.5
    }
)
```

## Applying Results

After calibration completes:

```python
apply_calibrated_parameters()   # auto-apply best-fit
# OR
apply_calibrated_parameters(parameter_overrides={"param": value})  # manual
```

This updates both XML and rules parameters in the session. Then re-export and re-compile to run a validation simulation.

## Full UQ Summary

```python
get_uq_summary()      # comprehensive status of all UQ work
list_uq_runs()        # all SA and calibration runs
```

## Tips

1. **Start with SA**: Run sensitivity analysis first to identify which parameters matter. Only calibrate the important ones.
2. **Use few replicates for SA** (3 is enough), more for calibration.
3. **Keep parameter bounds reasonable**: Unrealistically wide bounds make optimization harder.
4. **Check convergence**: ABC-SMC should show decreasing epsilon across populations.
5. **Validate results**: After calibration, run the model with best-fit parameters and compare to data visually.
