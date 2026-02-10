# PhysiBoSS Integration Guide

PhysiBoSS integrates MaBoSS boolean network models into PhysiCell cell types, enabling intracellular signaling to drive cell phenotype decisions.

## Overview

PhysiBoSS connects:
- **PhysiCell signals** (oxygen, pressure, contact) → **Boolean network input nodes**
- **Boolean network output nodes** → **PhysiCell behaviors** (apoptosis, migration, proliferation)

This allows intracellular signaling networks to dynamically control cell behavior based on the microenvironment.

## Prerequisites

1. MaBoSS boolean network files:
   - `.bnd` file — Boolean Network Definition (node logic)
   - `.cfg` file — Configuration (initial states, rates, parameters)
2. Cell types must be added first via `add_single_cell_type()`
3. PhysiBoSS module must be available (check tool availability)

## Workflow

```
add_single_cell_type("cancer")
  → add_physiboss_model(cell_type, bnd_file, cfg_file)
  → configure_physiboss_settings(cell_type, ...)
  → add_physiboss_input_link(cell_type, signal, node, ...)     [repeat]
  → add_physiboss_output_link(cell_type, node, behavior, ...)  [repeat]
  → apply_physiboss_mutation(cell_type, node, value)           [optional]
```

## Step 1: Add Boolean Network Model

```python
add_physiboss_model(
    cell_type="cancer",
    bnd_file="/path/to/model.bnd",
    cfg_file="/path/to/model.cfg"
)
```

This attaches the MaBoSS network to the specified cell type. Each cell will run its own instance of the boolean network.

## Step 2: Configure Settings

```python
configure_physiboss_settings(
    cell_type="cancer",
    intracellular_dt=6.0,      # PhysiBoSS timestep (min) — how often the network updates
    time_stochasticity=0,      # Randomness in update timing (0 = deterministic timing)
    scaling=1.0,               # Scaling factor for intracellular dynamics
    start_time=0.0,            # When to start the intracellular model (min)
    inheritance_global=False   # Whether daughter cells inherit parent's network state
)
```

### Timing considerations

- `intracellular_dt` should be ≥ the PhysiCell mechanics timestep (typically 6 min)
- Smaller dt = more frequent network updates but slower simulation
- `time_stochasticity > 0` adds randomness: each cell updates at slightly different times

### Inheritance

- `inheritance_global=False`: daughter cells start with random/default initial conditions
- `inheritance_global=True`: daughter cells inherit the parent's boolean network state at division

## Step 3: Input Links (Environment → Network)

Connect PhysiCell signals to boolean network nodes:

```python
add_physiboss_input_link(
    cell_type="cancer",
    physicell_signal="oxygen",      # PhysiCell signal name
    boolean_node="Hypoxia",         # MaBoSS node name
    action="activation",            # "activation" or "inhibition"
    threshold=5.0,                  # signal value at which node activates
    smoothing=0                     # 0 = sharp threshold, >0 = gradual
)
```

### How input links work

- **activation**: When `signal > threshold`, the boolean node is pushed toward ON (1)
- **inhibition**: When `signal > threshold`, the boolean node is pushed toward OFF (0)
- **threshold**: The signal level that triggers the link
- **smoothing**: 0 = binary switch at threshold; higher values = smoother transition

### Common input links

| PhysiCell Signal | Boolean Node | Action | Threshold | Notes |
|-----------------|-------------|--------|:---------:|-------|
| oxygen | Hypoxia | inhibition | 10 | Low O2 → Hypoxia ON |
| pressure | Stress | activation | 0.5 | High pressure → Stress ON |
| contact with immune | TNF | activation | 0.5 | Immune contact → TNF signaling |
| drug | Drug_effect | activation | 0.1 | Drug present → drug pathway ON |

Use `list_all_available_signals()` to see all available PhysiCell signals.

## Step 4: Output Links (Network → Phenotype)

Connect boolean network nodes to PhysiCell behaviors:

```python
add_physiboss_output_link(
    cell_type="cancer",
    boolean_node="Apoptosis",              # MaBoSS node name
    physicell_behavior="apoptosis",         # PhysiCell behavior name
    action="activation",                    # "activation" or "inhibition"
    value=1000000,                          # behavior value when node is ON
    base_value=0,                           # behavior value when node is OFF
    smoothing=0                             # 0 = binary, >0 = gradual
)
```

### How output links work

- When the boolean node is **ON** (1): behavior = `value`
- When the boolean node is **OFF** (0): behavior = `base_value`
- **action = "activation"**: node ON → behavior set to `value`
- **action = "inhibition"**: node ON → behavior set to `base_value` (inverted)

### Common output links

| Boolean Node | PhysiCell Behavior | Action | Value | Base Value | Notes |
|-------------|-------------------|--------|:-----:|:----------:|-------|
| Apoptosis | apoptosis | activation | 1e6 | 0 | Node ON → rapid apoptosis |
| Migration | migration speed | activation | 1.0 | 0 | Node ON → cells migrate |
| Proliferation | cycle entry | activation | 0.001 | 0 | Node ON → cells divide |
| Quiescence | cycle entry | inhibition | 0 | 0.001 | Node ON → cells stop dividing |

Use `list_all_available_behaviors()` to see all available PhysiCell behaviors.

## Step 5: Mutations (Optional)

Fix boolean nodes to specific values to simulate genetic mutations:

```python
# Simulate p53 loss-of-function
apply_physiboss_mutation(
    cell_type="cancer",
    node_name="p53",
    fixed_value=0          # 0 = always OFF, 1 = always ON
)

# Simulate constitutive Ras activation
apply_physiboss_mutation(
    cell_type="cancer",
    node_name="Ras",
    fixed_value=1          # always ON
)
```

### Mutation strategies

| Mutation Type | Fixed Value | Biological Meaning |
|--------------|:----------:|-------------------|
| Loss-of-function | 0 | Gene/protein is knocked out |
| Gain-of-function | 1 | Gene/protein is constitutively active |

Multiple mutations can be applied to the same cell type.

## MaBoSS Context

If you've previously analyzed a MaBoSS model (e.g., in a separate MaBoSS MCP session), you can store that context:

```python
set_maboss_context(
    model_name="tumor_signaling",
    bnd_file_path="/path/to/model.bnd",
    cfg_file_path="/path/to/model.cfg",
    target_cell_type="cancer",
    available_nodes="Apoptosis,Migration,Proliferation,Hypoxia,p53,Ras",
    output_nodes="Apoptosis,Migration,Proliferation",
    simulation_results="Steady state: 40% Apoptosis, 30% Migration, 30% Proliferation",
    biological_context="Tumor cell fate decisions under hypoxia"
)
```

Retrieve it later:
```python
get_maboss_context()
```

## Tips

1. **Check node names carefully**: Boolean node names in input/output links must exactly match the .bnd file
2. **Use high apoptosis values**: For death-triggering output links, use `value=1e6` to ensure immediate apoptosis when the node activates
3. **Set base_value for output links**: If `base_value=0` and the node is OFF, the behavior is completely suppressed. Make sure this is intentional.
4. **Test without links first**: Verify the PhysiCell simulation works before adding PhysiBoSS complexity
5. **Monitor network states**: Check simulation output for boolean network state distributions
