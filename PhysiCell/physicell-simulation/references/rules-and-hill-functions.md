# Rules and Hill Functions — Deep Dive

## The PhysiCell Rules System

PhysiCell's signal-behavior rules define how cells respond to their environment. Rules are stored in a CSV file (`cell_rules.csv`) with this format (no header):

```
cell_type, signal, direction, behavior, base_value, half_max, hill_power, apply_to_dead
```

Example row:
```
tumor,oxygen,decreases,necrosis,0,3.75,8,0
```

## Hill Function Mathematics

### The Core Equation

For `direction = "increases"`:

```
                                      signal^n
behavior = base + (sat - base) × ─────────────────
                                  half_max^n + signal^n
```

For `direction = "decreases"`:

```
                                            signal^n
behavior = sat + (base - sat) × ─────────────────────
                                  half_max^n + signal^n
```

Where:
- **base** = `base_value` from the CSV (= `min_signal` parameter in the tool)
- **sat** = saturation value = **XML default rate** for that behavior
- **half_max** = signal concentration at 50% effect
- **n** = `hill_power` = Hill coefficient (steepness)

### Parameter Interpretation

#### `base_value` (CSV column 5, tool parameter: `min_signal`)

The behavior value at the signal extreme **opposite** to the direction:
- For "increases": base_value is the behavior when signal is LOW (near zero)
- For "decreases": base_value is the behavior when signal is HIGH (saturated)

Default in tool: 0. This often makes sense (e.g., no necrosis when oxygen is abundant), but check that the saturation value is nonzero.

#### Saturation value (NOT in CSV — comes from XML)

The behavior value at the signal extreme **aligned** with the direction:
- For "increases": saturation is the behavior when signal is HIGH
- For "decreases": saturation is the behavior when signal is LOW

**This value is the XML default rate for the target behavior.** It is set by:
- `configure_cell_parameters()` — sets apoptosis_rate, necrosis_rate, motility_speed
- Cell cycle model defaults — sets cycle entry / transition rates
- Template defaults — applied when `add_single_cell_type()` is called

#### `half_max` (CSV column 6)

The signal concentration at which the behavior is exactly halfway between base and saturation.

```
At signal = half_max:  behavior = (base + sat) / 2
```

Choosing half_max:
- For oxygen rules: typical half_max = 3-10 mmHg (hypoxia threshold)
- For pressure rules: typical half_max = 0.5-2.0 (dimensionless)
- For chemokine rules: depends on concentration scale

#### `hill_power` (CSV column 7)

Controls the steepness of the transition:

```
n = 1:  Gradual, Michaelis-Menten-like response
n = 2:  Moderate cooperativity
n = 4:  Fairly steep (DEFAULT)
n = 8:  Near-switch behavior (sharp threshold)
n = 16: Almost binary on/off
```

ASCII visualization (behavior vs signal, "increases" direction):

```
sat ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ x x x
                              x x x
                          x x       n=1 (gradual)
                        x
                      x
                    x
                  x
                x
              x
base ─ x x x ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      0         half_max              max signal

sat ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ x x x x x
                            x x x
                           x        n=8 (steep)
                          x
                         x
                        x
                       x
base ─ x x x x x x x x ─ ─ ─ ─ ─ ─ ─ ─ ─
      0         half_max              max signal
```

## The "From 0 Towards 0" Bug — Deep Dive

### What happens

When both `base_value = 0` and the XML default rate (saturation) = 0:

```
behavior = 0 + (0 - 0) × Hill(signal) = 0
```

The rule produces zero for ALL signal values. It silently does nothing.

### Where XML defaults come from

The saturation value for each behavior is determined by the cell type's XML configuration:

| Behavior | XML Source | How it's set |
|---|---|---|
| cycle entry | `<cycle>` model's transition rate | Cell cycle model template default |
| exit from cycle phase N | `<cycle>` transition_rates | Explicitly via library or template |
| apoptosis | `<death><model>` rate | `configure_cell_parameters(apoptosis_rate=X)` |
| necrosis | `<death><model>` rate | `configure_cell_parameters(necrosis_rate=X)` |
| migration speed | `<motility><speed>` | `configure_cell_parameters(motility_speed=X)` |
| persistence time | `<motility><persistence_time>` | `configure_cell_parameters(persistence_time=X)` |
| secretion of X | `<secretion><substrate><secretion_rate>` | `set_substrate_interaction(secretion_rate=X)` |
| uptake of X | `<secretion><substrate><uptake_rate>` | `set_substrate_interaction(uptake_rate=X)` |
| chemotactic response | `<motility><chemotaxis><chemotactic_sensitivity>` | Advanced motility settings |

### Common "from 0 towards 0" scenarios

1. **Transition rates**: Most cycle models have transition rates = 0 by default for unused phases. Adding a rule that targets these transitions does nothing.

2. **Custom behaviors**: Any behavior not explicitly initialized has XML default = 0.

3. **Secretion/uptake as rule targets**: If `set_substrate_interaction()` wasn't called first, the XML default is 0.

### How to verify rules work

After running a simulation, check `detailed_rules.txt` in the output directory. It shows the actual "from X towards Y" values PhysiCell computed for each rule. If you see "from 0.0000 towards 0.0000", the rule is doing nothing.

```
# In detailed_rules.txt:
tumor: oxygen decreases necrosis from 0.0000 towards 0.0028
#                                     ^^^^^^           ^^^^^^
#                                     base_value       saturation (from XML)
```

If saturation = 0.0000, the rule is broken.

### Fix procedure

1. Identify which behavior is the target
2. Set a nonzero value for that behavior BEFORE adding the rule:
   - For death rates: `configure_cell_parameters(apoptosis_rate=X, necrosis_rate=Y)`
   - For motility: `configure_cell_parameters(motility_speed=X)`
   - For secretion: `set_substrate_interaction(secretion_rate=X)`
   - For cycle entry: ensure the cycle model has nonzero transition rates (Ki67_basic and live models do by default)
3. Then add the rule — it will interpolate between base_value and the (now nonzero) XML default

## Worked Examples

### Example 1: Oxygen-dependent necrosis

**Goal**: Cells die (necrosis) when oxygen is low.

```python
# Step 1: Ensure necrosis rate is nonzero in XML
configure_cell_parameters(cell_type="tumor", necrosis_rate=0.00277)
# → XML default necrosis rate = 0.00277 (saturation value)

# Step 2: Add rule — oxygen DECREASES necrosis
# (high oxygen → less necrosis, low oxygen → more necrosis up to 0.00277)
add_single_cell_rule(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis",
    min_signal=0,         # base_value: necrosis rate at HIGH oxygen = 0
    half_max=3.75,        # oxygen level (mmHg) at 50% necrosis
    hill_power=8          # steep response (threshold-like)
)
# Result: necrosis rate goes from 0 (abundant O2) to 0.00277 (no O2) ✓
```

### Example 2: Pressure-dependent proliferation

**Goal**: High mechanical pressure stops cell division.

```python
# Step 1: cycle entry rate comes from Ki67_basic model (~0.00072 by default) ✓
# No extra configuration needed

# Step 2: Add rule — pressure DECREASES cycle entry
add_single_cell_rule(
    cell_type="tumor",
    signal="pressure",
    direction="decreases",
    behavior="cycle entry",
    min_signal=0,         # base_value: cycle rate at HIGH pressure = 0 (stopped)
    half_max=1.0,         # pressure level at 50% growth inhibition
    hill_power=4          # moderate steepness
)
# Result: proliferation goes from 0.00072 (no pressure) to 0 (high pressure) ✓
```

### Example 3: Chemokine-directed migration

**Goal**: Cells migrate faster toward chemokine source.

```python
# Step 1: Set base migration speed
configure_cell_parameters(cell_type="immune", motility_speed=0.5)
# → XML default migration speed = 0.5

# Step 2: Add rule — chemokine INCREASES migration speed
add_single_cell_rule(
    cell_type="immune",
    signal="chemokine",
    direction="increases",
    behavior="migration speed",
    min_signal=0.1,       # base_value: speed when no chemokine = 0.1
    half_max=0.5,         # chemokine level at 50% effect
    hill_power=4
)
# Result: speed goes from 0.1 (no chemokine) to 0.5 (high chemokine) ✓
```

### Example 4: BROKEN — "from 0 towards 0"

**Goal**: Drug increases apoptosis. **BUG**: forgot to set apoptosis rate.

```python
# Step 1: MISSING — apoptosis_rate left at default
configure_cell_parameters(cell_type="tumor", motility_speed=0.5)
# → apoptosis_rate = 0.0001 (template default — this is actually nonzero!)
# But if template default were 0...

# Step 2: Add rule
add_single_cell_rule(
    cell_type="tumor",
    signal="drug",
    direction="increases",
    behavior="apoptosis",
    min_signal=0,         # base_value = 0
    half_max=0.5,
    hill_power=4
)
# IF XML default apoptosis = 0:
#   behavior = 0 + (0 - 0) × Hill = 0  ← BROKEN!
# IF XML default apoptosis = 0.0001:
#   behavior = 0 + (0.0001 - 0) × Hill  ← works, but effect is tiny
```

**Fix**: Set a meaningful apoptosis rate first:
```python
configure_cell_parameters(cell_type="tumor", apoptosis_rate=0.01)
```
