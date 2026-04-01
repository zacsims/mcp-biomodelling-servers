# Rules and Hill Functions — Deep Dive

## The PhysiCell Rules System

PhysiCell's signal-behavior rules define how cells respond to their environment. Rules are stored in a CSV file (`cell_rules.csv`) with this format (no header):

```
cell_type, signal, direction, behavior, saturation_value, half_max, hill_power, apply_to_dead
```

Example row:
```
tumor,oxygen,decreases,necrosis,0,3.75,8,0
```

## Hill Function Mathematics

### The Core Equation

The behavior interpolates between the **XML default** (behavior at zero signal) and the **saturation_value** (behavior at maximum signal effect):

```
                                                signal^n
behavior = XML_default + (saturation_value - XML_default) × ─────────────────
                                                half_max^n + signal^n
```

For `direction = "decreases"`, the Hill term is inverted:

```
                                                half_max^n
behavior = saturation_value + (XML_default - saturation_value) × ─────────────────
                                                  half_max^n + signal^n
```

Both formulas produce the **same endpoints**:

| | At signal ≈ 0 | At signal → ∞ |
|---|---|---|
| **behavior =** | XML default | saturation_value |

The direction only affects the **shape** of the curve (sigmoid vs inverted sigmoid).

Where:
- **XML_default** = the behavior's value set via setter tools (e.g., `configure_cell_parameters(necrosis_rate=0.00277)`)
- **saturation_value** = the `saturation_value` parameter in `add_single_cell_rule()` (CSV column 5)
- **half_max** = signal concentration at 50% effect (CSV column 6)
- **n** = `hill_power` = Hill coefficient / steepness (CSV column 7)

### Parameter Interpretation

#### `saturation_value` (CSV column 5, tool parameter: `saturation_value`)

The behavior value when the signal has **maximum** effect:

- For `"decreases"`: set `saturation_value` **lower** than XML default (typically 0) — the behavior is suppressed at high signal
- For `"increases"`: set `saturation_value` **higher** than XML default — the behavior is amplified at high signal

Default in tool: 1.0. **Check that this makes sense for your behavior** — a necrosis rate of 1.0/min would be extremely aggressive.

#### XML default (NOT in CSV — comes from cell type config)

The behavior value when the signal is **absent** (signal ≈ 0). This is set by:
- `configure_cell_parameters()` — sets apoptosis_rate, necrosis_rate, motility_speed, etc.
- `set_cycle_transition_rate()` — sets cycle entry / phase transition rates
- `set_cell_transformation_rate()` — sets phenotype transition rates
- `set_substrate_interaction()` — sets secretion/uptake rates
- `set_advanced_chemotaxis()` — sets chemotactic sensitivity

**If the XML default is 0 AND saturation_value is 0, the rule does nothing** (the "from 0 towards 0" bug).

#### `half_max` (CSV column 6)

The signal concentration at which the behavior is exactly halfway between XML default and saturation_value.

```
At signal = half_max:  behavior = (XML_default + saturation_value) / 2
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
XML ─ x x x ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      0         half_max              max signal

sat ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ x x x x x
                            x x x
                           x        n=8 (steep)
                          x
                         x
                        x
                       x
XML ─ x x x x x x x x ─ ─ ─ ─ ─ ─ ─ ─ ─
      0         half_max              max signal
```

## The "From 0 Towards 0" Bug — Deep Dive

### What happens

When both `saturation_value = 0` and the XML default rate = 0:

```
behavior = 0 + (0 - 0) × Hill(signal) = 0
```

The rule produces zero for ALL signal values. It silently does nothing.

### Where XML defaults come from

| Behavior | XML Source | How it's set |
|---|---|---|
| cycle entry | `<cycle>` model's transition rate | `set_cycle_transition_rate()` |
| exit from cycle phase N | `<cycle>` transition_rates | `set_cycle_transition_rate(from_phase, to_phase)` |
| apoptosis | `<death><model>` rate | `configure_cell_parameters(apoptosis_rate=X)` |
| necrosis | `<death><model>` rate | `configure_cell_parameters(necrosis_rate=X)` |
| migration speed | `<motility><speed>` | `configure_cell_parameters(motility_speed=X)` |
| persistence time | `<motility><persistence_time>` | `configure_cell_parameters(persistence_time=X)` |
| secretion of X | `<secretion><substrate><secretion_rate>` | `set_substrate_interaction(secretion_rate=X)` |
| uptake of X | `<secretion><substrate><uptake_rate>` | `set_substrate_interaction(uptake_rate=X)` |
| chemotactic response | `<motility><chemotaxis><chemotactic_sensitivity>` | `set_advanced_chemotaxis(sensitivity=X)` |
| transition to X | `<cell_transformations>` | `set_cell_transformation_rate(rate=X)` |

### How to verify rules work

After running a simulation, check `detailed_rules.txt` in the output directory. It shows the actual "from X towards Y" values PhysiCell computed for each rule. If you see "from 0.0000 towards 0.0000", the rule is doing nothing.

```
# In detailed_rules.txt:
tumor: oxygen decreases necrosis from 0.0028 towards 0.0000
#                                     ^^^^^^           ^^^^^^
#                                     XML default       saturation_value
```

If both values are 0.0000, the rule is broken.

### Fix procedure

1. Identify which behavior is the target
2. Set a nonzero XML default for that behavior BEFORE adding the rule (see setter table above)
3. Then add the rule — it will interpolate between XML default and saturation_value

## Worked Examples

### Example 1: Oxygen-dependent necrosis

**Goal**: Cells die (necrosis) when oxygen is low.

```python
# Step 1: Ensure necrosis rate is nonzero in XML
configure_cell_parameters(cell_type="tumor", necrosis_rate=0.00277)
# → XML default necrosis rate = 0.00277

# Step 2: Add rule — oxygen DECREASES necrosis
# (high oxygen → less necrosis, low oxygen → more necrosis)
add_single_cell_rule(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis",
    saturation_value=0,   # necrosis → 0 at high O₂
    half_max=3.75,        # oxygen level (mmHg) at 50% effect
    hill_power=8          # steep response (threshold-like)
)
# At low O₂:  necrosis = 0.00277 (XML default)
# At high O₂: necrosis → 0 (saturation_value)  ✓
```

### Example 2: Pressure-dependent proliferation

**Goal**: High mechanical pressure stops cell division.

```python
# Step 1: Set cycle entry rate
set_cycle_transition_rate(cell_type="tumor", rate=0.00072)
# → XML default cycle entry = 0.00072

# Step 2: Add rule — pressure DECREASES cycle entry
add_single_cell_rule(
    cell_type="tumor",
    signal="pressure",
    direction="decreases",
    behavior="cycle entry",
    saturation_value=0,   # proliferation → 0 at high pressure
    half_max=1.0,         # pressure at 50% growth inhibition
    hill_power=4          # moderate steepness
)
# At low pressure:  cycle entry = 0.00072 (XML default)
# At high pressure: cycle entry → 0 (saturation_value)  ✓
```

### Example 3: Chemokine-directed migration

**Goal**: Cells migrate faster toward chemokine source.

```python
# Step 1: Set base migration speed
configure_cell_parameters(cell_type="immune", motility_speed=0.1)
# → XML default migration speed = 0.1

# Step 2: Add rule — chemokine INCREASES migration speed
add_single_cell_rule(
    cell_type="immune",
    signal="chemokine",
    direction="increases",
    behavior="migration speed",
    saturation_value=0.5, # speed → 0.5 at high chemokine
    half_max=0.5,         # chemokine level at 50% effect
    hill_power=4
)
# At low chemokine:  speed = 0.1 (XML default)
# At high chemokine: speed → 0.5 (saturation_value)  ✓
```

### Example 4: BROKEN — "from 0 towards 0"

**Goal**: Drug increases apoptosis. **BUG**: forgot to set apoptosis rate.

```python
# Step 1: MISSING — apoptosis_rate not set, XML default is 0

# Step 2: Add rule
add_single_cell_rule(
    cell_type="tumor",
    signal="drug",
    direction="increases",
    behavior="apoptosis",
    saturation_value=0,   # also 0!
    half_max=0.5,
    hill_power=4
)
# behavior = 0 + (0 - 0) × Hill = 0  ← BROKEN! Rule does nothing.
```

**Fix**: Set a meaningful apoptosis rate first, and use a nonzero saturation_value:
```python
configure_cell_parameters(cell_type="tumor", apoptosis_rate=5.31667e-5)
add_single_cell_rule(
    ..., saturation_value=0.01, ...  # apoptosis increases to 0.01 at high drug
)
```
