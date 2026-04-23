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

### The Core Equation (from PhysiCell source, authoritative)

PhysiCell uses **one** Hill formula regardless of direction. From `PhysiCell/core/PhysiCell_rules.cpp:348-354`:

```
Hill(s, half, n) = s^n / (s^n + half^n)    # 0 at s=0, 1 at s=∞

behavior(signal) = XML_default + (saturation_value - XML_default) × Hill(signal, half_max, hill_power)
```

The `direction` keyword does **not** change this formula. It just tells the CSV parser where to read the endpoints. From `PhysiCell_rules.cpp:1762-1773`:

| direction | XML default maps to | saturation_value (CSV col 5) maps to |
|---|---|---|
| `increases` | min of the Hill curve (low-signal end) | max of the Hill curve (high-signal end) |
| `decreases` | max of the Hill curve (low-signal end) | min of the Hill curve (high-signal end) |

In **both** cases:

| | At signal ≈ 0 | At signal → ∞ |
|---|---|---|
| **behavior =** | XML_default | saturation_value |

The only difference: for `decreases` you expect `XML_default > saturation_value` (behavior falls as signal rises); for `increases` you expect `XML_default < saturation_value` (behavior rises as signal rises). Nothing stops you from writing inconsistent values — PhysiCell won't complain, you'll just get a rule that doesn't do what its direction word implies.

### Mental model (how to avoid getting it wrong)

Ignore the `direction` keyword when you're reasoning about what a rule does. Ask two questions:

1. **What value do I want the behavior to have when the signal is ABSENT (signal = 0)?** → put it in **XML default** (via `configure_cell_parameters`, `set_cycle_transition_rate`, `set_cell_transformation_rate`, etc.).
2. **What value do I want the behavior to have when the signal is SATURATING (signal → ∞)?** → put it in **saturation_value** (the CSV col 5 / `add_single_cell_rule(saturation_value=...)` argument).

Then pick the direction keyword that matches:

- `XML_default < saturation_value` → `increases`
- `XML_default > saturation_value` → `decreases`

This order of operations prevents the most common mistake (see "Common wrong encoding" below).

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

This parameter is **required** (no default). Always choose a biologically meaningful value — never use an arbitrary number like 1.0 without justification.

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

### Example 5: Hypoxia induces a phenotype switch (XML default carries the firing rate)

**Goal**: Low oxygen transforms tumor cells to a motile post-hypoxic phenotype. At low O₂, transform at 0.001/min; at normoxia, transform rate → 0.

This is the opposite pattern from Examples 1–3: the *active* rate lives in the XML default, and `saturation_value` is zero (the "off" value at high signal).

```python
# Step 1: Put the firing rate in the XML default (the transformation_rate for tumor → motile_tumor)
set_cell_transformation_rate(
    source_cell_type="tumor",
    target_cell_type="motile tumor",
    rate=0.001,
)
# → XML default transformation_rate = 0.001 /min (this is the hypoxic rate)

# Step 2: Add rule — oxygen DECREASES transformation
# XML_default > saturation_value, so direction = "decreases"
add_single_cell_rule(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="transform to motile tumor",
    saturation_value=0,   # normoxic rate = 0 (behavior suppressed at high O₂)
    half_max=10.0,        # mmHg — HIF-stabilization threshold
    hill_power=4,
)
# At low O₂:  behavior = 0.001     (XML default — the hypoxic transform rate fires)
# At high O₂: behavior → 0         (saturation_value — no transform at normoxia)  ✓
```

Same pattern applies to hypoxia-induced necrosis, hypoxia-induced apoptosis, hypoxia-induced secretion, etc. — whenever the signal drives the behavior UP from zero, the behavior's **firing value goes in the XML default** and `direction = decreases`.

### Example 6: BROKEN — hypoxia rule with the firing rate in the wrong place

**Goal**: Same as Example 5. **BUG**: put the rate in `saturation_value` instead of XML default.

This is the single most common rule bug outside of "from 0 towards 0". It looks biologically correct, but encodes the **opposite** behavior.

```python
# Step 1: XML default left at 0 (transformation_rate unset)
# Step 2: Put the rate in saturation_value
add_single_cell_rule(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",       # user thought: "oxygen decreases as transformation increases → decreases"
    behavior="transform to motile tumor",
    saturation_value=0.001,      # "hypoxic rate" — BUT this is the high-O₂ asymptote, not the low-O₂ one
    half_max=10.0,
    hill_power=4,
)
# At low O₂:  behavior = 0        (XML default = 0 — no transformation at hypoxia!)
# At high O₂: behavior = 0.001    (saturation_value — transform fires at normoxia)  ✗
```

Calibrator runs a UQ sweep over `saturation_value`, sees all particles produce zero motile cells at hypoxia, and concludes "the calibration is broken." In fact the rule mechanism is just encoded **backwards**: the user confused "saturation" in the word sense with "the biologically active rate," but `saturation_value` always means "behavior when signal saturates" — i.e. the **high-signal endpoint**.

**Tell-tale signs of this mistake:**

- ABC posterior for the rate hugs the lower prior bound — sampler is trying to drive a normoxic rate to zero.
- `detailed_rules.txt` shows `from 0 towards X` for a rule the user described as "hypoxia-induced."
- The intended hypoxic effect never fires; the mirror effect fires at normoxia.

**Fix**: Swap the rate from `saturation_value` into XML default (via `set_cell_transformation_rate` / `configure_cell_parameters` / etc.), and set `saturation_value = 0`. Keep `direction = "decreases"`. See Example 5.

## Where the calibrated / UQ knob lives

When doing UQ or ABC calibration on a rule, you have to decide which parameter to sweep. The knob always lives at the **firing end** of the rule — never the off end:

| Rule type | Firing end | Put the UQ knob in |
|---|---|---|
| Low-signal-activated (`direction=decreases`, e.g., hypoxia-induced) | XML default | `configure_cell_parameters` / `set_cycle_transition_rate` / `set_cell_transformation_rate` value — substitute via XML xpath |
| High-signal-activated (`direction=increases`, e.g., chemokine-induced) | saturation_value | `add_single_cell_rule(saturation_value=...)` — substitute via CSV col 5 |

Concrete: if calibrating a hypoxia-induced transform rate over `[1e-4, 1e-2]`, the UQ parameter should be the **XML `<transformation_rate>`** not the CSV `saturation_value`. Sweeping the wrong one gives a rule whose firing magnitude does not respond to the prior.

**Known infrastructure pitfall** (uq_physicell as of 2026-04): the XPath setter in `uq_physicell.pc_model._set_xml_element_value` corrupts terminal selectors of the form `[@name='X']` by writing to the `name` attribute instead of the element's text content. Verify with a dry-run substitution + XML diff before launching a calibration run.

## Source reference

The formulas above are taken from `PhysiCell/core/PhysiCell_rules.cpp` on the upstream repo:

- `PhysiCell_rules.cpp:115-131` — `multivariate_Hill_response_function` (the Hill term; returns 0 for empty signal vector).
- `PhysiCell_rules.cpp:348-354` — `Hypothesis_Rule::evaluate` (the core `output = base + (max-base)*HU + (min-U)*DU` formula).
- `PhysiCell_rules.cpp:1762-1773` — `parse_csv_rule_v3` (the v2/v3 CSV → struct mapping; determines which of `min_value` / `max_value` takes CSV col 5 vs XML default, per direction).
- `PhysiCell_rules.cpp:1630-1639` — `parse_csv_rule_v1` (same mapping in the v1 CSV parser).

If a rule is behaving unexpectedly, read the actual `detailed_rules.txt` written by PhysiCell in the run's output directory: it records "from X towards Y" values that correspond directly to `(XML_default, saturation_value)` for each rule. A mismatch vs your intent in that file is definitive evidence of an encoding error.
