# PhysiCell MCP Server
## Model Context Protocol Integration for Multiscale Biological Simulation

This is a **Model Context Protocol (MCP) server** that exposes PhysiCell multicellular simulation capabilities to Large Language Models (LLMs). It enables LLMs to construct sophisticated physics-based tissue simulations with integrated Boolean gene networks through natural language interactions, and to **calibrate and validate models** against experimental data using uncertainty quantification.

### What is an MCP Server?

Model Context Protocol (MCP) is a standardized way to connect LLMs with external tools and data sources. This server:

- **Exposes PhysiCell functionality** as callable tools for LLMs
- **Enables natural language simulation construction** from biological scenarios
- **Provides PhysiBoSS integration** for multiscale gene-to-tissue modeling
- **Integrates UQ-PhysiCell** for sensitivity analysis, Bayesian calibration, and model validation
- **Supports complex workflow orchestration** across multiple biological scales

### LLM Integration Patterns

#### 1. Scenario-Driven Simulation Building
LLMs can construct complete simulations from biological descriptions:

**User Prompt**: *"Create a simulation of breast cancer cells in a hypoxic 3D environment with immune cell infiltration"*

**LLM Tool Chain**:
```
1. analyze_biological_scenario("Breast cancer in hypoxic 3D tissue with immune infiltration")
2. create_simulation_domain(3000, 3000, 500, max_time=7200)
3. add_single_substrate("oxygen", 100000, 0.01, 38)
4. add_single_cell_type("cancer_cell", "Ki67_basic")
5. add_single_cell_type("immune_cell", "live_cell")
6. add_single_cell_rule("cancer_cell", "oxygen", "decreases", "apoptosis rate")
7. export_xml_configuration("tumor_simulation.xml")
```

#### 2. Multiscale Integration Workflows
**User Prompt**: *"Integrate the p53 Boolean network into cancer cell behavior and simulate tumor growth under drug treatment"*

**Cross-Server Tool Chain**:
```
NeKo:       create_network(['TP53', 'MDM2', 'ATM']) → export_network('bnet')
MaBoSS:     bnet_to_bnd_and_cfg() → test Boolean dynamics
PhysiCell:  add_physiboss_model() → link genes to behaviors → simulate
```

#### 3. Interactive Simulation Design
**User Prompt**: *"What cell types and environmental factors should I include for studying drug resistance?"*

**LLM Response Pattern**:
- Analyzes biological scenario
- Suggests appropriate cell types and substrates
- Recommends signal-behavior rules
- Guides PhysiBoSS integration for gene-level control
- Provides complete simulation configuration

### Tool Categories Exposed

#### Simulation Framework
- `create_simulation_domain()` - Define 3D spatial and temporal boundaries
- `add_single_substrate()` - Add chemical environments (oxygen, drugs, nutrients)
- Session management with progress tracking across complex workflows

#### Cell Population Definition
- `add_single_cell_type()` - Define cancer, immune, stromal cell populations
- `configure_cell_parameters()` - Set size, motility, death rates
- `set_substrate_interaction()` - Define consumption and secretion

#### Behavioral Programming
- `add_single_cell_rule()` - Create environmental sensing and response
- `list_all_available_signals()` and `list_all_available_behaviors()` - Discovery tools
- Context-aware signal/behavior expansion based on simulation components

#### Initial Cell Placement
- `place_initial_cells()` - Place cells spatially using patterns (disc, rectangle, grid, annular, single)
- `get_initial_conditions_summary()` - Review current cell placements
- `remove_initial_cells()` - Clear placed cells
- `export_cells_csv()` - Export cells.csv for PhysiCell to load at simulation start

#### PhysiBoSS Multiscale Integration
- `add_physiboss_model()` - Integrate Boolean networks into cell behavior
- `add_physiboss_input_link()` - Connect environment to gene regulation
- `add_physiboss_output_link()` - Connect gene states to cell phenotypes
- `apply_physiboss_mutation()` - Simulate genetic perturbations

### Prompt Engineering Patterns

#### Pattern 1: Complete Simulation from Description
```
"Simulate [disease/scenario] with [environmental conditions] and [cell types]"
→ LLM automatically: scenario_analysis → domain_setup → cell_definition → rule_programming → export
```

#### Pattern 2: Multiscale Model Construction
```
"Connect this Boolean network to cell behavior in a tissue simulation"
→ LLM chains: physiboss_integration → input_output_linking → parameter_tuning → validation
```

#### Pattern 3: Iterative Simulation Refinement
```
"Add drug treatment effects to my existing simulation"
→ LLM extends: add_drug_substrate → modify_behavioral_rules → update_cell_interactions
```

#### Pattern 4: Initial Cell Placement with Spatial Patterns
```
"Place 500 tumor cells in a disc at the center and 100 immune cells in a ring around them"
→ LLM chains: place_initial_cells(tumor, random_disc) → place_initial_cells(immune, annular) → export_cells_csv
```

#### Pattern 5: Model Calibration Against Experimental Data
```
"Calibrate the tumor model's growth parameters against my experimental cell count data"
→ LLM chains: setup_uq → define_parameters → sensitivity_analysis → provide_data → calibrate → apply_parameters → validate
```

### Initial Cell Placement Guide

PhysiCell can load initial cell positions from a `cells.csv` file instead of using random placement. This is useful for:
- Recreating specific tissue architectures (tumor core + immune infiltrate)
- Setting up co-culture experiments with defined spatial arrangements
- Modeling layered tissues (epithelium, stroma, etc.)
- Reproducing experimental conditions from imaging data

#### How It Works

The `place_initial_cells()` tool generates x,y,z coordinates for cells and stores them in the session. Call it multiple times to build up complex layouts with different cell types and spatial patterns. Then `export_cells_csv()` writes the positions to a CSV file and configures the XML to load it.

#### Supported Spatial Patterns

| Pattern | Description | Key Parameters |
|---------|------------|----------------|
| `random_disc` | Uniformly distributed cells in a circular area | `center_x`, `center_y`, `radius`, `num_cells` |
| `random_rectangle` | Uniformly distributed cells in a rectangular area | `x_min`, `x_max`, `y_min`, `y_max`, `num_cells` |
| `single` | Place exactly one cell at a specific position | `center_x`, `center_y` |
| `grid` | Evenly spaced grid of cells | `x_min`, `x_max`, `y_min`, `y_max`, `spacing` |
| `annular` | Uniformly distributed cells in a ring | `center_x`, `center_y`, `radius`, `inner_radius`, `num_cells` |

#### Example Prompts

**Tumor with immune infiltration:**
> "Create a simulation with 500 breast cancer cells in a 200 micron disc at the center, surrounded by 100 cytotoxic T cells in a ring from 250 to 400 microns"

**LLM Tool Chain:**
```
1. create_simulation_domain(1000, 1000, 20)
2. add_single_substrate("oxygen", 100000, 0.01, 38)
3. add_single_cell_type("breast_cancer", "Ki67_basic")
4. add_single_cell_type("cytotoxic_T_cell", "live_cell")
5. place_initial_cells("breast_cancer", "random_disc", num_cells=500, radius=200)
6. place_initial_cells("cytotoxic_T_cell", "annular", num_cells=100, radius=400, inner_radius=250)
7. export_cells_csv()
8. export_xml_configuration()
9. create_physicell_project("tumor_immune")
```

**Tissue layers:**
> "Set up a simulation with an epithelial layer at the top and stromal cells below"

**LLM Tool Chain:**
```
1. place_initial_cells("epithelial", "grid", x_min=-400, x_max=400, y_min=100, y_max=200, spacing=15)
2. place_initial_cells("stromal", "random_rectangle", num_cells=300, x_min=-400, x_max=400, y_min=-200, y_max=80)
3. export_cells_csv()
```

**Scattered immune cells across the domain:**
> "Place 200 macrophages randomly across the entire simulation domain"

**LLM Tool Chain:**
```
1. place_initial_cells("macrophage", "random_rectangle", num_cells=200, x_min=-500, x_max=500, y_min=-500, y_max=500)
2. export_cells_csv()
```

#### Important Notes
- Cell types must be defined (via `add_single_cell_type()`) before placing cells
- Multiple `place_initial_cells()` calls accumulate — each call adds to the existing placements
- Use `remove_initial_cells()` to clear and start over
- `export_cells_csv()` automatically sets `number_of_cells=0` in the XML so PhysiCell uses the CSV instead of random placement
- Use `get_initial_conditions_summary()` to review placements before exporting

### Uncertainty Quantification & Model Calibration

The server integrates [UQ-PhysiCell](https://github.com/heberlr/UQ_PhysiCell) to provide end-to-end uncertainty quantification, sensitivity analysis, and model calibration capabilities. This enables LLMs to autonomously create models from natural language descriptions and then calibrate them against experimental data.

#### UQ Dependencies

```bash
# Core UQ (required)
pip install uq-physicell

# Bayesian Optimization (optional)
pip install torch botorch gpytorch

# ABC-SMC calibration (optional)
pip install pyabc
```

#### UQ Tool Categories

##### Setup & Parameter Definition
- `setup_uq_analysis()` - Initialize UQ for a compiled PhysiCell project (auto-detects config)
- `get_uq_parameter_suggestions()` - Analyze model and suggest calibratable parameters
- `define_uq_parameters()` - Select which parameters to vary with bounds and reference values
- `define_quantities_of_interest()` - Define what to measure from simulations (cell counts, etc.)

##### Sensitivity Analysis
- `run_sensitivity_analysis()` - Global (Sobol, LHS, Fast) or local (OAT) sensitivity analysis
- `get_sensitivity_results()` - View sensitivity indices and parameter rankings

##### Model Calibration
- `provide_experimental_data()` - Load reference/observed data CSV for calibration
- `run_bayesian_calibration()` - Multi-objective Bayesian optimization with Pareto front
- `run_abc_calibration()` - ABC-SMC posterior inference for full uncertainty estimates
- `get_calibration_status()` - Monitor long-running calibration jobs
- `get_calibration_results()` - View best-fit parameters, Pareto front, or posteriors

##### Validation & Application
- `apply_calibrated_parameters()` - Update model XML/rules with calibrated values
- `get_uq_summary()` - Overview of all UQ work in the session
- `list_uq_runs()` - List all SA and calibration runs with status

#### UQ Workflow

```
Natural Language Description
        ↓
Model Creation (existing tools)
        ↓
Compile & Run baseline simulation
        ↓
┌─── setup_uq_analysis() ───────────────────────────┐
│                                                     │
│  get_uq_parameter_suggestions()                     │
│         ↓                                           │
│  define_uq_parameters()                             │
│         ↓                                           │
│  define_quantities_of_interest()                    │
│         ↓                                           │
│  run_sensitivity_analysis()  ←── identify key params│
│         ↓                                           │
│  provide_experimental_data()                        │
│         ↓                                           │
│  run_bayesian_calibration()  ──or──  run_abc_calibration()
│         ↓                                           │
│  get_calibration_results()                          │
│         ↓                                           │
│  apply_calibrated_parameters()                      │
│         ↓                                           │
│  run_simulation()  ←── validation run               │
└─────────────────────────────────────────────────────┘
```

#### Parameter Types

The UQ system supports two types of calibratable parameters:

**XML Parameters** - Cell properties defined in PhysiCell_settings.xml, referenced by XPath:
```
.//cell_definitions/cell_definition[@name='tumor']/phenotype/cycle/phase_transition_rates/rate[1]
```

**Rules Parameters** - Hill function coefficients in cell_rules.csv, referenced by rule key:
```
tumor,oxygen,increases,cycle entry,half_max
tumor,oxygen,increases,cycle entry,hill_power
tumor,pressure,decreases,cycle entry,saturation
```

Each rule parameter targets a specific field (saturation/half_max/hill_power) of a specific signal-behavior rule.

#### Example: End-to-End Calibration

**User Prompt**: *"Create a tumor model with oxygen-dependent growth and pressure feedback, then calibrate it against my experimental cell count data"*

**LLM Tool Chain**:
```
# Phase 1: Model Creation
1. create_session()
2. create_simulation_domain(1500, 1500, 20, max_time=7200)
3. add_single_substrate("oxygen", 100000, 0.1, 38)
4. add_single_cell_type("tumor", "Ki67_basic")
5. add_single_cell_rule("tumor", "oxygen", "increases", "cycle entry",
     min_signal=0.00072, max_signal=0.0072, half_max=21.5, hill_power=4)
6. add_single_cell_rule("tumor", "pressure", "decreases", "cycle entry",
     min_signal=0, max_signal=0.00072, half_max=0.25, hill_power=3)
7. export_xml_configuration()
8. export_cell_rules_csv()
9. create_physicell_project("tumor_calibration")
10. compile_physicell_project("tumor_calibration")

# Phase 2: UQ Setup
11. setup_uq_analysis("tumor_calibration", num_replicates=3, num_workers=8)
12. get_uq_parameter_suggestions()
13. define_uq_parameters([
      {"name": "cycle_hfm", "type": "rules",
       "rule_key": "tumor,oxygen,increases,cycle entry,half_max",
       "ref_value": 21.5, "lower_bound": 10, "upper_bound": 35},
      {"name": "cycle_hp", "type": "rules",
       "rule_key": "tumor,oxygen,increases,cycle entry,hill_power",
       "ref_value": 4, "lower_bound": 1, "upper_bound": 10},
      {"name": "pressure_hfm", "type": "rules",
       "rule_key": "tumor,pressure,decreases,cycle entry,half_max",
       "ref_value": 0.25, "lower_bound": 0.05, "upper_bound": 1.0}
    ])
14. define_quantities_of_interest([
      {"name": "live_tumor", "function": "cell_count:tumor",
       "obs_column": "Tumor Count"}
    ])

# Phase 3: Sensitivity Analysis
15. run_sensitivity_analysis(method="Sobol", num_samples=64, num_workers=8)
16. get_sensitivity_results()  # Identify which params matter most

# Phase 4: Calibration
17. provide_experimental_data("obs_data.csv",
      column_mapping={"live_tumor": "Tumor Count"}, time_column="Time")
18. run_bayesian_calibration(num_initial_samples=10, num_iterations=50)
19. get_calibration_results()

# Phase 5: Validation
20. apply_calibrated_parameters()
21. run_simulation("tumor_calibration")
22. generate_simulation_gif()
```

#### Sensitivity Analysis Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `Sobol` | Global variance-based SA | Full parameter space exploration, interaction effects |
| `LHS` | Latin Hypercube Sampling | Efficient global space coverage |
| `OAT` | One-at-a-Time perturbation | Quick local sensitivity around reference values |
| `Fast` | Fourier Amplitude SA | Computationally efficient global SA |
| `Fractional Factorial` | Fractional factorial design | Screening many parameters |

#### Calibration Methods

| Method | Function | Output | Best For |
|--------|----------|--------|----------|
| **Bayesian Optimization** | `run_bayesian_calibration()` | Pareto-optimal parameter sets | Multi-objective fitting, expensive simulations |
| **ABC-SMC** | `run_abc_calibration()` | Posterior parameter distributions | Full uncertainty quantification, model selection |

#### Experimental Data Format

The calibration CSV should contain time-series observations:

```csv
Time,Tumor Count,Dead Cells
0,500,0
60,523,12
120,601,25
...
```

Column names are mapped to QoIs via `provide_experimental_data()`.

### Advanced PhysiBoSS Integration

#### Multiscale Architecture through MCP
The server enables LLMs to seamlessly connect molecular and cellular scales:

```
Gene Regulation (Boolean) ↔ Cell Behavior (PhysiCell) ↔ Tissue Dynamics (3D Physics)
        ↓                         ↓                        ↓
   Input: Environment        Output: Phenotype      Emergent: Population
   (oxygen, drugs)          (death, proliferation)   (growth, invasion)
```

#### LLM-Orchestrated Multiscale Workflows
**User Prompt**: *"Model how TP53 mutations affect tumor response to chemotherapy"*

**LLM Multiscale Tool Chain**:
1. **Network Level**: Construct TP53 regulatory network (NeKo)
2. **Boolean Level**: Simulate pathway dynamics (MaBoSS)  
3. **Cellular Level**: Link TP53 states to apoptosis/survival (PhysiCell PhysiBoSS)
4. **Tissue Level**: Simulate drug treatment effects on tumor population
5. **Analysis**: Compare wild-type vs mutant tumor responses

### Session Management for Complex Workflows

#### Multi-Session Orchestration
- `create_session()` - Isolated simulation environments
- `switch_session()` - Compare different scenarios
- `get_workflow_status()` - Track progress across complex builds

#### Workflow State Tracking
LLMs can monitor and guide users through simulation construction:
- Domain setup → Substrates → Cell types → Rules → PhysiBoSS → Export
- Progress percentages and next-step recommendations
- Error recovery with specific correction guidance

### Integration Benefits for LLMs

1. **Biological Scenario Translation**: Convert narrative descriptions into quantitative simulations
2. **Multiscale Coordination**: Orchestrate gene→cell→tissue modeling workflows
3. **Parameter Discovery**: Access to extensive signal/behavior libraries with context awareness
4. **Error Prevention**: Built-in validation and progress tracking
5. **Cross-Server Integration**: Seamless coordination with NeKo and MaBoSS servers

### Example LLM Conversation Flow

**User**: *"I want to study how hypoxia drives cancer cell invasion with p53 mutations."*

**LLM with MCP Access**:
1. **Scenario Analysis**: "I'll create a hypoxic tumor simulation with p53-controlled invasion"
2. **Domain Setup**: Creates 3D environment with oxygen gradients
3. **Cell Definition**: Adds cancer cells with motility and invasion capabilities  
4. **PhysiBoSS Integration**: 
   - Links oxygen levels to HIF1A activation
   - Connects p53 mutations to survival/invasion decisions
   - Programs hypoxia-induced migration behaviors
5. **Simulation Export**: Generates complete XML configuration
6. **Next Steps**: "The simulation is ready. Would you like me to add immune cells or test different p53 mutation scenarios?"

### Technical Architecture

- **Protocol**: Model Context Protocol (MCP) standard
- **Interface**: JSON-RPC tool calling with complex object handling
- **State Management**: Session-based simulation building with persistence
- **PhysiBoSS Integration**: Direct Boolean network coupling to cellular physics
- **UQ-PhysiCell Integration**: Sensitivity analysis, Bayesian optimization, ABC-SMC calibration
- **Cross-Server Coordination**: Seamless file handoff from NeKo/MaBoSS workflows

### Installation & Setup

#### Prerequisites

1. **Python 3.10+** - Required for the MCP server
2. **[uv](https://docs.astral.sh/uv/)** - Python package manager (used to run the server and manage dependencies)
   ```bash
   # Install uv (macOS/Linux)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **PhysiCell** - The C++ simulation framework must be compiled separately
   ```bash
   # Clone and compile PhysiCell
   git clone https://github.com/MathCancer/PhysiCell.git
   cd PhysiCell
   make                    # Compile the default project
   make save PROJ=template # Save the template project
   ```
   The MCP server will reference your PhysiCell installation when creating and compiling simulation projects.

#### Install the MCP Server

Clone this repository:
```bash
git clone https://github.com/your-org/mcp-biomodelling-servers.git
cd mcp-biomodelling-servers/PhysiCell
```

All Python dependencies (including `fastmcp`, `physicell-settings`, `cairosvg`, `pillow`, and `uq-physicell`) are declared in `pyproject.toml` and are **automatically installed** when the server is launched via `uv run`.

#### Optional UQ Dependencies

The core UQ features (sensitivity analysis) work out of the box. For advanced calibration methods, install the optional extras:

```bash
# Bayesian Optimization calibration (torch + botorch)
uv pip install --project /path/to/mcp-biomodelling-servers/PhysiCell torch botorch gpytorch

# ABC-SMC calibration (pyabc)
uv pip install --project /path/to/mcp-biomodelling-servers/PhysiCell pyabc

# Or install everything at once
uv pip install --project /path/to/mcp-biomodelling-servers/PhysiCell "physicell[all]"
```

The server gracefully degrades if these are not installed - core simulation and sensitivity analysis tools will still work, and calibration tools will return a message indicating which packages need to be installed.

#### Setup for Claude Desktop

Add the PhysiCell MCP server to your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "PhysiCell": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/mcp-biomodelling-servers/PhysiCell",
        "python",
        "/absolute/path/to/mcp-biomodelling-servers/PhysiCell/server.py"
      ]
    }
  }
}
```

Replace `/absolute/path/to/mcp-biomodelling-servers` with the actual path where you cloned the repository. Restart Claude Desktop after saving.

#### Setup for Claude Code

Use the `claude mcp add` command to register the server:

```bash
claude mcp add PhysiCell \
  -s user \
  -- uv run \
  --project /absolute/path/to/mcp-biomodelling-servers/PhysiCell \
  python /absolute/path/to/mcp-biomodelling-servers/PhysiCell/server.py
```

This registers the MCP server at the user level (`-s user`), making it available across all projects. You can also use `-s project` to register it only for the current project.

To verify the server is registered:
```bash
claude mcp list
```

#### Recommended: Add CLAUDE.md and Install Skill

Copy the included `CLAUDE.md` to `~/.claude/CLAUDE.md` to ensure Claude Code always uses the PhysiCell MCP tools directly rather than attempting manual workarounds:

```bash
cp /path/to/mcp-biomodelling-servers/PhysiCell/CLAUDE.md ~/.claude/CLAUDE.md
```

The repository also includes a `physicell-simulation` AgentSkill (in the `physicell-simulation/` directory) that provides detailed simulation guidance and prevents common configuration mistakes like the "from 0 towards 0" rule bug. The skill is automatically discovered by Claude Code via the `.claude-plugin/plugin.json` manifest when working in this project directory.

#### Verifying the Installation

Once configured, start Claude Desktop or Claude Code. The PhysiCell tools should be available. You can verify by asking:

> "What PhysiCell tools are available?"

or calling the `get_help()` tool. If the UQ tools show as unavailable, check that `uq-physicell` is installed in the server's virtual environment.

### Dependencies Summary

| Package | Purpose | Required? |
|---------|---------|-----------|
| `fastmcp` | MCP server framework | Yes (auto-installed) |
| `physicell-settings` | PhysiCell XML config manipulation | Yes (auto-installed) |
| `cairosvg` | SVG rendering for visualization | Yes (auto-installed) |
| `pillow` | Image processing for GIF generation | Yes (auto-installed) |
| `uq-physicell` | Sensitivity analysis & calibration | Yes (auto-installed) |
| `torch`, `botorch`, `gpytorch` | Bayesian Optimization calibration | Optional |
| `pyabc` | ABC-SMC calibration | Optional |

All required dependencies are automatically installed by `uv run` from `pyproject.toml`. Only the optional calibration backends need manual installation.

### AgentSkill: `physicell-simulation`

This repository includes a **Claude Code AgentSkill** that provides LLMs with comprehensive guidance for building PhysiCell simulations correctly. The skill uses progressive disclosure — metadata at startup, full instructions on activation, reference files on demand — keeping context efficient while preventing common configuration mistakes.

#### Why an AgentSkill?

LLMs frequently make configuration mistakes that produce broken or silent-failure simulations. The most common is the **"from 0 towards 0" bug**: setting a Hill function rule where both the `base_value` and the XML default rate are 0, causing the rule to interpolate between 0 and 0 (no effect). This was observed in real sessions where transition rules silently did nothing.

The skill prevents this and other mistakes by providing:
- Mandatory tool ordering (workflow sequence)
- Hill function parameter mapping with the critical "from 0 towards 0" prevention checklist
- Common mistakes quick reference table
- Typical biological parameter values
- Post-simulation verification checklist

#### Skill Structure

```
physicell-simulation/
├── SKILL.md                                   # Main instructions (~260 lines)
├── references/
│   ├── rules-and-hill-functions.md            # Hill function math, "from 0 towards 0" prevention
│   ├── parameter-reference.md                 # Typical values for biological scenarios
│   ├── troubleshooting.md                     # Error diagnosis by symptom
│   ├── uq-calibration-workflow.md             # UQ/sensitivity/calibration details
│   ├── physiboss-integration.md               # Boolean network integration
│   └── literature-validation.md               # Literature validation multi-server workflow
└── scripts/
    └── validate_config.py                     # Pre-flight XML/CSV validation script
```

#### Pre-Flight Validation Script

The `validate_config.py` script checks exported XML + rules CSV for common issues before running a simulation:

```bash
python3 physicell-simulation/scripts/validate_config.py PhysiCell_settings.xml cell_rules.csv
```

It checks for:
- "From 0 towards 0" rules (the #1 silent-failure bug)
- Missing substrates or cell types referenced in rules
- Dirichlet boundary consistency
- CRLF line endings in CSV files
- Invalid Hill function parameters

Exit codes: `0` = PASS, `1` = WARN, `2` = FAIL.

### Learn More

- **PhysiCell**: [Official Documentation](http://physicell.org/)
- **PhysiBoSS**: [Publication](https://doi.org/10.1093/bioinformatics/btz279)
- **UQ-PhysiCell**: [Documentation](https://uq-physicell.readthedocs.io/)
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/)

This MCP server transforms PhysiCell from a complex simulation framework into an **LLM-accessible multiscale modeling platform**, enabling natural language-driven construction, calibration, and validation of sophisticated gene-to-tissue simulations.
