# Literature Validation MCP Server
## Validate PhysiCell Cell Rules Against Published Biomedical Literature

This is a **Model Context Protocol (MCP) server** that validates PhysiCell cell behavior rules against published biomedical literature using [Edison Scientific's PaperQA3 API](https://platform.edisonscientific.com/). Edison automatically searches **150M+ papers** — no manual paper curation needed. It enables LLMs to automatically check whether proposed signal-behavior relationships in a simulation are supported by experimental evidence.

### What It Does

When building PhysiCell simulations, users define **cell rules** — relationships like "oxygen decreases cancer cell migration speed" or "TNF increases macrophage apoptosis rate". These rules encode biological assumptions, but are they actually supported by published literature?

This server answers that question by:

1. **Asking direction-agnostic questions** about each rule's biological plausibility — PaperQA independently determines the direction of the relationship from literature, preventing leading-question bias
2. **Automatically searching 150M+ papers** via Edison's PaperQA3 API — no manual paper collection, no PDF downloads, no indexing
3. **Returning evidence-based verdicts** with literature-determined direction (`DIRECTION`), evidence strength (`VERDICT`), citations, and parameter suggestions

### Architecture: LLM-Orchestrated Multi-Server Workflow

The Literature Validation server works alongside the PhysiCell MCP server, orchestrated by the LLM:

```
LLM (Claude)
  |
  |-- PhysiCell MCP ---------- get_rules_for_validation()
  |                            store_validation_results()
  |                            get_validation_report()
  |
  |-- LiteratureValidation --- validate_rule()
       MCP (this server)       validate_rules_batch()
                               get_validation_summary()
```

The LLM reads rules from the PhysiCell session, validates them here via the Edison API, and stores results back in the PhysiCell session. No paper management or additional MCP servers are needed.

### Why a Separate Server?

The Edison client and its dependencies are separate from the PhysiCell MCP server's dependency chain. Running the validation server in its own process avoids version conflicts and keeps the PhysiCell server lightweight.

### Tools

#### `validate_rule(name, cell_type, signal, direction, behavior, ...)`

The core validation tool. Checks for a cached answer file on disk first — if found, returns the cached result instantly. Otherwise, constructs a **direction-agnostic** question (e.g., "What is the effect of oxygen on migration_speed in cancer cells?") and sends it to Edison's PaperQA3 API. Edison automatically searches 150M+ papers and returns an evidence-based answer. Returns:

- **Literature direction**: `DIRECTION: INCREASES`, `DECREASES`, or `AMBIGUOUS` — what PaperQA determined from the evidence
- **Direction match**: Whether the literature direction agrees with the proposed rule direction. Mismatches are prominently flagged.
- **Support level**: `VERDICT: STRONG`, `MODERATE`, `WEAK`, or `UNSUPPORTED` — evidence strength
- **Evidence summary**: PaperQA's narrative answer with reasoning
- **References**: Cited papers

Optional parameters `half_max` and `hill_power` are mentioned "for reference" in the question without stating the proposed direction.

The full answer is saved to `~/Documents/LiteratureValidation/{name}/answers/{cell_type}_{signal}_{behavior}.md`. The PhysiCell MCP server's `store_validation_results()` reads these files directly to extract verdicts, ensuring the agent cannot fabricate or alter results.

```
Input:  name = "tumor_model_v1"
        cell_type = "cancer", signal = "oxygen",
        direction = "decreases", behavior = "migration_speed",
        half_max = 10.0, hill_power = 4.0
Output: Literature Direction: decreases
        Direction: Confirmed by literature
        Support Level: STRONG
        Evidence: "Hypoxia-induced migration is well established..."
        References: [Smith et al. 2020, Jones et al. 2019, ...]
```

#### `validate_rules_batch(name, rules)`

Validates multiple rules against literature **in parallel**. Uncached rules are sent to Edison as a single batch (up to 10 concurrent API calls). Rules that already have answer files on disk from a previous validation are served from cache instantly — no duplicate API calls. Each rule dict needs `cell_type`, `signal`, `direction`, `behavior`, and optionally `half_max` and `hill_power`. Returns combined results with per-rule verdicts and a cache/query breakdown.

```
Input:  name = "tumor_model_v1"
        rules = [
          {"cell_type": "cancer", "signal": "oxygen",
           "direction": "decreases", "behavior": "migration_speed"},
          {"cell_type": "cancer", "signal": "pressure",
           "direction": "decreases", "behavior": "cycle entry"}
        ]
Output: Batch results for 2 rules (e.g., "1 cached, 1 queried via Edison API")
```

#### `get_validation_summary(name)`

Returns a summary of all validations performed: support level distribution, rules needing attention (unsupported/weak), and well-supported rules.

### Support Levels

PaperQA returns two verdicts per rule:

**DIRECTION** — what the literature says about the relationship direction:
| Direction | Meaning |
|-----------|---------|
| **increases** | Literature indicates increasing signal increases the behavior |
| **decreases** | Literature indicates increasing signal decreases the behavior |
| **ambiguous** | Evidence is mixed or insufficient to determine direction |

**VERDICT** — evidence strength (independent of direction):
| Level | Meaning |
|-------|---------|
| **strong** | Well-established in literature with extensive quantitative evidence |
| **moderate** | Supported by multiple studies, consistent qualitative evidence |
| **weak** | Limited or preliminary evidence, few studies |
| **unsupported** | No relevant evidence found |

Support levels are extracted from the `VERDICT:` line in PaperQA's structured answer. A fifth level, **contradictory**, is auto-assigned by the PhysiCell MCP server when the literature direction disagrees with the proposed rule direction. Rules flagged as `unsupported` or `contradictory` should be reviewed and potentially revised before running simulations.

### Complete Validation Workflow

```
# Step 1: Export rules from the PhysiCell session
PhysiCell MCP:  get_rules_for_validation()
                → Returns JSON list of all rules with parameters

# Step 2: Validate all rules at once
LitValidation:  validate_rules_batch("tumor_model_v1", rules)
                → Edison searches 150M+ papers automatically
                → Returns per-rule support levels with evidence

# Step 3: Review results
LitValidation:  get_validation_summary("tumor_model_v1")

# Step 4: Store results back in the PhysiCell session
PhysiCell MCP:  store_validation_results([
                  {"cell_type": "cancer", "signal": "oxygen",
                   "direction": "decreases", "behavior": "migration_speed",
                   "collection_name": "tumor_model_v1"},
                  ...
                ])
                → VERDICT and DIRECTION extracted server-side
                → Direction mismatches auto-flagged as contradictory

# Step 5: View the full report
PhysiCell MCP:  get_validation_report()
                → Comprehensive report with direction match status,
                  support levels, evidence, and unvalidated rules
```

### Installation & Setup

#### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **EDISON_PLATFORM_API_KEY** — Get from [Edison Scientific](https://platform.edisonscientific.com/profile)

```bash
# Set your Edison API key
export EDISON_PLATFORM_API_KEY="your-api-key"
```

#### Install Dependencies

```bash
cd mcp-biomodelling-servers/LiteratureValidation
uv sync
```

Dependencies (`fastmcp`, `edison-client`, `httpx`) are declared in `pyproject.toml` and automatically installed by `uv`.

#### Register with Claude Code

```bash
claude mcp add LiteratureValidation \
  -s user \
  -- uv run \
  --project /absolute/path/to/mcp-biomodelling-servers/LiteratureValidation \
  python /absolute/path/to/mcp-biomodelling-servers/LiteratureValidation/server.py
```

#### Register with Claude Desktop

Add to your Claude Desktop configuration:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "LiteratureValidation": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/mcp-biomodelling-servers/LiteratureValidation",
        "python",
        "/absolute/path/to/mcp-biomodelling-servers/LiteratureValidation/server.py"
      ],
      "env": {
        "EDISON_PLATFORM_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### Verify Installation

After registering, the LiteratureValidation tools should appear alongside PhysiCell tools. Ask:

> "What literature validation tools are available?"

or call `validate_rule()` with a test rule to confirm the server is responding.

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `EDISON_PLATFORM_API_KEY` | *(required)* | API key for Edison Scientific PaperQA3 |
| `LITERATURE_VALIDATION_DIR` | `~/Documents/LiteratureValidation` | Root directory for answer files |

### File Storage

Answer files are stored under `LITERATURE_VALIDATION_DIR`:

```
~/Documents/LiteratureValidation/
  tumor_model_v1/
    answers/
      cancer_oxygen_migration_speed.md   # Edison answer file
      cancer_pressure_cycle_entry.md     # (direction-agnostic filename)
      ...
  tumor_immune/
    answers/
      ...
```

Answer files are written by `validate_rule()` and contain the full PaperQA response including `DIRECTION:` and `VERDICT:` lines. The PhysiCell MCP server's `store_validation_results()` reads these files directly from disk as the authoritative source of truth — this prevents the agent from fabricating or altering validation results.

### Answer File Caching

Both `validate_rule()` and `validate_rules_batch()` check for existing answer files before calling Edison. If an answer file already exists for a rule (keyed by `{cell_type}_{signal}_{behavior}`), the cached result is returned instantly with no API call.

This is important for two scenarios:
- **Model rebuilds** — if parameter errors cause a crash and the model must be rebuilt from scratch, re-running validation reuses all cached answers
- **Context compaction** — if the conversation compacts and loses track of validation state, the agent can re-run the full validation pipeline and it completes instantly from cache

To force a fresh validation (e.g., after changing a rule's direction), delete the corresponding answer file from disk before re-validating.

### Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `fastmcp` | MCP server framework | Yes |
| `edison-client` | Edison Scientific PaperQA3 API client | Yes |
| `httpx` | Async HTTP client | Yes |

### Integration with PhysiCell MCP

The PhysiCell MCP server provides three companion tools for the validation workflow:

| Tool | Purpose |
|------|---------|
| `get_rules_for_validation()` | Exports current rules as structured JSON for validation |
| `store_validation_results(validations)` | Reads PaperQA answer files from disk and persists results in the PhysiCell session. Each validation dict needs `cell_type`, `signal`, `direction`, `behavior`, and optionally `collection_name`. VERDICT and DIRECTION are extracted server-side from the authoritative answer files — the agent cannot override them. Direction mismatches are auto-flagged as `contradictory`. |
| `get_validation_report()` | Generates a comprehensive report with support levels, direction match status, evidence, citations, and parameter suggestions |

### Learn More

- **Edison Scientific PaperQA3**: [Platform](https://platform.edisonscientific.com/) — The API powering literature queries over 150M+ papers
- **PhysiCell**: [Official Site](http://physicell.org/) — The simulation framework whose rules are validated
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/) — The protocol enabling tool interoperability
