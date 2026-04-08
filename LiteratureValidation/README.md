# Literature Search MCP Server
## Evidence-Based Parameter Guidance for PhysiCell Simulations

A **Model Context Protocol (MCP) server** that searches published biomedical literature using [Edison Scientific's PaperQA3 API](https://platform.edisonscientific.com/). Edison automatically searches **150M+ papers** — no manual paper curation, PDF downloads, or indexing needed.

### What It Does

When building PhysiCell simulations, agents need evidence-based parameter values for cell behavior rules — questions like "What oxygen concentration triggers tumor necrosis?" or "What is a typical migration speed for breast cancer cells?". This server answers those questions by searching the literature and returning cited, evidence-based answers.

The server exposes a single tool:

- **`search_literature(query, collection_name)`** — Ask any biological question, get an answer with citations

The agent calls `search_literature()` *before* proposing rules, using the returned evidence to choose parameter values (half_max, hill_power, rates, etc.). After adding each rule, the agent records the evidence basis via the PhysiCell MCP's `store_rule_justification()` tool.

### Architecture

```
LLM (Claude)
  |
  |-- PhysiCell MCP ---------- add_single_cell_rule()
  |                            store_rule_justification()
  |                            get_rule_justifications()
  |
  |-- LiteratureValidation --- search_literature()
       MCP (this server)
```

The LLM searches literature here to inform parameter choices, builds rules via the PhysiCell MCP, and stores justifications back in the PhysiCell session. No paper management or additional servers needed.

### Workflow

Literature search happens *during* model building, not as a separate validation phase:

```
# 1. Search for evidence before adding a rule
search_literature("What oxygen concentration causes half-maximal necrosis in solid tumors?")
→ Returns: "Tumor necrosis occurs below ~5 mmHg O₂, with half-maximal
   effect around 3.75 mmHg..." (with citations)

# 2. Use the evidence to add the rule (PhysiCell MCP)
add_single_cell_rule(cell_type="tumor", signal="oxygen",
    direction="decreases", behavior="necrosis",
    half_max=3.75, hill_power=8, saturation_value=0)

# 3. Record the justification (PhysiCell MCP)
store_rule_justification(cell_type="tumor", signal="oxygen",
    direction="decreases", behavior="necrosis",
    justification="Hypoxia-induced necrosis threshold ~5 mmHg O₂",
    key_citations="Vaupel 2004, Grimes 2014")

# 4. After all rules are added, generate the evidence report (PhysiCell MCP)
get_rule_justifications()
→ Report showing each rule's evidence basis, flagging unjustified rules
```

### Tool: `search_literature`

Ask any biological question. Edison searches 150M+ papers and returns an evidence-based answer with citations.

**Parameters:**
- `query` (str, required) — Any question about biology, cell behavior, parameters, etc.
- `collection_name` (str, default: `"default"`) — Cache namespace for organizing results

**Example queries:**
```
"What is the effect of oxygen on tumor cell necrosis?"
"What oxygen concentration causes half-maximal necrosis in solid tumors?"
"What is a typical migration speed for breast cancer cells in vitro?"
"Do macrophages increase or decrease tumor proliferation?"
"What is the Hill coefficient for oxygen-dependent cell cycle arrest?"
```

**Returns:** Markdown-formatted answer with evidence summary and cited references.

### Caching

Results are cached to disk by SHA-256 hash of the normalized query. Repeated queries return instantly with no API call. Cache location:

```
~/Documents/LiteratureValidation/
  {collection_name}/
    answers/
      {hash}.md    # Full Edison answer
      ...
```

To force a fresh query, delete the corresponding answer file.

### Why a Separate Server?

The Edison client and its dependencies are separate from the PhysiCell MCP server's dependency chain. Running in its own process avoids version conflicts and keeps the PhysiCell server lightweight.

### Installation & Setup

#### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **EDISON_PLATFORM_API_KEY** — Get from [Edison Scientific](https://platform.edisonscientific.com/profile)

```bash
export EDISON_PLATFORM_API_KEY="your-api-key"
```

#### Install Dependencies

```bash
cd mcp-biomodelling-servers/LiteratureValidation
uv sync
```

#### Register with Claude Code

```bash
claude mcp add LiteratureValidation \
  -s user \
  -- uv run \
  --project /absolute/path/to/mcp-biomodelling-servers/LiteratureValidation \
  python /absolute/path/to/mcp-biomodelling-servers/LiteratureValidation/server.py
```

#### Register with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `EDISON_PLATFORM_API_KEY` | *(required)* | API key for Edison Scientific PaperQA3 |
| `LITERATURE_VALIDATION_DIR` | `~/Documents/LiteratureValidation` | Root directory for cached answer files |

### Dependencies

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `edison-client` | Edison Scientific PaperQA3 API client |
| `httpx` | Async HTTP client |

### Learn More

- **Edison Scientific PaperQA3**: [Platform](https://platform.edisonscientific.com/) — 150M+ paper search API
- **PhysiCell**: [Official Site](http://physicell.org/) — Multicellular simulation framework
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/) — Tool interoperability protocol
