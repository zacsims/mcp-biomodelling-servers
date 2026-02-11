# Literature Validation MCP Server
## Validate PhysiCell Cell Rules Against Published Biomedical Literature

This is a **Model Context Protocol (MCP) server** that validates PhysiCell cell behavior rules against published biomedical literature using [PaperQA2](https://github.com/Future-House/paper-qa) for retrieval-augmented generation (RAG) based question answering. It enables LLMs to automatically check whether proposed signal-behavior relationships in a simulation are supported by experimental evidence.

### What It Does

When building PhysiCell simulations, users define **cell rules** — relationships like "oxygen decreases cancer cell migration speed" or "TNF increases macrophage apoptosis rate". These rules encode biological assumptions, but are they actually supported by published literature?

This server answers that question by:

1. **Generating optimized search queries** for finding relevant papers on PubMed
2. **Indexing paper abstracts and full text** into a PaperQA2 document store
3. **Asking focused questions** about each rule's biological plausibility
4. **Returning evidence-based verdicts** with support levels, citations, and parameter suggestions

### Architecture: LLM-Orchestrated Multi-Server Workflow

The Literature Validation server does not work alone. It is one part of a four-server system orchestrated by the LLM:

```
LLM (Claude)
  |
  |-- PhysiCell MCP ---------- get_rules_for_validation()
  |                            store_validation_results()
  |                            get_validation_report()
  |
  |-- PubMed MCP ------------- search_pubmed()
  |                            get_article_details()
  |
  |-- bioRxiv MCP ------------ search_preprints()
  |                            get_preprint_details()
  |
  |-- LiteratureValidation --- suggest_search_queries()
       MCP (this server)       create_paper_collection()
                               add_papers_to_collection()
                               add_papers_by_id()
                               validate_rule()
                               validate_rules_batch()
                               get_validation_summary()
```

The LLM reads rules from the PhysiCell session, searches PubMed and bioRxiv for evidence, indexes papers here (with full PDF download when available), runs validation queries, and stores results back in the PhysiCell session. No server-to-server communication is needed — the LLM handles all coordination.

### Why a Separate Server?

PaperQA2 requires Python 3.11+, while the PhysiCell MCP server is pinned to Python 3.10 (due to its dependency chain). Running PaperQA in a separate process with its own virtual environment avoids version conflicts entirely.

### Tools

#### `create_paper_collection(name)`

Creates a named document collection for a validation session. Each collection is an independent PaperQA document store, so you can maintain separate literature bases for different biological topics.

```
Input:  name = "hypoxia_migration"
Output: Collection created with papers directory at
        ~/Documents/LiteratureValidation/hypoxia_migration/papers/
```

#### `add_papers_to_collection(name, papers, fetch_pdfs=False)`

Indexes papers into a collection. Each paper is a dict with `title` and `text` (abstract or full text), plus optional metadata (`pmid`, `doi`, `biorxiv_doi`, `authors`, `year`). When `fetch_pdfs=True`, attempts to download full PDFs before falling back to text-only indexing.

```
Input:  name = "hypoxia_migration"
        papers = [
          {"title": "Hypoxia induces cancer cell migration...",
           "text": "Abstract text here...",
           "pmid": "12345678", "year": "2020"}
        ]
        fetch_pdfs = True
Output: 1 / 1 papers indexed (PDFs: 1 | Text-only: 0)
```

The typical workflow is: search PubMed via a PubMed MCP server, collect abstracts, then pass them here for indexing.

#### `add_papers_by_id(name, pmids=None, biorxiv_dois=None, fetch_pdfs=True)`

Streamlined tool for adding papers when you already have PubMed IDs or bioRxiv DOIs. Automatically fetches metadata (title, abstract) and downloads full PDFs when available.

PDF discovery uses a three-layer strategy:
1. **bioRxiv direct download** — always available for bioRxiv preprints
2. **metapub FindIt** — publisher-specific strategies for 68+ publishers (Elsevier, Wiley, Springer, Nature, PLOS, BMC, PMC, etc.)
3. **Unpaywall** — finds preprints (arXiv, bioRxiv, medRxiv) and green OA copies of paywalled papers

```
Input:  name = "hypoxia_migration"
        pmids = ["35486828", "33264437", "28400552"]
        fetch_pdfs = True
Output: 3 / 3 papers indexed (PDFs: 2 | Text-only: 1)
```

#### `suggest_search_queries(cell_type, signal, direction, behavior)`

Generates 5 optimized PubMed search queries for a given rule. Uses built-in synonym maps for common biological terms (e.g., "oxygen" expands to "oxygen OR hypoxia OR O2 OR pO2") and produces queries at different specificity levels:

1. **Primary** — most specific, exact terms
2. **Broad** — OR-grouped synonyms for signal, behavior, and cell type
3. **MeSH-style** — quoted terms for precise matching
4. **Quantitative** — targets dose-response data (EC50, Hill coefficient)
5. **Reviews** — filtered to review articles for overview evidence

```
Input:  cell_type = "cancer", signal = "oxygen",
        direction = "decreases", behavior = "migration_speed"
Output: 5 ready-to-use PubMed query strings
```

#### `validate_rule(name, cell_type, signal, direction, behavior, ...)`

The core validation tool. Constructs a focused question about the rule's biological plausibility and asks PaperQA to answer it using the indexed literature. Returns:

- **Support level**: `strong`, `moderate`, `weak`, `contradictory`, or `unsupported`
- **Evidence summary**: PaperQA's narrative answer with reasoning
- **References**: Cited papers from the collection

Optional parameters `half_max` and `hill_power` trigger additional checks on whether the proposed parameter values are consistent with published dose-response data.

```
Input:  name = "hypoxia_migration"
        cell_type = "cancer", signal = "oxygen",
        direction = "decreases", behavior = "migration_speed",
        half_max = 10.0, hill_power = 4.0
Output: Support Level: STRONG
        Evidence: "Hypoxia-induced migration is well established..."
        References: [Smith et al. 2020, Jones et al. 2019, ...]
```

#### `validate_rules_batch(name, rules)`

Validates multiple rules sequentially against the same collection. Each rule dict needs `cell_type`, `signal`, `direction`, `behavior`, and optionally `half_max` and `hill_power`. Returns combined results with per-rule verdicts.

```
Input:  name = "tumor_model"
        rules = [
          {"cell_type": "cancer", "signal": "oxygen",
           "direction": "decreases", "behavior": "migration_speed"},
          {"cell_type": "cancer", "signal": "pressure",
           "direction": "decreases", "behavior": "cycle entry"}
        ]
Output: Batch results for 2 rules with individual support levels
```

#### `get_validation_summary(name)`

Returns a summary of all validations performed on a collection: support level distribution, rules needing attention (unsupported/contradictory), and well-supported rules.

### Support Levels

| Level | Meaning |
|-------|---------|
| **strong** | Well-established in literature with extensive evidence |
| **moderate** | Supported by multiple studies, consistent with evidence |
| **weak** | Limited or preliminary evidence, few studies |
| **contradictory** | Conflicting evidence in the literature |
| **unsupported** | No evidence found, or evidence contradicts the rule |

Support levels are determined by analyzing PaperQA's answer text for characteristic phrases. Rules flagged as `unsupported` or `contradictory` should be reviewed and potentially revised before running simulations.

### Complete Validation Workflow

Here is the full end-to-end workflow as the LLM would execute it:

```
# Step 1: Export rules from the PhysiCell session
PhysiCell MCP:  get_rules_for_validation()
                → Returns JSON list of all rules with parameters

# Step 2: Create a literature collection
LitValidation:  create_paper_collection("tumor_model_v1")

# Step 3: For each rule, generate search queries
LitValidation:  suggest_search_queries("cancer", "oxygen",
                  "decreases", "migration_speed")
                → Returns 5 PubMed query strings

# Step 4: Search PubMed and bioRxiv for papers
PubMed MCP:     search_pubmed("cancer cells oxygen decreases migration")
                get_article_details(pmid_list)
                → Returns titles, abstracts, metadata
bioRxiv MCP:    search_preprints(category="cancer biology", recent_days=90)
                → Returns preprint DOIs

# Step 5: Index papers into the collection (with full PDF download)
LitValidation:  add_papers_by_id("tumor_model_v1",
                  pmids=["12345678", ...],
                  biorxiv_dois=["10.1101/2024.01.15.123456", ...],
                  fetch_pdfs=True)

# Step 6: Repeat steps 3-5 for each rule to build the literature base

# Step 7: Validate all rules at once
LitValidation:  validate_rules_batch("tumor_model_v1", rules)
                → Returns per-rule support levels with evidence

# Step 8: Review results
LitValidation:  get_validation_summary("tumor_model_v1")

# Step 9: Store results back in the PhysiCell session
PhysiCell MCP:  store_validation_results(validation_results)

# Step 10: View the full report
PhysiCell MCP:  get_validation_report()
                → Comprehensive report with all validations,
                  parameter suggestions, and unvalidated rules
```

### Example: Validating a Hypoxia-Migration Rule

**User prompt**: *"Validate my simulation rules against published literature"*

The LLM would:

1. Call `get_rules_for_validation()` on PhysiCell MCP — finds rule: "oxygen decreases cancer cell migration_speed, half_max=10, hill_power=4"

2. Call `suggest_search_queries("cancer", "oxygen", "decreases", "migration_speed")` — gets queries like:
   - `cancer cells oxygen decreases migration`
   - `("cancer cells" OR "tumor cells") AND ("oxygen" OR "hypoxia") AND ("migration" OR "motility")`
   - `"oxygen" AND "migration" AND (EC50 OR "half-maximal" OR "dose-response")`

3. Search PubMed and collect 15 relevant abstracts

4. Call `add_papers_to_collection("hypoxia_migration", papers)` — indexes all 15

5. Call `validate_rule("hypoxia_migration", "cancer", "oxygen", "decreases", "migration_speed", half_max=10, hill_power=4)`

6. Gets back:
   > **Support Level: STRONG**
   >
   > Hypoxia-induced cancer cell migration is well established in the literature. Multiple studies demonstrate that low oxygen environments activate HIF-1alpha signaling, which upregulates migration-associated genes. The proposed half-max of 10 mmHg is consistent with published EC50 values for hypoxia-responsive migration in breast cancer cells (8-15 mmHg range). A Hill coefficient of 4 suggests a moderately switch-like response, which is reasonable for HIF-1alpha-mediated transcriptional programs.

7. Stores results in the PhysiCell session for the validation report

### Installation & Setup

#### Prerequisites

- **Python 3.11+** — Required by PaperQA2
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **OPENAI_API_KEY** — PaperQA uses OpenAI models via litellm for question answering

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-proj-..."
```

#### Install Dependencies

```bash
cd mcp-biomodelling-servers/LiteratureValidation
uv sync
```

Dependencies (`fastmcp`, `paper-qa`, `httpx`) are declared in `pyproject.toml` and automatically installed by `uv`.

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
        "OPENAI_API_KEY": "sk-proj-..."
      }
    }
  }
}
```

#### Verify Installation

After registering, the LiteratureValidation tools should appear alongside PhysiCell tools. Ask:

> "What literature validation tools are available?"

or call `suggest_search_queries()` with a test rule to confirm the server is responding.

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for OpenAI (used by PaperQA via litellm) |
| `LITERATURE_VALIDATION_DIR` | `~/Documents/LiteratureValidation` | Root directory for paper collections |

### PaperQA2 Configuration

The server configures PaperQA with:

- **LLM**: `gpt-4o` via litellm (main reasoning)
- **Summary LLM**: `gpt-4o-mini` via litellm (document summarization)
- **Embeddings**: `text-embedding-3-small` via litellm
- **Evidence retrieval**: Top 10 chunks, max 5 sources per answer

### File Storage

Papers and collections are stored under `LITERATURE_VALIDATION_DIR`:

```
~/Documents/LiteratureValidation/
  hypoxia_migration/
    papers/
      a1b2c3d4e5f6.pdf    # Full PDF (when available)
      b2c3d4e5f6a1.txt    # Abstract text (fallback)
      ...
  tumor_immune/
    papers/
      ...
```

PDF files are downloaded from PubMed Central, publisher sites (via metapub), or preprint repositories (via Unpaywall). Text files contain metadata headers (title, authors, year, PMID, DOI) followed by the abstract.

### Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `fastmcp` | MCP server framework | Yes |
| `paper-qa` | RAG-based question answering over scientific literature | Yes |
| `httpx` | Async HTTP client for API calls | Yes |
| `metapub` | PDF discovery across 68+ publishers via FindIt | Yes |
| `pypdf[image]` | PDF parsing with image extraction support | Yes |
| `setuptools` | Runtime dependency for metapub/eutils | Yes |
| `coredis` | Redis client (paper-qa dependency) | Yes |

### Integration with PhysiCell MCP

The PhysiCell MCP server provides three companion tools for the validation workflow:

| Tool | Purpose |
|------|---------|
| `get_rules_for_validation()` | Exports current rules as structured JSON for validation |
| `store_validation_results(validations)` | Persists validation results in the PhysiCell session state |
| `get_validation_report()` | Generates a comprehensive report with support levels, evidence, citations, and parameter suggestions |

Validation results stored in the PhysiCell session include per-rule support levels, evidence summaries, suggested parameter adjustments, and key citations. These persist alongside the simulation configuration and appear in `get_simulation_summary()`.

### Learn More

- **PaperQA2**: [GitHub](https://github.com/Future-House/paper-qa) — The RAG engine powering literature queries
- **PhysiCell**: [Official Site](http://physicell.org/) — The simulation framework whose rules are validated
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/) — The protocol enabling tool interoperability
