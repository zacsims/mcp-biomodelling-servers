# Literature Validation Guide

Validate PhysiCell cell rules against published biomedical literature using a multi-server orchestration workflow.

## Architecture

Three MCP servers are involved:

1. **PhysiCell MCP** — Exports rules, stores validation results, generates reports
2. **LiteratureValidation MCP** — Creates paper collections, suggests queries, validates rules using PaperQA2
3. **PubMed MCP** (Anthropic plugin) — Searches PubMed, retrieves article abstracts

The LLM orchestrates between all three servers.

## When to Validate

- **After defining cell rules**, before running the simulation
- **For novel biological assumptions** that aren't well-established
- **When rules produce unexpected results** — check if the biology is correct
- **Before publishing** simulation results

## Complete Workflow

### Step 1: Export Rules from PhysiCell

```python
mcp__PhysiCell__get_rules_for_validation()
```

Returns rules as structured JSON:
```json
[
    {
        "cell_type": "tumor",
        "signal": "oxygen",
        "direction": "decreases",
        "behavior": "necrosis",
        "half_max": 3.75,
        "hill_power": 8
    }
]
```

### Step 2: Create Paper Collection

```python
mcp__LiteratureValidation__create_paper_collection("tumor_oxygen_model")
```

Creates a PaperQA2 document store for indexing and querying papers.

### Step 3: Get Search Queries

For each rule (or batch):

```python
mcp__LiteratureValidation__suggest_search_queries(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis"
)
```

Returns ~5 queries at different specificity levels:
```
1. "oxygen hypoxia tumor necrosis cell death"
2. "hypoxia-induced necrosis cancer cells"
3. "oxygen tension necrotic core tumor spheroid"
4. "HIF pathway necrosis solid tumor"
5. "pO2 threshold tumor cell death mechanism"
```

### Step 4: Search PubMed

Use the PubMed MCP tools with the suggested queries:

```python
search_pubmed("hypoxia-induced necrosis cancer cells", max_results=10)
# Then for each relevant result:
get_article_details(pmid="12345678")
```

### Step 5: Index Papers

Format papers and add to the collection:

```python
mcp__LiteratureValidation__add_papers_to_collection(
    name="tumor_oxygen_model",
    papers=[
        {
            "title": "Hypoxia-induced necrosis in solid tumors",
            "text": "Abstract text or full text content...",
            "pmid": "12345678",
            "doi": "10.1234/example",
            "authors": "Smith et al.",
            "year": "2023"
        },
        {
            "title": "Another relevant paper",
            "text": "Abstract text...",
            "pmid": "23456789"
        }
    ]
)
```

**Paper dict format**:
- `title` (str, required) — Paper title
- `text` (str, required) — Abstract or full text content
- `pmid` (str, optional) — PubMed ID
- `doi` (str, optional) — Digital Object Identifier
- `authors` (str, optional) — Author list
- `year` (str, optional) — Publication year

### Step 6: Repeat Steps 3-5

Iterate for all rules to build a comprehensive literature base. You can batch queries for efficiency.

### Step 7: Validate Rules

Validate all rules at once:

```python
mcp__LiteratureValidation__validate_rules_batch(
    name="tumor_oxygen_model",
    rules=[
        {
            "cell_type": "tumor",
            "signal": "oxygen",
            "direction": "decreases",
            "behavior": "necrosis",
            "half_max": 3.75,
            "hill_power": 8
        }
    ]
)
```

Or validate one at a time:

```python
mcp__LiteratureValidation__validate_rule(
    name="tumor_oxygen_model",
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis",
    half_max=3.75,
    hill_power=8
)
```

### Step 8: Get Validation Summary

```python
mcp__LiteratureValidation__get_validation_summary("tumor_oxygen_model")
```

Returns distribution of support levels and flagged rules.

### Step 9: Store Results in PhysiCell Session

```python
mcp__PhysiCell__store_validation_results(validations=[
    {
        "cell_type": "tumor",
        "signal": "oxygen",
        "direction": "decreases",
        "behavior": "necrosis",
        "support_level": "strong",
        "evidence_summary": "Well-established that hypoxia induces necrosis in solid tumors",
        "suggested_half_max": 3.75,
        "suggested_hill_power": 8,
        "key_citations": ["Smith et al. 2023", "Jones et al. 2022"]
    }
])
```

**Validation dict fields**:
- `cell_type` (str, required)
- `signal` (str, required)
- `direction` (str, required) — "increases" or "decreases"
- `behavior` (str, required)
- `support_level` (str, required) — "strong", "moderate", "weak", "contradictory", "unsupported"
- `evidence_summary` (str, required) — Summary of literature evidence
- `suggested_half_max` (float, optional) — Literature-suggested value
- `suggested_hill_power` (float, optional) — Literature-suggested value
- `key_citations` (list[str], optional) — Key paper references

### Step 10: Generate Report

```python
mcp__PhysiCell__get_validation_report()
```

Returns a markdown report showing each rule's validation status, evidence, and suggestions.

## Support Level Interpretation

| Level | Meaning | Action |
|-------|---------|--------|
| **strong** | Multiple papers directly support this relationship | Keep as-is, consider using suggested parameters |
| **moderate** | Some supporting evidence, possibly indirect | Acceptable, note limitations |
| **weak** | Limited or tangential evidence | Consider if the relationship is critical; add caveats |
| **contradictory** | Literature shows opposite or conflicting results | Review carefully; may need to remove or reverse the rule |
| **unsupported** | No evidence found in indexed papers | Not necessarily wrong, but needs justification; consider adding more papers |

## Acting on Validation Results

### For `strong` or `moderate` support
- Keep the rule
- Consider updating `half_max` and `hill_power` if the validation suggests specific values based on dose-response data

### For `weak` support
- Document the assumption
- Consider whether the rule is essential to the model
- Look for additional papers that might strengthen the evidence

### For `contradictory` support
- Read the cited papers carefully
- The contradiction may be context-dependent (different cell type, species, conditions)
- Consider reversing the direction or removing the rule
- Document the decision and rationale

### For `unsupported` support
- Not the same as "disproven" — may just mean no papers were indexed
- Try broader search queries
- Consider if the relationship is derived from first principles or expert knowledge
- If keeping, explicitly document it as an assumption

## Worked Example

### Validating "oxygen decreases cancer cell migration"

**Step 1**: Export rules
```python
rules = mcp__PhysiCell__get_rules_for_validation()
# Contains: {"cell_type":"tumor", "signal":"oxygen", "direction":"decreases", "behavior":"migration speed"}
```

**Step 2**: Create collection
```python
mcp__LiteratureValidation__create_paper_collection("tumor_migration")
```

**Step 3**: Get queries
```python
queries = mcp__LiteratureValidation__suggest_search_queries(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="migration speed"
)
# Returns queries about hypoxia-induced migration, HIF and cell motility, etc.
```

**Step 4**: Search PubMed and get abstracts
```python
results = search_pubmed("hypoxia induced cancer cell migration", max_results=10)
# Get details for top results
paper1 = get_article_details(pmid="...")
paper2 = get_article_details(pmid="...")
```

**Step 5**: Index papers
```python
mcp__LiteratureValidation__add_papers_to_collection("tumor_migration", papers=[
    {"title": paper1.title, "text": paper1.abstract, "pmid": paper1.pmid},
    {"title": paper2.title, "text": paper2.abstract, "pmid": paper2.pmid}
])
```

**Step 7**: Validate
```python
result = mcp__LiteratureValidation__validate_rule(
    name="tumor_migration",
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="migration speed"
)
# Result: support_level="strong"
# Evidence: "Multiple studies show hypoxia promotes cancer cell migration via HIF-1α..."
```

**Step 9**: Store
```python
mcp__PhysiCell__store_validation_results(validations=[result])
```

**Step 10**: Report
```python
mcp__PhysiCell__get_validation_report()
```

## Tips

1. **Index more papers** for better validation accuracy — PaperQA2 works better with more context
2. **Include review articles** — they summarize the state of knowledge
3. **Search for both supporting and contradicting evidence** — use queries like "oxygen does NOT affect migration"
4. **Validate early** — it's easier to change rules before the simulation is tuned
5. **Keep collections organized** — one collection per model or biological question
