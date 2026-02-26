# Literature Search Guide

Use the LiteratureValidation MCP server's `search_literature()` tool to ask any biological question during model building. This server uses Edison Scientific's PaperQA3 API, which automatically searches 150M+ papers — no manual paper curation, PDF downloads, or indexing needed.

## Architecture

Two MCP servers work together:

1. **LiteratureValidation MCP** — `search_literature(query)` → returns evidence-based answers with citations
2. **PhysiCell MCP** — `store_rule_justification()` → records evidence basis for each rule; `get_rule_justifications()` → generates report

The LLM orchestrates between both servers during model building.

## When to Use

**Always, before adding cell rules.** This is the default workflow for every simulation:
- **Before choosing half_max, hill_power, or rate values** — search for evidence to inform parameter choices
- **For every rule** — get citations to justify modeling decisions
- **When rules produce unexpected results** — check if the biology is correct

Do NOT skip literature search because:
- You "already know" the values from training data (training data may be wrong or outdated)
- The user didn't explicitly ask for literature (it's always the default)
- You want to save time (the call takes seconds and results are cached)

## Workflow

### 1. Ask biological questions

Use `search_literature()` to ask any question:

```python
# Mechanisms
mcp__LiteratureValidation__search_literature(query="What is the effect of oxygen on tumor cell necrosis?")

# Parameter values
mcp__LiteratureValidation__search_literature(query="What oxygen concentration causes half-maximal necrosis in solid tumors?")

# Cell behaviors
mcp__LiteratureValidation__search_literature(query="What is a typical migration speed for breast cancer cells?")

# Hill function parameters
mcp__LiteratureValidation__search_literature(query="What is the Hill coefficient for oxygen-dependent cell cycle arrest?")
```

Results are cached — repeated queries return instantly.

### 2. Add rules informed by evidence

Use the answers to choose parameter values, then add the rule via PhysiCell MCP.

### 3. Record justification

After adding each rule:

```python
mcp__PhysiCell__store_rule_justification(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis",
    justification="Tumor cells undergo necrosis below ~5 mmHg O2. Half-max of 3.75 mmHg consistent with Vaupel 2004.",
    key_citations="Vaupel 2004, Grimes 2014"
)
```

### 4. Generate report

After all rules are added:

```python
mcp__PhysiCell__get_rule_justifications()
```

This flags any unjustified rules and exports to `rule_justifications.md`.

## Tips

1. **Search before adding rules** — it's easier to choose good parameters upfront than to revise later
2. **Ask specific questions** — "What is the EC50 for oxygen-dependent necrosis?" beats "Tell me about oxygen and necrosis"
3. **Include units context** — mention units in your query (e.g., "in mmHg", "in mM") to get relevant quantitative answers
4. **Use collection_name for organization** — group related queries (e.g., `collection_name="tumor_model_v1"`)
5. **PubMed/bioRxiv for browsing** — use those MCPs to discover papers and topics; use Edison (`search_literature`) for deep evidence search with citations
