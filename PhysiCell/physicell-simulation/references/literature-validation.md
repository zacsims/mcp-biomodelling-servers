# Literature Research and Validation Guide

Use the LiteratureValidation MCP server to find parameters from published literature and validate PhysiCell cell rules. This server downloads and indexes papers (as PDFs when available) into PaperQA2, then uses RAG to answer questions about biological relationships and parameter values.

## IMPORTANT: Never read full papers into context

**Do NOT use `get_full_text_article` from the PubMed MCP.** It returns 50KB+ of raw text into the conversation, which wastes context and is far less effective than PaperQA's chunked retrieval.

**Do NOT use `get_article_metadata` to manually extract parameters from abstracts.** Let PaperQA do the extraction via `validate_rule()`.

Instead, use `add_papers_by_id()` to download PDFs and index them in PaperQA, then query with `validate_rule()`.

## Architecture

Four MCP servers are involved:

1. **PhysiCell MCP** — Exports rules, stores validation results, generates reports
2. **LiteratureValidation MCP** — Downloads PDFs, indexes papers in PaperQA2, validates rules against literature
3. **PubMed MCP** (`plugin_pubmed_PubMed`) — Searches PubMed, retrieves PMIDs and basic metadata
4. **bioRxiv MCP** (`plugin_biorxiv_bioRxiv`) — Searches bioRxiv/medRxiv preprints by category and date

The LLM orchestrates between all four servers.

## When to Use

- **Finding parameters from literature** — user says "base on literature", "determine from published data", "use published values"
- **After defining cell rules** — validate assumptions before running simulation
- **For novel biological assumptions** that aren't well-established
- **When rules produce unexpected results** — check if the biology is correct

## Complete Workflow

### Phase 1: Build a Paper Collection

**Step 1: Create collection**
```python
mcp__LiteratureValidation__create_paper_collection("tumor_oxygen_model")
```

**Step 2: Get search queries**
```python
mcp__LiteratureValidation__suggest_search_queries(
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis"
)
```
Returns optimized PubMed queries AND suggested bioRxiv categories.

**Step 3: Search PubMed**
```python
mcp__plugin_pubmed_PubMed__search_articles(
    query="hypoxia-induced necrosis cancer cells",
    max_results=10
)
```
Collect PMIDs from the search results.

**Step 4: Add papers by PMID with PDF download**
```python
mcp__LiteratureValidation__add_papers_by_id(
    name="tumor_oxygen_model",
    pmids=["35486828", "33264437", "28558982"],
    fetch_pdfs=True
)
```
This automatically:
- Fetches title, abstract, authors, year from NCBI
- Downloads full PDFs via metapub FindIt (68+ publishers) and Unpaywall
- Skips papers without available PDFs (no abstract-only fallback)
- Indexes PDFs in PaperQA

**Step 5 (optional): Search bioRxiv**
```python
mcp__plugin_biorxiv_bioRxiv__search_preprints(
    category="cancer biology",
    recent_days=90,
    limit=20
)
```
Note: bioRxiv uses category + date filtering, not keyword search.

**Step 6 (optional): Add bioRxiv preprints**
```python
mcp__LiteratureValidation__add_papers_by_id(
    name="tumor_oxygen_model",
    biorxiv_dois=["10.1101/2024.01.15.123456"]
)
```
bioRxiv PDFs are always available (100% success rate).

**Repeat** steps 2-6 for each biological relationship you need to investigate.

### Phase 2: Query the Literature

**Step 7: Validate rules**

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

Or validate one at a time (useful for iterating on parameter values):
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

PaperQA queries the indexed papers and returns:
- Whether the relationship is supported
- Evidence summary with citations
- Whether proposed parameter values are consistent with published data

**Step 8: Get validation summary**
```python
mcp__LiteratureValidation__get_validation_summary("tumor_oxygen_model")
```

### Phase 3: Persist Results in PhysiCell Session

**Step 9: Store results**
```python
mcp__PhysiCell__store_validation_results(validations=[
    {
        "cell_type": "tumor",
        "signal": "oxygen",
        "direction": "decreases",
        "behavior": "necrosis",
        "collection_name": "tumor_oxygen_model"
    }
])
```
The server reads PaperQA answer files directly from disk — VERDICT and DIRECTION are extracted server-side. Direction mismatches are auto-flagged as `contradictory`.

**Step 10: Generate report**
```python
mcp__PhysiCell__get_validation_report()
```

## Alternative: Add Papers with Metadata

If you already have paper metadata (e.g., from PubMed search results), you can use `add_papers_to_collection` directly:

```python
mcp__LiteratureValidation__add_papers_to_collection(
    name="tumor_oxygen_model",
    papers=[
        {
            "title": "Hypoxia-induced necrosis in solid tumors",
            "text": "Abstract text...",
            "pmid": "12345678",
            "doi": "10.1234/example",
            "authors": "Smith et al.",
            "year": "2023"
        }
    ],
    fetch_pdfs=True  # Download PDF; skip if unavailable
)
```

Only papers with downloadable full PDFs are indexed. Papers without available PDFs are skipped entirely.

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
- The contradiction may be context-dependent (different cell type, species, conditions)
- Consider reversing the direction or removing the rule
- Document the decision and rationale

### For `unsupported` support
- Not the same as "disproven" — may just mean no papers were indexed
- Try broader search queries
- Consider if the relationship is derived from first principles or expert knowledge
- If keeping, explicitly document it as an assumption

## Tips

1. **Use `add_papers_by_id` with `fetch_pdfs=True`** — PDFs contain far more information than abstracts alone
2. **Index 5-10+ papers per rule** for better validation accuracy — PaperQA2 works better with more context
3. **Include review articles** — they summarize the state of knowledge
4. **Search for both supporting and contradicting evidence** — use queries like "oxygen does NOT affect migration"
5. **Validate early** — it's easier to change rules before the simulation is tuned
6. **Keep collections organized** — one collection per model or biological question
7. **Combine PubMed and bioRxiv** — bioRxiv has the latest research (preprints), PubMed has peer-reviewed articles
