# Literature Research and Validation Guide

Use the LiteratureValidation MCP server to validate PhysiCell cell rules against published literature. This server uses Edison Scientific's PaperQA3 API, which automatically searches 150M+ papers — no manual paper curation, PDF downloads, or indexing needed.

## IMPORTANT: Never read full papers into context

**Do NOT use `get_full_text_article` from any PubMed MCP.** It returns 50KB+ of raw text into the conversation, which wastes context.

**Do NOT manually extract parameters from abstracts or web results.** Let PaperQA do the extraction via `validate_rule()`.

## Architecture

Two MCP servers are involved:

1. **PhysiCell MCP** — Exports rules, stores validation results, generates reports
2. **LiteratureValidation MCP** — Validates rules against 150M+ papers via Edison PaperQA3 API

The LLM orchestrates between both servers. No paper management or additional MCP servers (PubMed, bioRxiv) are needed for validation.

## When to Use

- **Finding parameters from literature** — user says "base on literature", "determine from published data", "use published values"
- **After defining cell rules** — validate assumptions before running simulation
- **For novel biological assumptions** that aren't well-established
- **When rules produce unexpected results** — check if the biology is correct

## Complete Workflow

### Phase 1: Validate Rules

**Step 1: Get rules from PhysiCell session**
```python
mcp__PhysiCell__get_rules_for_validation()
```

**Step 2: Validate all rules**

Validate all rules at once:
```python
mcp__LiteratureValidation__validate_rules_batch(
    name="tumor_model_v1",
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
Edison automatically searches 150M+ papers and returns evidence-based verdicts with citations.

Or validate one at a time (useful for iterating on parameter values):
```python
mcp__LiteratureValidation__validate_rule(
    name="tumor_model_v1",
    cell_type="tumor",
    signal="oxygen",
    direction="decreases",
    behavior="necrosis",
    half_max=3.75,
    hill_power=8
)
```

PaperQA queries 150M+ papers and returns:
- Whether the relationship is supported
- Evidence summary with citations
- Whether proposed parameter values are consistent with published data

**Step 3: Get validation summary**
```python
mcp__LiteratureValidation__get_validation_summary("tumor_model_v1")
```

### Phase 2: Persist Results in PhysiCell Session

**Step 4: Store results**
```python
mcp__PhysiCell__store_validation_results(validations=[
    {
        "cell_type": "tumor",
        "signal": "oxygen",
        "direction": "decreases",
        "behavior": "necrosis",
        "collection_name": "tumor_model_v1"
    }
])
```
The server reads PaperQA answer files directly from disk — VERDICT and DIRECTION are extracted server-side. Direction mismatches are auto-flagged as `contradictory`.

**Step 5: Generate report**
```python
mcp__PhysiCell__get_validation_report()
```

## Support Level Interpretation

| Level | Meaning | Action |
|-------|---------|--------|
| **strong** | Multiple papers directly support this relationship | Keep as-is, consider using suggested parameters |
| **moderate** | Some supporting evidence, possibly indirect | Acceptable, note limitations |
| **weak** | Limited or tangential evidence | Consider if the relationship is critical; add caveats |
| **contradictory** | Literature shows opposite or conflicting results | Review carefully; may need to remove or reverse the rule |
| **unsupported** | No evidence found | Not necessarily wrong, but needs justification |

## Acting on Validation Results

### For `strong` or `moderate` support
- Keep the rule
- Consider updating `half_max` and `hill_power` if the validation suggests specific values based on dose-response data

### For `weak` support
- Document the assumption
- Consider whether the rule is essential to the model

### For `contradictory` support
- The contradiction may be context-dependent (different cell type, species, conditions)
- Consider reversing the direction or removing the rule
- Document the decision and rationale

### For `unsupported` support
- Not the same as "disproven" — may just mean the evidence is sparse
- Consider if the relationship is derived from first principles or expert knowledge
- If keeping, explicitly document it as an assumption

## Tips

1. **Validate early** — it's easier to change rules before the simulation is tuned
2. **Include `signal_units`** when calling `validate_rule()` — prevents unit confusion (e.g., "mmHg" for oxygen)
3. **Search for both supporting and contradicting evidence** — Edison searches broadly but DIRECTION/VERDICT parsing catches mismatches
4. **Review all rules** — even "obvious" relationships may have nuances in the literature
