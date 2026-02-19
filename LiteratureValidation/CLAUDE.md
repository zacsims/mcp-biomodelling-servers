# Global Instructions

## LiteratureValidation MCP Server

A LiteratureValidation MCP server is connected. It validates PhysiCell cell rules against published biomedical literature using Edison Scientific's PaperQA3 API, which automatically searches 150M+ papers. Call its tools using the MCP tool calling mechanism.

### Available Tools
- `validate_rule(name, cell_type, signal, direction, behavior, ...)` - Validate a single rule against literature
- `validate_rules_batch(name, rules)` - Validate multiple rules at once
- `get_validation_summary(name)` - Summary of validation results

### Validation Workflow
1. Export rules from PhysiCell MCP with `get_rules_for_validation()`
2. Call `validate_rules_batch(name, rules)` — Edison automatically searches 150M+ papers
3. Call `get_validation_summary(name)` to review results
4. Save results back to PhysiCell MCP with `store_validation_results()`
5. Call `get_validation_report()` to generate the formal report

No paper collection management is needed — Edison handles paper discovery automatically.
