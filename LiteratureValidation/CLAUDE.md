# Global Instructions

## LiteratureValidation MCP Server

A LiteratureValidation MCP server is connected. It validates PhysiCell cell rules against published biomedical literature using PaperQA2 for RAG-based question answering. Call its tools using the MCP tool calling mechanism.

### Available Tools
- `create_paper_collection(name)` - Create a named collection for literature
- `add_papers_to_collection(name, papers)` - Index papers (abstracts/full text)
- `validate_rule(name, cell_type, signal, direction, behavior, ...)` - Validate a single rule
- `validate_rules_batch(name, rules)` - Validate multiple rules at once
- `get_validation_summary(name)` - Summary of validation results
- `suggest_search_queries(cell_type, signal, direction, behavior)` - Generate PubMed queries

### Validation Workflow
1. Export rules from PhysiCell MCP with `get_rules_for_validation()`
2. For each rule, call `suggest_search_queries()` to get PubMed query suggestions
3. Search PubMed for papers (via PubMed MCP)
4. Call `add_papers_to_collection()` to index paper abstracts
5. Call `validate_rule()` or `validate_rules_batch()` to evaluate rules
6. Save results back to PhysiCell MCP with `store_validation_results()`
