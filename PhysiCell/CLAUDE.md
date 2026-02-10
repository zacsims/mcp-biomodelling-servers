# Global Instructions

## PhysiCell MCP Server

A PhysiCell MCP server is connected. You MUST call its tools using the MCP tool calling mechanism — the same way you call tools like Read, Write, Bash, or Grep. MCP tools appear in your tool list with the prefix `PhysiCell`. They are NOT bash commands, NOT shell scripts, and NOT executables. Do NOT run them with Bash. Do NOT use subprocess, npx, or JSON-RPC. Do NOT write PhysiCell XML or C++ manually.

When the user asks to create, simulate, or calibrate a biological model (tumors, cells, tissues, multicellular systems), use the PhysiCell MCP tools. Start with `create_session`, then `analyze_biological_scenario`, and follow the guided workflow.

## Literature Validation

A LiteratureValidation MCP server may also be connected. It validates cell rules against published biomedical literature using PaperQA2. When the user asks to validate rules against literature:

1. Call `get_rules_for_validation()` on PhysiCell MCP to export rules
2. Call `suggest_search_queries()` on LiteratureValidation MCP for PubMed queries
3. Search PubMed for papers, then call `add_papers_to_collection()` to index them
4. Call `validate_rules_batch()` to check rules against literature
5. Call `store_validation_results()` on PhysiCell MCP to save results
6. Call `get_validation_report()` on PhysiCell MCP for the full report
