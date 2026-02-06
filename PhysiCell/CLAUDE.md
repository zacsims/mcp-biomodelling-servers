# Global Instructions

## PhysiCell MCP Server

A PhysiCell MCP server named "PhysiCell" is connected. Its tools are available as MCP tool calls prefixed with `mcp__PhysiCell__` (e.g., `mcp__PhysiCell__create_session`, `mcp__PhysiCell__add_single_cell_type`). **Never** try to call them via subprocess, npx, JSON-RPC, or by writing PhysiCell XML/C++ manually.

When the user asks to create, simulate, or calibrate a biological model (tumors, cells, tissues, multicellular systems), use the PhysiCell MCP tools directly. Start with `create_session()`, then `analyze_biological_scenario()`, and follow the guided workflow.
