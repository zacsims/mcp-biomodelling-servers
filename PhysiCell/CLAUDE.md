# Global Instructions

## PhysiCell MCP Server

A PhysiCell MCP server is connected and its tools are available as direct tool calls (e.g., `create_session()`, `add_single_cell_type()`, etc.). **Never** try to call them via subprocess, npx, JSON-RPC, or by writing PhysiCell XML/C++ manually.

When the user asks to create, simulate, or calibrate a biological model (tumors, cells, tissues, multicellular systems), use the PhysiCell MCP tools directly. Start with `create_session()`, then `analyze_biological_scenario()`, and follow the guided workflow.
