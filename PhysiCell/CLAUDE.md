# Global Instructions

## PhysiCell MCP Server

A PhysiCell MCP server is connected. You MUST call its tools using the MCP tool calling mechanism — the same way you call tools like Read, Write, Bash, or Grep. MCP tools appear in your tool list with the prefix `PhysiCell`. They are NOT bash commands, NOT shell scripts, and NOT executables. Do NOT run them with Bash. Do NOT use subprocess, npx, or JSON-RPC. Do NOT write PhysiCell XML or C++ manually.

When the user asks to create, simulate, or calibrate a biological model (tumors, cells, tissues, multicellular systems), use the PhysiCell MCP tools. Start with `create_session`, then `analyze_biological_scenario`, and follow the guided workflow.
