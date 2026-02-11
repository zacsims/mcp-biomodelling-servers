#!/usr/bin/env python3
"""Install PhysiCell MCP server in Claude Desktop."""

import json
import os
from pathlib import Path

def install():
    # Claude Desktop config location
    config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"

    # Server location
    server_dir = Path(__file__).parent.resolve()
    server_path = server_dir / "server.py"

    # Load existing config or create new
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add PhysiCell server
    config["mcpServers"]["PhysiCell"] = {
        "command": "uv",
        "args": [
            "run",
            "--project",
            str(server_dir),
            "python",
            str(server_path)
        ]
    }

    # Write config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Installed PhysiCell MCP server in Claude Desktop")
    print(f"  Server: {server_path}")
    print(f"  Config: {config_path}")
    print(f"\nRestart Claude Desktop to use the server.")

if __name__ == "__main__":
    install()
