# MCP Bio‑Modelling Servers

This repository centralizes **Model Context Protocol (MCP)** servers that wrap Python‑based *mechanistic / systems biology* modelling tools. Each subfolder contains a `server.py` entrypoint plus a README describing the specific tool interface.

Current servers (see their own READMEs & upstream docs):

| Tool | Folder | Upstream Documentation |
|------|--------|------------------------|
| MaBoSS | `MaBoSS/` | https://github.com/colomoto/pyMaBoSS |
| NeKo | `NeKo/` | https://github.com/sysbio-curie/Neko |
| PhysiCell (settings wrapper) | `PhysiCell/` | https://github.com/marcorusc/PhysiCell_Settings |
| Literature Validation (PaperQA2) | `LiteratureValidation/` | https://github.com/Future-House/paper-qa |

All servers are Python processes speaking MCP over stdio.

---
## MCP Background
The Model Context Protocol standardizes how external tools expose *tools* and *resources* to AI assistants / IDEs. Spec & introduction: https://modelcontextprotocol.io/docs/getting-started/intro

Each `server.py` advertises modelling actions (e.g. run simulations, manage sessions) to any MCP‑aware client (e.g. VS Code with GitHub Copilot Chat MCP support).

---
## Repository Layout
```
MaBoSS/                # MaBoSS MCP server (Boolean / stochastic models)
NeKo/                  # NeKo MCP server
PhysiCell/             # PhysiCell settings / sessions MCP server
LiteratureValidation/  # Literature validation MCP server (PaperQA2)
.claude/agents/        # Claude Code subagents built on top of the MCP servers
README.md
```
Consult the README within each tool folder for: purpose, required Python packages, and any model/data file expectations. Installation instructions for the modelling tools themselves live *there* (or in the upstream project links above) — they are intentionally not duplicated here.

---
## Environment Assumption
All tools are Python‑based. Create (and manage) a single Conda environment that contains the dependencies for MaBoSS, NeKo, and PhysiCell. The exact creation commands are up to you (not prescribed here). Once created, note the absolute path to its Python interpreter (e.g. `/home/you/miniforge3/envs/mcp_modelling/bin/python`).

---
## Configure in VS Code (GitHub Copilot Chat / MCP)
1. Clone this repo somewhere stable (no spaces in path recommended).
2. Open VS Code.
3. Ensure the Copilot Chat (or other MCP‑capable) extension is installed and updated.
4. Press `Ctrl + Shift + P` and search for the command that opens the MCP configuration JSON (e.g. “MCP: Open Configuration” or locate `mcp.json`). On Linux it typically lives at: `~/.config/Code/User/mcp.json`.
5. Add entries pointing to each `server.py` using the Conda environment’s Python.

Example (adapt paths to your system; based on the working setup):

```jsonc
{
  "servers": {
    "maboss": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/MaBoSS/server.py"
      ],
      "env": {
        "PATH": "/home/you/miniforge3/envs/mcp_modelling/bin:${Path}",
        "CONDA_PREFIX": "/home/you/miniforge3/envs/mcp_modelling"
      }
    },
    "neko": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/NeKo/server.py"
      ]
    },
    "physicell": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/PhysiCell/server.py"
      ]
    },
    "literature_validation": {
      "type": "stdio",
      "command": "/home/you/miniforge3/envs/mcp_modelling_py311/bin/python",
      "args": [
        "/absolute/path/to/mcp-biomodelling-servers/LiteratureValidation/server.py"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  },
  "inputs": [
    {
      "id": "memory_file_path",
      "type": "promptString",
      "description": "Path to the memory storage file",
      "password": false
    }
  ]
}
```

Notes:
- Replace `/home/you/...` and `/absolute/path/to/...` with your actual directories.
- Keep all three servers referencing the *same* Conda interpreter to share installed libraries.
- Add further environment variables (e.g. data directories) per tool README if required.

After saving, reload / restart VS Code so the MCP client reconnects.

Activation / usage guidance in VS Code: https://code.visualstudio.com/docs/copilot/chat/mcp-servers

You should then see the servers’ tools listed in the Copilot Chat “/tools” (or similar) UI. Invoke them by name with required parameters.

---
## Claude Code Agents

In addition to the raw MCP servers, this repo ships a set of **Claude Code subagents** that compose the MCP tools into focused, task-specific assistants for agentic PhysiCell model construction, validation, calibration, UQ, and spatial analysis.

See [`.claude/agents/README.md`](.claude/agents/README.md) for the full roster, workflow, and configuration notes.

Quick roster:

| Agent | Role |
|---|---|
| `model-constructor` | Builds a PhysiCell ABM end-to-end from a biological scenario with literature-backed justifications. |
| `literature-rule-validator` | Audits rules against literature; classifies evidentiary strength. |
| `spatial-analysis` | Analyzes spatial organization (imaging data or sim output). |
| `parameter-calibration` | ABC / Bayesian calibration against experimental data. |
| `uq` | Sensitivity analysis and uncertainty quantification. |

These are Claude-Code–specific; other MCP clients (VS Code Copilot Chat, etc.) do not consume `.claude/agents/` and should continue to use the MCP servers directly.

### Agentic Install

Rather than working through the install manually, you can have Claude Code drive the entire setup. Paste one of the prompts below into a Claude Code session opened at the root of this repo — it uses Bash, `uv`, and `claude mcp add` directly.

**Prerequisites Claude cannot do for you:**
- Install [Claude Code](https://docs.claude.com/claude-code) itself.
- Get an [Edison Scientific API key](https://platform.edisonscientific.com/profile) for LiteratureValidation. Put it in your shell profile (`~/.zshrc` or `~/.bashrc`) so Claude Code inherits it:
  ```bash
  export EDISON_PLATFORM_API_KEY="your-key"
  ```

#### One-shot prompt — install everything

```
Install the full bio-modelling MCP stack for this repo. Work from the repo root.

1. Verify `uv` is installed. If not, install it via
   `curl -LsSf https://astral.sh/uv/install.sh | sh` and source the profile.
2. If $HOME/PhysiCell does not exist, clone https://github.com/MathCancer/PhysiCell.git
   to $HOME/PhysiCell and run `make` there. Report any build errors verbatim
   — do not silently continue on failure.
3. Register the PhysiCell MCP server at user scope:
     claude mcp add PhysiCell -s user -- uv run \
       --project $(pwd)/PhysiCell python $(pwd)/PhysiCell/server.py
4. For LiteratureValidation: confirm $EDISON_PLATFORM_API_KEY is set (fail loudly
   if not). Run `uv sync` inside ./LiteratureValidation, then register:
     claude mcp add LiteratureValidation -s user \
       -e EDISON_PLATFORM_API_KEY=$EDISON_PLATFORM_API_KEY \
       -- uv run --project $(pwd)/LiteratureValidation \
       python $(pwd)/LiteratureValidation/server.py
5. Install and register spatialtissuepy:
     uv tool install "spatialtissuepy[mcp,viz] @ git+https://github.com/emcramer/spatialtissuepy"
     claude mcp add spatialtissuepy -s user -- \
       spatialtissuepy-mcp --data-dir $HOME/tissue_data
6. Run `claude mcp list` and report which servers are ✓ Connected.
7. Tell me to restart Claude Code, then run `/agents` to see the five subagents
   shipped in .claude/agents/.

Stop and report immediately if any step fails. Do not retry failing commands
in a loop; diagnose the root cause.
```

#### Piecewise prompts — install one server at a time

Use these if the one-shot run hits a snag and you want to troubleshoot a single server.

**PhysiCell** (assumes `uv` is installed):
```
Compile PhysiCell (clone https://github.com/MathCancer/PhysiCell.git to
$HOME/PhysiCell if missing, then run `make` in that directory). Then register
the PhysiCell MCP:
  claude mcp add PhysiCell -s user -- uv run \
    --project $(pwd)/PhysiCell python $(pwd)/PhysiCell/server.py
Verify with `claude mcp list`.
```

**LiteratureValidation** (requires `EDISON_PLATFORM_API_KEY` exported in the parent shell):
```
Run `uv sync` inside ./LiteratureValidation. Then register the MCP server,
passing the Edison API key as an environment variable:
  claude mcp add LiteratureValidation -s user \
    -e EDISON_PLATFORM_API_KEY=$EDISON_PLATFORM_API_KEY \
    -- uv run --project $(pwd)/LiteratureValidation \
    python $(pwd)/LiteratureValidation/server.py
Fail loudly if $EDISON_PLATFORM_API_KEY is not set. Verify with `claude mcp list`.
```

**spatialtissuepy**:
```
Install the spatialtissuepy tool (from GitHub) and register its MCP server:
  uv tool install "spatialtissuepy[mcp,viz] @ git+https://github.com/emcramer/spatialtissuepy"
  claude mcp add spatialtissuepy -s user -- \
    spatialtissuepy-mcp --data-dir $HOME/tissue_data
Verify with `claude mcp list`.
```

#### After install

Restart Claude Code, then run `/agents` — you should see `model-constructor`, `literature-rule-validator`, `spatial-analysis`, `parameter-calibration`, and `uq`. Kick the tires with something like *"Build a PhysiCell model of breast cancer cells in a hypoxic 3D environment with immune infiltration"* — Claude Code should dispatch to `model-constructor`.

---
## Adding Another Server
1. Create a new folder with `server.py` and a README describing the underlying modelling tool and dependencies.
2. Follow existing server structure for registering MCP tools.
3. Update your `mcp.json` with a new block (use the same Conda Python path).
4. Document any additional env vars in that folder README.

---
## License
Project is MIT (see existing LICENSE file). Underlying tools retain their own licenses — consult upstream repositories.

---
## Quick Reference
| Action | What to Do |
|--------|-----------|
| Get tool install steps | Open the tool’s subfolder README or upstream link |
| Ensure deps present | Install into your chosen Conda env (user‑defined) |
| Configure MCP | Edit `~/.config/Code/User/mcp.json` as above |
| Reload servers | Reload VS Code window |
| Learn MCP | Spec: modelcontextprotocol.io; VS Code guide link above |

---
Happy modelling!
