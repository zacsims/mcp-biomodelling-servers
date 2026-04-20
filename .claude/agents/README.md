# Claude Code Agents for Bio-Modelling

This directory contains **Claude Code subagents** — specialized agent definitions that wrap the MCP servers in this repo (PhysiCell, LiteratureValidation, spatialtissuepy) into focused, task-specific assistants for building, validating, calibrating, and analyzing agent-based biological models.

These are *not* MCP tools themselves. They are Claude Code agent prompts that consume the MCP tools. They live under `.claude/agents/` so the Claude Code CLI auto-discovers them when invoked from the repo.

---

## Roster

| Agent | Role | Primary MCP servers |
|---|---|---|
| [`model-constructor`](model-constructor.md) | Builds a complete PhysiCell ABM end-to-end from a biological scenario — session, domain, substrates, cell types, mechanics, rules (with literature-backed justifications), PhysiBoSS, initial conditions, XML export. | PhysiCell, LiteratureValidation |
| [`literature-rule-validator`](literature-rule-validator.md) | Independent audit of existing PhysiCell rules against published literature. Classifies each rule as supported / weakly supported / contradicted / unsupported. Read-mostly. | PhysiCell, LiteratureValidation |
| [`spatial-analysis`](spatial-analysis.md) | Spatial organization analysis of cells in multiplexed imaging data or PhysiCell simulation output — clustering, colocalization, neighborhoods, hotspots, topology, ML-ready features. | spatialtissuepy, PhysiCell |
| [`parameter-calibration`](parameter-calibration.md) | Calibrates PhysiCell parameters against experimental data via ABC or Bayesian methods. Produces posteriors, applies calibrated values. | PhysiCell |
| [`uq`](uq.md) | Uncertainty quantification and sensitivity analysis. Sobol indices with convergence discipline; identifies which parameters are worth calibrating. | PhysiCell |

All five agents use `model: opus` — the work is reasoning-heavy (literature synthesis, model architecture, statistical interpretation) and benefits from the strongest model.

---

## Typical workflow

The agents are designed to compose. A full end-to-end run looks like:

```
biological scenario
        │
        ▼
┌────────────────────┐
│ model-constructor  │  builds model + rules with literature justifications
└────────┬───────────┘
         ├──────────────────┐
         ▼                  ▼
┌─────────────────────┐   ┌──────────────────────┐
│ literature-rule-    │   │ uq                   │  identifies influential
│ validator           │   │ (sensitivity)        │  parameters
└─────────────────────┘   └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ parameter-calibration│  fits influential params
                          └──────────┬───────────┘   to experimental data
                                     ▼
                          ┌──────────────────────┐
                          │ spatial-analysis     │  analyzes simulation output
                          └──────────────────────┘
```

Not every run needs every agent. Common subsets:
- **Build + validate only**: `model-constructor` → `literature-rule-validator`. Produces a defensible baseline model.
- **Calibrate existing model**: `uq` → `parameter-calibration`. Skip UQ if the parameter set is already small.
- **Analyze existing data**: `spatial-analysis` standalone on imaging data or sim output.

---

## Design principles

### Evidence discipline (constructor + validator)
Every non-default parameter and every rule must be backed by a literature citation stored via `store_rule_justification`. The constructor writes justifications as it builds; the validator audits and refines them.

### State-based handoff
Agents hand off work through PhysiCell's session state (`get_rule_justifications`, `list_loaded_components`, etc.), not through ad-hoc files. This keeps the system's single source of truth inside the MCP server's session, not scattered across the filesystem.

### Narrow tool allowlists
Each agent's `tools:` frontmatter is explicit and minimal for its job. The validator cannot call `add_single_cell_rule`; calibration cannot define UQ runs beyond what it needs. This prevents scope creep and makes agent behavior predictable.

### Autonomous construction
`model-constructor` calls the configuration MCP tools directly rather than emitting a plan for review. This is the intended mode — the value proposition is end-to-end construction. If you want a draft-first flow, override the system prompt.

### Statistical honesty
`parameter-calibration` and `uq` are prompted to surface non-identifiability, convergence problems, and posterior correlations explicitly, not paper over them with point estimates.

---

## Prerequisites

1. **Claude Code installed** — these are Claude Code subagent files. Other MCP-aware clients (e.g. VS Code Copilot Chat) do not consume `.claude/agents/`.
2. **All three MCP servers configured in Claude Code**:
   - `physicell` — see [`PhysiCell/README.md`](../../PhysiCell/README.md)
   - `LiteratureValidation` — see [`LiteratureValidation/README.md`](../../LiteratureValidation/README.md) (requires `ANTHROPIC_API_KEY`)
   - `spatialtissuepy` — the companion server referenced in `PhysiCell/README.md`
3. **Conda environment** with dependencies for all three servers, same interpreter path.

A minimal Claude Code MCP config (`~/.claude.json` or project `.mcp.json`):

```jsonc
{
  "mcpServers": {
    "physicell": {
      "type": "stdio",
      "command": "/path/to/envs/mcp_modelling/bin/python",
      "args": ["/abs/path/to/mcp-biomodelling-servers/PhysiCell/server.py"]
    },
    "LiteratureValidation": {
      "type": "stdio",
      "command": "/path/to/envs/mcp_modelling/bin/python",
      "args": ["/abs/path/to/mcp-biomodelling-servers/LiteratureValidation/server.py"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    },
    "spatialtissuepy": {
      "type": "stdio",
      "command": "/path/to/envs/mcp_modelling/bin/python",
      "args": ["/abs/path/to/spatialtissuepy/server.py"]
    }
  }
}
```

---

## Invoking agents

From a Claude Code session rooted in this repo, agents are auto-discovered. Invoke them via the `Agent` tool by name:

- `Agent({ subagent_type: "model-constructor", prompt: "Build a PhysiCell model of ..." })`
- `Agent({ subagent_type: "literature-rule-validator", prompt: "Audit the rules in session <id>." })`
- `Agent({ subagent_type: "spatial-analysis", prompt: "Characterize clustering of T cells in the loaded multiplex dataset." })`

Each agent starts with no memory of the parent conversation — briefing it well matters. Include the session ID, scenario description, and specific goal.

---

## Extending

To add a new agent, create `<name>.md` in this directory with frontmatter:

```yaml
---
name: agent-name
description: One-line description shown when Claude Code picks agents.
model: opus | sonnet | haiku | inherit
tools: Tool1, Tool2, ...
---

System prompt body.
```

- Keep tool lists narrow — broad allowlists erode the reason for having a subagent.
- The `description` field is what other agents (and Claude Code itself) use to decide when to dispatch — write it for matching, not marketing.
- If a new agent needs to hand off work, prefer MCP-server state over filesystem artifacts.

---

## Caveats

- **PhysiCell MCP server session scope**: agents operate on the current default session. If multiple agents run concurrently against the same session, they will see each other's writes. Use separate sessions for parallel experiments.
- **Literature calls are not free**: `search_literature` hits the PaperQA3 API and costs Anthropic tokens. The constructor and validator are designed to ask specific questions, not scattershot queries, but expect real cost on large rule sets.
- **Autonomous construction trusts the scenario description**: garbage-in, garbage-out. A vague "build a tumor model" will produce a generic result. Scenario descriptions should specify cell types, key interactions, tissue context, and what the model needs to reproduce.
