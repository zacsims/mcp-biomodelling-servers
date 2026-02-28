# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastmcp>=2.14.4",
#   "spatialtissuepy[viz,network]>=0.2.0",
# ]
# ///
"""
SpatialTissue MCP Server

Spatial statistics panel analysis on PhysiCell simulation output.
Exposes 10 tools for building/running composable metric panels and
three complex analysis families: Spatial LDA, cell-graph network
analysis, and topological Mapper.

Install the full skill for the metric catalogue:
    /spatial-analysis  (Claude Code skill)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — import shared artifact_manager from repo root
# ---------------------------------------------------------------------------
_SERVER_ROOT = Path(__file__).parent
sys.path.insert(0, str(_SERVER_ROOT))         # for session_manager
sys.path.insert(0, str(_SERVER_ROOT.parent))  # for artifact_manager
from artifact_manager import get_artifact_dir, safe_artifact_path, write_session_meta

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------
from session_manager import (  # noqa: E402
    PanelEntry,
    ensure_session,
    get_current_session,
    session_manager,
)

# ---------------------------------------------------------------------------
# Lazy-load spatialtissuepy (mirrors PhysiCell pcdl pattern)
# ---------------------------------------------------------------------------
_spatialtissuepy_checked = False
SPATIALTISSUEPY_AVAILABLE = False
_PhysiCellSimulation = None
_StatisticsPanel = None
_load_panel = None   # spatialtissuepy.summary.load_panel
_SpatialLDA = None
_SpatialMapper = None
_CellGraph = None
_network_module = None  # full spatialtissuepy.network module


def _ensure_spatialtissuepy() -> None:
    global _spatialtissuepy_checked, SPATIALTISSUEPY_AVAILABLE
    global _PhysiCellSimulation, _StatisticsPanel, _load_panel
    global _SpatialLDA, _SpatialMapper, _CellGraph, _network_module

    if _spatialtissuepy_checked:
        return
    _spatialtissuepy_checked = True

    try:
        from spatialtissuepy.synthetic import PhysiCellSimulation as _PC
        from spatialtissuepy.summary import StatisticsPanel as _SP
        from spatialtissuepy.summary import load_panel as _lp
        from spatialtissuepy.lda import SpatialLDA as _LDA
        from spatialtissuepy.topology import SpatialMapper as _SM
        from spatialtissuepy.network import CellGraph as _CG
        import spatialtissuepy.network as _net

        _PhysiCellSimulation = _PC
        _StatisticsPanel = _SP
        _load_panel = _lp
        _SpatialLDA = _LDA
        _SpatialMapper = _SM
        _CellGraph = _CG
        _network_module = _net
        SPATIALTISSUEPY_AVAILABLE = True
    except ImportError as exc:
        SPATIALTISSUEPY_AVAILABLE = False
        import logging
        logging.getLogger(__name__).warning(f"spatialtissuepy not available: {exc}")


_INSTALL_MSG = (
    "## Error: spatialtissuepy Not Installed\n\n"
    "Install the library and its extras:\n\n"
    "```bash\npip install spatialtissuepy[viz,network]\n```\n"
)

# ---------------------------------------------------------------------------
# Matplotlib (optional — only needed for plot generation)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("SpatialTissue")

# ---------------------------------------------------------------------------
# Agent manual
# ---------------------------------------------------------------------------
_MANUAL = """\
# SpatialTissue MCP Server — Agent Manual

## Overview
10 tools for composable spatial statistics on PhysiCell simulation output.

## Quick workflow
1. `create_session` — start a new analysis session
2. `load_preset_panel` OR `add_metric` (repeat) — build your metric panel
3. `view_panel` — confirm the panel configuration
4. `run_panel` — execute all metrics across every timestep
5. (optional) `run_spatial_lda`, `run_network_analysis`, `run_spatial_mapper`

## input_folder
Pass the PhysiCell simulation `output/` folder path (contains `output*.xml` files).
Obtain it from `mcp__PhysiCell__get_simulation_status()` or `create_physicell_project()`.

## Full capability catalogue
Invoke the `/spatial-analysis` Claude Code skill for the complete metric reference.
"""


@mcp.prompt()
def spatial_analysis_manual() -> str:
    """Agent manual for the SpatialTissue MCP server."""
    return _MANUAL


@mcp.resource("resource://spatial-tissue/manual")
def spatial_analysis_manual_resource() -> str:
    """Agent manual (resource version)."""
    return _MANUAL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """ISO-style timestamp string safe for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _parse_json_list(raw: Optional[str], name: str) -> Optional[List[Any]]:
    """Parse a JSON array string; return None on empty/null input."""
    if not raw:
        return None
    try:
        val = json.loads(raw)
        if not isinstance(val, list):
            raise ValueError(f"{name} must be a JSON array")
        return val
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


def _parse_json_dict(raw: Optional[str], name: str) -> Dict[str, Any]:
    """Parse a JSON object string; return empty dict on empty/null input."""
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        if not isinstance(val, dict):
            raise ValueError(f"{name} must be a JSON object")
        return val
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


def _format_params(params: Dict[str, Any]) -> str:
    return json.dumps(params) if params else "{}"


def _trend(first: float, last: float) -> str:
    """Return a trend arrow for a metric value pair."""
    try:
        if abs(first) < 1e-10:
            return "~" if abs(last) < 1e-10 else ("↑" if last > 0 else "↓")
        ratio = last / first
        if ratio > 1.05:
            return "↑"
        if ratio < 0.95:
            return "↓"
        return "~"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_panel_timeseries(df: Any, panel: List[PanelEntry]) -> Any:
    """Generate a multi-subplot time series figure from panel results DataFrame."""
    import pandas as pd

    if not HAS_MATPLOTLIB:
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle("Spatial Panel — Time Series", fontsize=12)

    x = list(range(len(df)))
    try:
        x = df.index.tolist()
    except Exception:
        pass

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row][col_idx]
        ax.plot(x, df[col].values, marker=".", linewidth=1.5)
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("time")
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(j, n_cols)
        axes[row][col_idx].set_visible(False)

    fig.tight_layout()
    return fig


def _format_panel_results(df: Any, csv_path: Path, png_path: Optional[Path]) -> str:
    """Format run_panel results as Markdown."""
    import pandas as pd

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n_ts = len(df)

    lines = [
        f"## Panel Results\n",
        f"Ran **{len(numeric_cols)} metrics** across **{n_ts} timestep(s)**.\n",
        "### Summary (first → last timestep)\n",
        "| Metric | t=0 | t=final | Trend |",
        "|--------|-----|---------|-------|",
    ]

    for col in numeric_cols:
        first_val = df[col].iloc[0] if n_ts > 0 else float("nan")
        last_val = df[col].iloc[-1] if n_ts > 0 else float("nan")
        try:
            fv = f"{first_val:.4g}"
            lv = f"{last_val:.4g}"
            tr = _trend(float(first_val), float(last_val))
        except Exception:
            fv, lv, tr = "—", "—", "?"
        lines.append(f"| `{col}` | {fv} | {lv} | {tr} |")

    lines.append("")
    lines.append(f"**CSV:** `{csv_path}`")
    if png_path and png_path.exists():
        lines.append(f"**Plot:** `{png_path}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: create_session
# ---------------------------------------------------------------------------

@mcp.tool()
def create_session(
    set_as_default: bool = True,
    session_id: Optional[str] = None,
) -> str:
    """Create a new spatial analysis session."""
    try:
        sid = session_manager.create_session(
            set_as_default=set_as_default,
            session_id=session_id,
        )
        write_session_meta(_SERVER_ROOT, sid, "SpatialTissue")
        default_str = " (set as default)" if set_as_default else ""
        return (
            f"## Session Created\n\n"
            f"**Session ID:** `{sid}`{default_str}\n\n"
            f"Next: call `load_preset_panel` or `add_metric` to build your panel."
        )
    except Exception as exc:
        return f"## Error\n\n{exc}"


# ---------------------------------------------------------------------------
# Tool 2: list_sessions
# ---------------------------------------------------------------------------

@mcp.tool()
def list_sessions() -> str:
    """List all spatial analysis sessions."""
    sessions = session_manager.list_sessions()
    if not sessions:
        return "## Sessions\n\nNo active sessions. Call `create_session` to start."

    default_id = session_manager.get_default_session_id()
    now = time.time()
    lines = [
        "## Active Sessions\n",
        "| Session ID | Panel metrics | Last output folder | Age (min) | Default |",
        "|-----------|--------------|-------------------|-----------|---------|",
    ]
    for s in sorted(sessions, key=lambda x: x.created_at, reverse=True):
        is_default = "✓" if s.session_id == default_id else ""
        age_min = f"{(now - s.created_at) / 60:.1f}"
        folder = Path(s.last_output_folder).name if s.last_output_folder else "—"
        lines.append(
            f"| `{s.session_id[:8]}…` | {len(s.panel)} | {folder} | {age_min} | {is_default} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: load_preset_panel
# ---------------------------------------------------------------------------

@mcp.tool()
def load_preset_panel(
    preset: str,
    session_id: Optional[str] = None,
) -> str:
    """Load a predefined metric panel, replacing current panel configuration.

    preset: 'basic' | 'spatial' | 'neighborhood' | 'comprehensive'
    """
    _ensure_spatialtissuepy()
    if not SPATIALTISSUEPY_AVAILABLE:
        return _INSTALL_MSG

    VALID_PRESETS = ("basic", "spatial", "neighborhood", "comprehensive")
    if preset not in VALID_PRESETS:
        return (
            f"## Error: Unknown Preset\n\n"
            f"`{preset}` is not a valid preset. Choose from: "
            + ", ".join(f"`{p}`" for p in VALID_PRESETS)
        )

    try:
        lib_panel = _load_panel(preset)  # type: ignore[misc]
        entries = [
            PanelEntry(metric_type=m.name, params=dict(m.params))
            for m in lib_panel.metrics
        ]

        sess = ensure_session(session_id)
        sess.panel = entries

        lines = [
            f"## Panel Loaded: `{preset}`\n",
            f"**{len(entries)} metrics** loaded into session `{sess.session_id[:8]}…`.\n",
            "| # | Metric | Parameters |",
            "|---|--------|-----------|",
        ]
        for i, e in enumerate(entries, 1):
            lines.append(f"| {i} | `{e.metric_type}` | `{_format_params(e.params)}` |")
        lines.append("\nCall `run_panel(output_folder=...)` to execute.")
        return "\n".join(lines)
    except Exception as exc:
        return f"## Error\n\n{exc}"


# ---------------------------------------------------------------------------
# Tool 4: add_metric
# ---------------------------------------------------------------------------

@mcp.tool()
def add_metric(
    metric_type: str,
    params_json: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Add one metric to the analysis panel.

    metric_type: metric name from the SpatialAnalysis skill catalogue.
    params_json: optional JSON object of parameters, e.g. '{"radius": 50}'.
    """
    _ensure_spatialtissuepy()

    try:
        params = _parse_json_dict(params_json, "params_json")
    except ValueError as exc:
        return f"## Error\n\n{exc}"

    # Validate metric name against registry (best-effort)
    if SPATIALTISSUEPY_AVAILABLE:
        try:
            from spatialtissuepy.summary.registry import get_metric
            get_metric(metric_type)
        except (KeyError, Exception) as exc:
            # Provide guidance but still allow adding (metric may be custom)
            try:
                from spatialtissuepy.summary.registry import list_metrics
                all_metrics = list_metrics()
                note = (
                    f"## Warning: Metric Not Found\n\n"
                    f"`{metric_type}` is not in the registry.\n\n"
                    f"**Available metrics:** {', '.join(f'`{m}`' for m in all_metrics)}\n\n"
                    f"The metric was NOT added. Check the `/spatial-analysis` skill for the full catalogue."
                )
                return note
            except Exception:
                return f"## Error\n\n{exc}"

    sess = ensure_session(session_id)
    entry = PanelEntry(metric_type=metric_type, params=params)
    sess.panel.append(entry)

    return (
        f"## Metric Added\n\n"
        f"Added `{entry.label}` to session `{sess.session_id[:8]}…`.\n\n"
        f"Panel now has **{len(sess.panel)} metric(s)**. "
        f"Call `view_panel` to review or `run_panel` to execute."
    )


# ---------------------------------------------------------------------------
# Tool 5: remove_metric
# ---------------------------------------------------------------------------

@mcp.tool()
def remove_metric(
    metric_type: str,
    session_id: Optional[str] = None,
) -> str:
    """Remove a metric from the analysis panel by metric_type."""
    sess = ensure_session(session_id)
    before = len(sess.panel)
    sess.panel = [e for e in sess.panel if e.metric_type != metric_type]
    after = len(sess.panel)

    if before == after:
        return (
            f"## Not Found\n\n"
            f"`{metric_type}` was not in the panel. "
            f"Current panel: {', '.join(f'`{e.metric_type}`' for e in sess.panel) or '(empty)'}."
        )
    removed = before - after
    return (
        f"## Metric Removed\n\n"
        f"Removed {removed} entry/entries for `{metric_type}`. "
        f"Panel now has **{after} metric(s)**."
    )


# ---------------------------------------------------------------------------
# Tool 6: view_panel
# ---------------------------------------------------------------------------

@mcp.tool()
def view_panel(session_id: Optional[str] = None) -> str:
    """Show the current panel configuration as a Markdown table."""
    sess = get_current_session(session_id)
    if sess is None:
        return (
            "## No Session\n\nNo active session found. Call `create_session` first."
        )

    if not sess.panel:
        return (
            f"## Panel Empty — Session `{sess.session_id[:8]}…`\n\n"
            "No metrics configured. Call `add_metric` or `load_preset_panel`."
        )

    lines = [
        f"## Panel Configuration — Session `{sess.session_id[:8]}…`\n",
        f"**{len(sess.panel)} metric(s)** queued.\n",
        "| # | Metric | Parameters | Display Label |",
        "|---|--------|-----------|--------------|",
    ]
    for i, e in enumerate(sess.panel, 1):
        lines.append(
            f"| {i} | `{e.metric_type}` | `{_format_params(e.params)}` | {e.label} |"
        )

    if sess.last_panel_csv_path:
        lines.append(f"\n**Last results:** `{sess.last_panel_csv_path}`")

    lines.append("\nCall `run_panel(output_folder=...)` to execute.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: run_panel
# ---------------------------------------------------------------------------

@mcp.tool()
def run_panel(
    output_folder: str,
    session_id: Optional[str] = None,
) -> str:
    """Run all panel metrics across every timestep and save results.

    output_folder: path to PhysiCell simulation output directory.
    """
    _ensure_spatialtissuepy()
    if not SPATIALTISSUEPY_AVAILABLE:
        return _INSTALL_MSG

    sess = ensure_session(session_id)
    if not sess.panel:
        return (
            "## Error: Empty Panel\n\n"
            "No metrics in panel. Call `add_metric` or `load_preset_panel` first."
        )

    try:
        # Build StatisticsPanel from session entries
        panel = _StatisticsPanel()  # type: ignore[misc]
        for entry in sess.panel:
            panel.add(entry.metric_type, **entry.params)

        # Load simulation
        sim = _PhysiCellSimulation.from_output_folder(output_folder)  # type: ignore[misc]

        # Run summarize — returns time × metrics DataFrame
        df = sim.summarize(panel)

        # Save results
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        ts = _ts()
        csv_path = safe_artifact_path(art_dir, f"panel_{ts}.csv")
        df.to_csv(csv_path)
        sess.last_output_folder = output_folder
        sess.last_panel_csv_path = str(csv_path)

        # Generate plot
        png_path: Optional[Path] = None
        if HAS_MATPLOTLIB:
            try:
                fig = _plot_panel_timeseries(df, sess.panel)
                if fig is not None:
                    png_path = safe_artifact_path(art_dir, f"panel_plot_{ts}.png")
                    fig.savefig(png_path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
            except Exception:
                pass

        return _format_panel_results(df, csv_path, png_path)

    except Exception as exc:
        return f"## Error\n\n`run_panel` failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 8: run_spatial_lda
# ---------------------------------------------------------------------------

@mcp.tool()
def run_spatial_lda(
    output_folder: str,
    n_topics: int = 5,
    neighborhood_radius: float = 50.0,
    timestep_indices: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Fit a Spatial LDA model to discover microenvironment topics.

    timestep_indices: optional JSON int list, e.g. '[0,5,10]'. Default: all timesteps.
    """
    _ensure_spatialtissuepy()
    if not SPATIALTISSUEPY_AVAILABLE:
        return _INSTALL_MSG

    try:
        indices = _parse_json_list(timestep_indices, "timestep_indices")
    except ValueError as exc:
        return f"## Error\n\n{exc}"

    try:
        sim = _PhysiCellSimulation.from_output_folder(  # type: ignore[misc]
            output_folder, include_dead_cells=False
        )

        # Collect SpatialTissueData objects
        n_ts = len(sim)
        selected = indices if indices is not None else list(range(n_ts))
        data_list = [sim.get_timestep(i).to_spatial_data() for i in selected]

        if not data_list:
            return "## Error\n\nNo timesteps selected."

        # Fit LDA
        lda = _SpatialLDA(  # type: ignore[misc]
            n_topics=n_topics,
            neighborhood_radius=neighborhood_radius,
        )
        lda.fit(data_list)

        # Topic summary
        summary_dict = lda.topic_summary()

        # Save topic composition to CSV
        sess = ensure_session(session_id)
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        ts = _ts()

        import pandas as pd
        if isinstance(summary_dict, dict) and summary_dict:
            summary_df = pd.DataFrame(summary_dict)
            csv_path = safe_artifact_path(art_dir, f"lda_topics_{ts}.csv")
            summary_df.to_csv(csv_path)
            csv_str = f"\n**Topics CSV:** `{csv_path}`"
        else:
            csv_str = ""

        # Store summary in session
        sess.last_lda_summary = {
            "n_topics": n_topics,
            "n_timesteps_fitted": len(data_list),
            "neighborhood_radius": neighborhood_radius,
            "output_folder": output_folder,
        }
        sess.last_output_folder = output_folder

        # Format topic preview
        lines = [
            f"## Spatial LDA Results\n",
            f"Fitted **{n_topics} topics** on **{len(data_list)} timestep(s)**.",
            f"Neighborhood radius: **{neighborhood_radius} µm**.\n",
        ]
        if isinstance(summary_dict, dict):
            lines.append("### Topic Summary\n```")
            for k, v in list(summary_dict.items())[:10]:
                lines.append(f"  {k}: {v}")
            lines.append("```")
        lines.append(csv_str)
        lines.append(
            "\n**Tip:** Topics represent recurring microenvironment compositions. "
            "High weight on a topic = that cell neighbourhood pattern dominates at that location."
        )
        return "\n".join(lines)

    except Exception as exc:
        return f"## Error\n\n`run_spatial_lda` failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 9: run_network_analysis
# ---------------------------------------------------------------------------

_ALL_NETWORK_METRICS = ["degree", "betweenness", "closeness", "clustering", "assortativity", "homophily"]


@mcp.tool()
def run_network_analysis(
    output_folder: str,
    metrics: Optional[str] = None,
    graph_method: str = "proximity",
    radius: float = 30.0,
    timestep_indices: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Build cell graphs and compute network metrics across timesteps.

    metrics: optional JSON list, e.g. '["betweenness","degree","assortativity"]'. Default: all.
    graph_method: 'proximity' | 'knn' | 'delaunay'.
    """
    _ensure_spatialtissuepy()
    if not SPATIALTISSUEPY_AVAILABLE:
        return _INSTALL_MSG

    try:
        metric_list = _parse_json_list(metrics, "metrics")
        indices = _parse_json_list(timestep_indices, "timestep_indices")
    except ValueError as exc:
        return f"## Error\n\n{exc}"

    if metric_list is None:
        metric_list = _ALL_NETWORK_METRICS

    invalid = [m for m in metric_list if m not in _ALL_NETWORK_METRICS]
    if invalid:
        return (
            f"## Error: Unknown Metrics\n\n"
            f"Unknown: {invalid}. Valid: {_ALL_NETWORK_METRICS}"
        )

    if graph_method not in ("proximity", "knn", "delaunay"):
        return "## Error\n\ngraph_method must be 'proximity', 'knn', or 'delaunay'."

    try:
        import pandas as pd

        sim = _PhysiCellSimulation.from_output_folder(output_folder)  # type: ignore[misc]
        n_ts = len(sim)
        selected = indices if indices is not None else list(range(n_ts))

        rows = []
        for i in selected:
            step = sim.get_timestep(i)
            data = step.to_spatial_data()
            try:
                t_val = float(step.time) if hasattr(step, "time") else i
            except Exception:
                t_val = float(i)

            cg = _CellGraph.from_spatial_data(  # type: ignore[misc]
                data, method=graph_method, radius=radius
            )

            row: Dict[str, Any] = {"time": t_val, "n_nodes": cg.n_nodes, "n_edges": cg.n_edges}

            for metric_name in metric_list:
                try:
                    if metric_name == "degree":
                        row["mean_degree"] = (
                            cg.n_edges * 2 / cg.n_nodes if cg.n_nodes > 0 else 0.0
                        )
                    elif metric_name == "betweenness":
                        stats = _network_module.mean_centrality_by_type(cg, "betweenness")  # type: ignore[misc]
                        for ctype, val in stats.items():
                            row[f"betweenness_{ctype}"] = val
                    elif metric_name == "closeness":
                        stats = _network_module.mean_centrality_by_type(cg, "closeness")  # type: ignore[misc]
                        for ctype, val in stats.items():
                            row[f"closeness_{ctype}"] = val
                    elif metric_name == "clustering":
                        row["avg_clustering"] = _network_module.average_clustering(cg)  # type: ignore[misc]
                    elif metric_name == "assortativity":
                        row["type_assortativity"] = _network_module.type_assortativity(cg)  # type: ignore[misc]
                    elif metric_name == "homophily":
                        row["homophily_ratio"] = _network_module.homophily_ratio(cg)  # type: ignore[misc]
                except Exception:
                    row[f"{metric_name}_error"] = float("nan")

            rows.append(row)

        df = pd.DataFrame(rows)

        sess = ensure_session(session_id)
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        ts = _ts()
        csv_path = safe_artifact_path(art_dir, f"network_metrics_{ts}.csv")
        df.to_csv(csv_path, index=False)
        sess.last_output_folder = output_folder
        sess.last_network_summary = {
            "metrics": metric_list,
            "graph_method": graph_method,
            "radius": radius,
            "n_timesteps": len(rows),
            "csv_path": str(csv_path),
        }

        # Format summary
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "time"]
        lines = [
            f"## Network Analysis Results\n",
            f"Method: **{graph_method}**, radius: **{radius} µm**, timesteps: **{len(rows)}**.\n",
            "### Mean Values Across Timesteps\n",
            "| Metric | Mean | Min | Max |",
            "|--------|------|-----|-----|",
        ]
        for col in numeric_cols[:20]:  # cap at 20 columns for readability
            m = df[col].mean()
            mn = df[col].min()
            mx = df[col].max()
            lines.append(f"| `{col}` | {m:.4g} | {mn:.4g} | {mx:.4g} |")

        lines.append(f"\n**CSV:** `{csv_path}`")
        return "\n".join(lines)

    except Exception as exc:
        return f"## Error\n\n`run_network_analysis` failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 10: run_spatial_mapper
# ---------------------------------------------------------------------------

@mcp.tool()
def run_spatial_mapper(
    output_folder: str,
    filter_fn: str = "density",
    n_intervals: int = 10,
    overlap: float = 0.5,
    timestep: int = -1,
    session_id: Optional[str] = None,
) -> str:
    """Run topological Mapper analysis on a single timestep.

    filter_fn: 'density' | 'pca' | 'eccentricity' | 'distance_to_type:<type_name>'.
    timestep: index into simulation timesteps. -1 means last timestep.
    """
    _ensure_spatialtissuepy()
    if not SPATIALTISSUEPY_AVAILABLE:
        return _INSTALL_MSG

    # Parse filter_fn — handle 'distance_to_type:<typename>' specially
    mapper_filter: Any = filter_fn
    if filter_fn.startswith("distance_to_type:"):
        cell_type = filter_fn.split(":", 1)[1].strip()
        if not cell_type:
            return "## Error\n\nNo cell type specified in `distance_to_type:<type_name>`."
        try:
            from spatialtissuepy.topology.spatial_filters import distance_to_type_filter
            mapper_filter = distance_to_type_filter(cell_type)
        except Exception as exc:
            return f"## Error\n\nCould not create distance_to_type filter: {exc}"
    elif filter_fn not in ("density", "pca", "eccentricity"):
        return (
            "## Error: Unknown filter_fn\n\n"
            "Valid options: `density`, `pca`, `eccentricity`, `distance_to_type:<type_name>`."
        )

    try:
        sim = _PhysiCellSimulation.from_output_folder(output_folder)  # type: ignore[misc]
        n_ts = len(sim)
        ts_idx = timestep if timestep >= 0 else max(0, n_ts + timestep)
        step = sim.get_timestep(ts_idx)
        data = step.to_spatial_data()

        mapper = _SpatialMapper(  # type: ignore[misc]
            filter_fn=mapper_filter,
            n_intervals=n_intervals,
            overlap=overlap,
        )
        result = mapper.fit(data)

        sess = ensure_session(session_id)
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        ts_str = _ts()

        # Try to save a graph visualization
        png_path: Optional[Path] = None
        if HAS_MATPLOTLIB:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                import networkx as nx
                pos = nx.spring_layout(result.graph, seed=42)
                # Colour nodes by dominant cell type if available
                node_colors = []
                for node_id in result.graph.nodes():
                    comp = result.node_compositions.get(node_id, {})
                    dominant = max(comp, key=comp.get) if comp else "unknown"  # type: ignore[arg-type]
                    node_colors.append(hash(dominant) % 256 / 256.0)
                nx.draw(
                    result.graph,
                    pos=pos,
                    ax=ax,
                    node_color=node_colors,
                    cmap=plt.cm.tab20,
                    node_size=100,
                    with_labels=False,
                    edge_color="gray",
                    alpha=0.8,
                )
                ax.set_title(
                    f"Mapper Graph — timestep {ts_idx}\n"
                    f"{result.n_nodes} nodes, {result.n_edges} edges, "
                    f"{result.n_components} components",
                    fontsize=9,
                )
                png_path = safe_artifact_path(art_dir, f"mapper_graph_{ts_str}.png")
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

        sess.last_mapper_summary = {
            "n_nodes": result.n_nodes,
            "n_edges": result.n_edges,
            "n_components": result.n_components,
            "filter_fn": filter_fn,
            "timestep_index": ts_idx,
            "output_folder": output_folder,
            "plot_path": str(png_path) if png_path else None,
        }
        sess.last_output_folder = output_folder

        lines = [
            f"## Mapper Results — Timestep {ts_idx}\n",
            f"Filter: **{filter_fn}**, intervals: **{n_intervals}**, overlap: **{overlap}**.\n",
            "| Property | Value |",
            "|----------|-------|",
            f"| Nodes | {result.n_nodes} |",
            f"| Edges | {result.n_edges} |",
            f"| Connected components | {result.n_components} |",
        ]
        if png_path and png_path.exists():
            lines.append(f"\n**Graph plot:** `{png_path}`")
        lines.append(
            "\n**Tip:** Isolated components suggest disconnected tissue regions. "
            "Loops in the mapper graph indicate cyclic spatial patterns."
        )
        return "\n".join(lines)

    except Exception as exc:
        return f"## Error\n\n`run_spatial_mapper` failed: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
