# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastmcp>=2.0.0",
#   "physicell-settings>=0.4.5",
#   "cairosvg>=2.7.0",
#   "pillow>=10.0.0",
#   "uq-physicell>=1.2.4",
#   "pcdl>=4.0.0",
# ]
# ///
"""
PhysiCell MCP Server with Session Management

This server provides tools for configuring PhysiCell biological simulations:
- Create simulation domains and add substrates
- Define cell types and their behaviors
- Create signal-behavior rules for realistic cell responses
- Export configurations for PhysiCell execution

Features lightweight session management and progress tracking.
"""

import sys
import os
import glob
import time
import subprocess
import signal
import uuid
import re
import math
import random
import csv
import json
import configparser
from pathlib import Path
from typing import Annotated, Any, Optional, Dict, List
from dataclasses import dataclass, field
import threading
from threading import Lock
from pydantic import Field

# Add the physicell_config package to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Make repo root importable for shared artifact_manager
sys.path.insert(0, str(current_dir.parent))
from artifact_manager import get_artifact_dir, list_artifacts, clean_artifacts, write_session_meta, list_artifact_sessions as _list_artifact_sessions_on_disk

_SERVER_ROOT = current_dir

from physicell_config import PhysiCellConfig
from physicell_config.config.embedded_signals_behaviors import (
    get_signals_behaviors,
    get_signal_by_name,
    get_behavior_by_name,
    update_signals_behaviors_context_from_config,
    get_expanded_signals,
    get_expanded_behaviors
)
from physicell_config.config.embedded_defaults import get_default_parameters

# Try to import PhysiBoSS module - only need to check availability
try:
    import physicell_config.modules.physiboss
    PHYSIBOSS_AVAILABLE = True
except ImportError:
    PHYSIBOSS_AVAILABLE = False

# Import session management
from session_manager import (
    session_manager, SessionState, WorkflowStep, MaBoSSContext,
    UQContext, UQParameterDef, RuleValidationResult,
    get_current_session, ensure_session, analyze_and_update_session_from_config
)

# Lazy-load UQ-PhysiCell modules to avoid slow startup (VTK/ome_types take ~4s)
# Availability flags are resolved on first access via _ensure_uq_imports()
_uq_imports_checked = False
UQ_AVAILABLE = False
UQ_BO_AVAILABLE = False
UQ_ABC_AVAILABLE = False

# These will be populated by _ensure_uq_imports()
PhysiCell_Model = None
ModelAnalysisContext = None
uq_run_simulations = None
BOCalibrationContext = None
uq_run_bo = None
SumSquaredDifferences = None
Manhattan = None
Chebyshev = None
ABCCalibrationContext = None
uq_run_abc = None

def _ensure_uq_imports():
    """Lazy-load UQ modules on first use. Called by UQ tool functions."""
    global _uq_imports_checked, UQ_AVAILABLE, UQ_BO_AVAILABLE, UQ_ABC_AVAILABLE
    global PhysiCell_Model, ModelAnalysisContext, uq_run_simulations
    global BOCalibrationContext, uq_run_bo, SumSquaredDifferences, Manhattan, Chebyshev
    global ABCCalibrationContext, uq_run_abc

    if _uq_imports_checked:
        return

    _uq_imports_checked = True

    try:
        from uq_physicell import PhysiCell_Model as _PM
        from uq_physicell.model_analysis import ModelAnalysisContext as _MAC
        from uq_physicell.model_analysis import run_simulations as _rs
        PhysiCell_Model = _PM
        ModelAnalysisContext = _MAC
        uq_run_simulations = _rs
        UQ_AVAILABLE = True
    except ImportError:
        UQ_AVAILABLE = False

    try:
        from uq_physicell.bo import (
            CalibrationContext as _BOCtx,
            run_bayesian_optimization as _run_bo,
            SumSquaredDifferences as _SSD,
            Manhattan as _Man,
            Chebyshev as _Cheb
        )
        BOCalibrationContext = _BOCtx
        uq_run_bo = _run_bo
        SumSquaredDifferences = _SSD
        Manhattan = _Man
        Chebyshev = _Cheb
        UQ_BO_AVAILABLE = True
    except ImportError:
        UQ_BO_AVAILABLE = False

    try:
        from uq_physicell.abc import (
            CalibrationContext as _ABCCtx,
            run_abc_calibration as _run_abc
        )
        ABCCalibrationContext = _ABCCtx
        uq_run_abc = _run_abc
        UQ_ABC_AVAILABLE = True
    except ImportError:
        UQ_ABC_AVAILABLE = False

# Lazy-load pcdl (PhysiCell data loader) for post-simulation analysis
_pcdl_imports_checked = False
PCDL_AVAILABLE = False
pcdl_TimeStep = None
pcdl_TimeSeries = None

def _ensure_pcdl_imports():
    """Lazy-load pcdl modules on first use. Called by analysis tool functions."""
    global _pcdl_imports_checked, PCDL_AVAILABLE, pcdl_TimeStep, pcdl_TimeSeries

    if _pcdl_imports_checked:
        return

    _pcdl_imports_checked = True

    try:
        from pcdl import TimeStep as _TS, TimeSeries as _TSeries
        pcdl_TimeStep = _TS
        pcdl_TimeSeries = _TSeries
        PCDL_AVAILABLE = True
    except ImportError:
        PCDL_AVAILABLE = False

from mcp.server.fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
from mcp.types import Icon

# ============================================================================
# PHYSICELL PROJECT EXECUTION INFRASTRUCTURE
# ============================================================================

PHYSICELL_ROOT = Path(os.environ.get("PHYSICELL_ROOT", Path.home() / "PhysiCell"))
USER_PROJECTS_DIR = PHYSICELL_ROOT / "user_projects"
TEMPLATE_DIR = PHYSICELL_ROOT / "sample_projects" / "template"
MCP_OUTPUT_DIR = Path(os.environ.get("MCP_OUTPUT_DIR", Path.home() / "PhysiCell_MCP_Output"))

@dataclass
class SimulationRun:
    """Tracks a running or completed PhysiCell simulation."""
    simulation_id: str
    project_name: str
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    status: str = "pending"  # pending, running, completed, failed, stopped
    output_folder: str = ""
    config_file: str = ""
    started_at: float = 0.0
    completed_at: Optional[float] = None
    return_code: Optional[int] = None
    error_message: str = ""
    log_file: str = ""

# Global simulation tracking
running_simulations: Dict[str, SimulationRun] = {}
simulations_lock = Lock()

logo_path = current_dir / "PhysiCell-Logo-64.png"
img = Image(path=str(logo_path))
icon = Icon(src=img.to_data_uri(), mimeType="image/png", sizes=["64x64"])
mcp = FastMCP(name="PhysiCell", icons=[icon])

# Initialize MCP server
""" hatch_mcp = HatchMCP("PhysiCell-Config-Builder",
                     fast_mcp=mcp,
                     origin_citation="PhysiCell: An Open Source Physics-Based Cell Simulator",
                     mcp_citation="https://github.com/marcorusc/Hatch_Pkg_Dev/tree/main/PhysiCell")
 """
# Legacy global variables for backward compatibility
# These are now managed through the session manager
config = None
scenario_context = ""


def _set_legacy_config(new_config):
    """Set config in current session for backward compatibility."""
    session = ensure_session()
    session.config = new_config
    global config
    config = new_config

def _set_legacy_scenario_context(context):
    """Set scenario context in current session."""
    session = ensure_session()
    session.scenario_context = context
    global scenario_context
    scenario_context = context

# ============================================================================
# SESSION MANAGEMENT TOOLS
# ============================================================================

@mcp.tool()
def create_session(
    set_as_default: bool = Field(default=True, description="Set this as the default session for subsequent operations."),
    session_name: Optional[str] = Field(default=None, description="Optional human-readable name for cross-server linking (e.g., 'gastric_cancer_v1')."),
) -> str:
    """Create a new PhysiCell simulation session.

    Sessions allow independent work on multiple simulation configurations.
    Create one before calling any setup tools.

    Returns:
        str: Session ID and instructions.
    """
    session_id = session_manager.create_session(set_as_default, session_name)

    result = f"**Session created:** {session_id[:8]}..."
    if session_name:
        result += f" ({session_name})"
    result += "\n"
    result += f"**Next steps:**\n"
    result += f"1. `analyze_biological_scenario()` - Set your biological context\n"
    result += f"2. `create_simulation_domain()` - Define spatial framework\n"
    result += f"3. Use `get_workflow_status()` to track progress"

    return result

@mcp.tool()
def list_sessions() -> str:
    """List all active simulation sessions with their status and progress.

    Returns:
        str: Formatted list of sessions with progress information.
    """
    sessions = session_manager.list_sessions()
    
    if not sessions:
        return "No active sessions. Use `create_session()` to start."
    
    result = f"## Active Sessions ({len(sessions)})\n\n"
    
    default_id = session_manager.get_default_session_id()
    
    for session in sessions:
        age_hours = (time.time() - session.created_at) / 3600
        progress = session.get_progress_percentage()
        
        # Mark default session
        default_marker = " (default)" if session.session_id == default_id else ""
        
        result += f"**{session.session_id[:8]}...{default_marker}**\n"
        result += f"- Age: {age_hours:.1f} hours\n"
        result += f"- Progress: {progress:.0f}%\n"
        result += f"- Components: {session.substrates_count} substrates, {session.cell_types_count} cell types, {session.rules_count} rules\n"
        
        if session.scenario_context:
            result += f"- Scenario: {session.scenario_context[:50]}{'...' if len(session.scenario_context) > 50 else ''}\n"
        
        result += "\n"
    
    result += "Use `set_default_session(session_id)` to switch between sessions."
    
    return result

@mcp.tool()
def list_artifact_sessions() -> str:
    """List all PhysiCell sessions that have artifact files on disk (including past server runs).

    Unlike list_sessions() which only shows in-memory sessions, this scans the
    artifacts/ directory and reads session_meta.json files, so previously created
    sessions are visible even after a server restart.

    Use the returned session_id and file paths to resume earlier work, e.g.:
      load_xml_configuration(xml_path='/path/to/artifacts/<uuid>/PhysiCell_settings.xml')
    """
    sessions = _list_artifact_sessions_on_disk(_SERVER_ROOT, server_name="PhysiCell")
    if not sessions:
        return "No artifact sessions found on disk."
    lines = ["## PhysiCell Artifact Sessions (on disk)\n"]
    for s in sessions:
        sid = s["session_id"]
        label = s.get("label") or ""
        created = s.get("created_at", "")[:19].replace("T", " ")
        files = s.get("files", [])
        lines.append(f"- **{sid[:8]}...**" + (f" ({label})" if label else ""))
        lines.append(f"  Full ID: `{sid}`")
        if created:
            lines.append(f"  Created: {created} UTC")
        if files:
            lines.append(f"  Files: {', '.join(files)}")
        else:
            lines.append("  Files: (none)")
    return "\n".join(lines)

@mcp.tool()
def set_default_session(
    session_id: Annotated[str, Field(description="ID of the session to activate. May be shortened to the first 8 characters.")],
) -> str:
    """Switch the default session used by subsequent tool calls.

    Returns:
        str: Confirmation of the session switch.
    """
    # Allow partial session IDs
    if len(session_id) == 8:
        sessions = session_manager.list_sessions()
        matching_sessions = [s for s in sessions if s.session_id.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0].session_id
        elif len(matching_sessions) > 1:
            return "Error: Ambiguous session ID. Multiple sessions match."
        else:
            return "Error: Session not found."
    
    success = session_manager.set_default_session(session_id)
    if success:
        session = session_manager.get_session(session_id)
        progress = session.get_progress_percentage()
        return f"**Switched to session:** {session_id[:8]}... (Progress: {progress:.0f}%)"
    else:
        return "Error: Session not found."

@mcp.tool()
def get_workflow_status() -> str:
    """Return the current workflow status and recommended next steps.

    Alias for `get_simulation_summary()` — provides identical information.
    Prefer `get_simulation_summary()` for new code.

    Returns:
        str: Progress summary with completed steps and next recommendations.
    """
    return get_simulation_summary()

@mcp.tool()
def delete_session(
    session_id: Annotated[str, Field(description="The ID of the session to delete permanently.")],
) -> str:
    """Delete a simulation session and all its state permanently.

    Returns:
        str: Confirmation of deletion.
    """
    success = session_manager.delete_session(session_id)
    if success:
        return f"**Session deleted:** {session_id[:8]}..."
    else:
        return "Error: Session not found"

@mcp.tool()
def set_maboss_context(
    model_name: Annotated[str, Field(description="Name of the MaBoSS model.")],
    bnd_file_path: Annotated[str, Field(description="Absolute path to the .bnd boolean network file.")],
    cfg_file_path: Annotated[str, Field(description="Absolute path to the .cfg configuration file.")],
    target_cell_type: Annotated[str, Field(description="Cell type this boolean model will be integrated into.")],
    available_nodes: str = Field(default="", description="Comma-separated list of available boolean nodes."),
    output_nodes: str = Field(default="", description="Comma-separated list of output nodes."),
    simulation_results: str = Field(default="", description="Summary of MaBoSS simulation behaviour."),
    biological_context: str = Field(default="", description="Original biological question or context."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Store MaBoSS model context for integration into a PhysiCell simulation.

    Typically called after analysing a MaBoSS model in the MaBoSS server.
    The stored context is used by `add_physiboss_model()` and `add_physiboss_input_link()`.

    Returns:
        str: Confirmation of context storage.
    """
    session = get_current_session(session_id)
    if not session:
        return "Error: No active session. Use `create_session()` first."
    
    maboss_context = MaBoSSContext(
        model_name=model_name,
        bnd_file_path=bnd_file_path,
        cfg_file_path=cfg_file_path,
        available_nodes=[node.strip() for node in available_nodes.split(",") if node.strip()],
        output_nodes=[node.strip() for node in output_nodes.split(",") if node.strip()],
        simulation_results=simulation_results,
        target_cell_type=target_cell_type,
        biological_context=biological_context
    )
    
    session.maboss_context = maboss_context
    
    result = f"**MaBoSS context stored:**\n"
    result += f"- Model: {model_name}\n"
    result += f"- Target cell type: {target_cell_type}\n"
    result += f"- Available nodes: {len(maboss_context.available_nodes)}\n"
    result += f"- Output nodes: {len(maboss_context.output_nodes)}\n"
    if simulation_results:
        result += f"- Simulation results available\n"
    result += f"**Next step:** Continue with PhysiCell simulation setup."
    
    return result

@mcp.tool()
def get_maboss_context(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """Return the stored MaBoSS context for the current session.

    Shows the linked boolean model name, file paths, available nodes, and
    simulation results previously stored via `set_maboss_context()`.

    Returns:
        str: MaBoSS context information, or a message if none is stored.
    """
    session = get_current_session(session_id)
    if not session:
        return "Error: No active session."
    
    if not session.maboss_context:
        return "No MaBoSS context available in current session."
    
    ctx = session.maboss_context
    result = f"## MaBoSS Context\n\n"
    result += f"**Model:** {ctx.model_name}\n"
    result += f"**Target Cell Type:** {ctx.target_cell_type}\n"
    result += f"**Files:**\n"
    result += f"- BND: {ctx.bnd_file_path}\n"
    result += f"- CFG: {ctx.cfg_file_path}\n\n"
    
    if ctx.available_nodes:
        result += f"**Available Nodes ({len(ctx.available_nodes)}):**\n"
        for node in ctx.available_nodes:
            result += f"- {node}\n"
        result += "\n"
    
    if ctx.output_nodes:
        result += f"**Output Nodes ({len(ctx.output_nodes)}):**\n"
        for node in ctx.output_nodes:
            result += f"- {node}\n"
        result += "\n"
    
    if ctx.simulation_results:
        result += f"**Simulation Results:**\n{ctx.simulation_results}\n\n"
    
    if ctx.biological_context:
        result += f"**Biological Context:**\n{ctx.biological_context}"
    
    return result

# ============================================================================
# XML CONFIGURATION LOADING
# ============================================================================

@mcp.tool()
def load_xml_configuration(
    filepath: Annotated[str, Field(description="Absolute path to the PhysiCell XML configuration file to load.")],
    session_name: Optional[str] = Field(default=None, description="Optional name for the session, useful for cross-server tracking."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Load an existing PhysiCell XML configuration file into the current session.

    After loading, inspect with `analyze_loaded_configuration()` or modify directly
    using tools such as `configure_cell_parameters()` and `add_single_cell_rule()`.

    Returns:
        str: Summary of loaded components and next steps.
    """
    try:
        session = ensure_session(session_id)
        xml_path = Path(filepath)
        
        if not xml_path.exists():
            return f"File not found: {filepath}"
        
        if not xml_path.is_file():
            return f"Path is not a file: {filepath}"
        
        # Create new config and load XML
        config = PhysiCellConfig()
        
        # First validate the XML
        is_valid, error_msg = config.validate_xml_file(str(xml_path))
        if not is_valid:
            return f"Invalid XML: {error_msg}"
        
        # Load the XML configuration
        config.load_xml(str(xml_path))
        session.config = config
        
        # Update session state
        session.loaded_from_xml = True
        session.original_xml_path = str(xml_path.absolute())
        session.mark_step_complete(WorkflowStep.XML_LOADED)
        
        # Analyze loaded content and update session counters
        analyze_and_update_session_from_config(session, config)
        
        # Concise summary
        parts = [f"{len(session.loaded_substrates)} substrates"]
        parts.append(f"{len(session.loaded_cell_types)} cell types")
        if session.loaded_physiboss_models:
            parts.append(f"{len(session.loaded_physiboss_models)} PhysiBoSS")
        if session.has_existing_rules:
            parts.append("rules")
        
        result = f"Loaded {xml_path.name}: {', '.join(parts)}"
        result += f"\nNext: analyze_loaded_configuration() or start modifying with existing tools"
        return result
        
    except Exception as e:
        return f"Load error: {str(e)}"

@mcp.tool()
def validate_xml_file(
    filepath: Annotated[str, Field(description="Absolute path to the PhysiCell XML file to validate.")],
) -> str:
    """Validate a PhysiCell XML configuration file without loading it.

    Returns:
        str: Validation result — valid confirmation or error description.
    """
    try:
        xml_path = Path(filepath)
        if not xml_path.exists():
            return f"File not found: {filepath}"
        
        config = PhysiCellConfig()
        is_valid, error_msg = config.validate_xml_file(str(xml_path))
        
        return f"Valid PhysiCell XML: {xml_path.name}" if is_valid else f"Invalid: {error_msg}"
            
    except Exception as e:
        return f"Validation error: {str(e)}"

@mcp.tool()
def analyze_loaded_configuration(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """Show an overview of the loaded XML configuration with modification instructions.

    Returns:
        str: Configuration summary (domain, substrates, cell types) with tool hints.
    """
    session = get_current_session(session_id)
    if not session or not session.config or not session.loaded_from_xml:
        return "No XML configuration loaded. Use load_xml_configuration() first."
    
    config = session.config
    lines = []
    
    # Source info
    if session.original_xml_path:
        lines.append(f"Source: {Path(session.original_xml_path).name}")
        if session.xml_modification_count > 0:
            lines.append(f"Modified: {session.xml_modification_count} times")
    
    # Domain
    try:
        domain_size = f"{config.domain.x_max-config.domain.x_min}x{config.domain.y_max-config.domain.y_min}x{config.domain.z_max-config.domain.z_min}"
        lines.append(f"Domain: {domain_size} μm")
    except:
        lines.append("Domain: configured")
    
    # Components with modification hints
    if session.loaded_substrates:
        lines.append(f"Substrates ({len(session.loaded_substrates)}): {', '.join(session.loaded_substrates)}")
        lines.append("  → Modify interactions: set_substrate_interaction(cell_type, substrate, ...)")
    
    if session.loaded_cell_types:
        lines.append(f"Cell types ({len(session.loaded_cell_types)}): {', '.join(session.loaded_cell_types)}")
        lines.append("  → Modify parameters: configure_cell_parameters(cell_type, ...)")
        lines.append("  → Add rules: add_single_cell_rule(cell_type, signal, ...)")
    
    if session.loaded_physiboss_models:
        lines.append(f"PhysiBoSS ({len(session.loaded_physiboss_models)}): {', '.join(session.loaded_physiboss_models)}")
        lines.append("  → Configure: configure_physiboss_settings(cell_type, ...)")
    
    lines.append("Use list_loaded_components() for detailed properties")
    
    session.mark_step_complete(WorkflowStep.XML_ANALYZED)
    return "\n".join(lines)

@mcp.tool()
def list_loaded_components(
    component_type: str = Field(default="all", description="Filter results: 'substrates', 'cell_types', 'physiboss', or 'all'."),
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """List components from a loaded XML configuration with details and modification hints.

    Returns:
        str: Detailed component information with parameter values and tool hints.
    """
    session = get_current_session(session_id)
    if not session or not session.config or not session.loaded_from_xml:
        return "No XML configuration loaded. Use load_xml_configuration() first."
    
    config = session.config
    lines = []
    
    if component_type in ["all", "substrates"] and session.loaded_substrates:
        lines.append("SUBSTRATES:")
        for name in session.loaded_substrates:
            try:
                substrate = config.substrates.get_substrate(name)
                if substrate:
                    lines.append(f"  {name}: D={substrate.diffusion_coefficient}, decay={substrate.decay_rate}, init={substrate.initial_condition}")
            except:
                lines.append(f"  {name}: properties not accessible")
        lines.append("  → Add interactions: set_substrate_interaction(cell_type, substrate, secretion_rate=X, uptake_rate=Y)")
        lines.append("")
    
    if component_type in ["all", "cell_types"] and session.loaded_cell_types:
        lines.append("CELL TYPES:")
        for name in session.loaded_cell_types:
            try:
                cell_type = config.cell_types.get_cell_type(name)
                if cell_type:
                    vol = cell_type.phenotype.volume.total
                    speed = cell_type.phenotype.motility.speed
                    cycle = cell_type.cycle_model
                    
                    physiboss = ""
                    if (hasattr(cell_type, 'phenotype') and hasattr(cell_type.phenotype, 'intracellular') and 
                        cell_type.phenotype.intracellular):
                        physiboss = ", PhysiBoSS enabled"
                    
                    lines.append(f"  {name}: vol={vol}, speed={speed}, cycle={cycle}{physiboss}")
            except:
                lines.append(f"  {name}: properties not accessible")
        
        lines.append("  → Modify parameters: configure_cell_parameters(cell_type, volume_total=X, motility_speed=Y, ...)")
        lines.append("  → Add rules: add_single_cell_rule(cell_type, signal, direction, behavior, ...)")
        lines.append("")
    
    if component_type in ["all", "physiboss"] and session.loaded_physiboss_models:
        lines.append("PHYSIBOSS MODELS:")
        for name in session.loaded_physiboss_models:
            lines.append(f"  {name}: Intracellular boolean network enabled")
        lines.append("  → Configure: configure_physiboss_settings(cell_type, intracellular_dt=X, ...)")
        lines.append("  → Add links: add_physiboss_input_link() / add_physiboss_output_link()")
        lines.append("")
    
    if not lines:
        return f"No {component_type} components found in loaded configuration"
    
    return "\n".join(lines).strip()

# ============================================================================
# BIOLOGICAL SCENARIO ANALYSIS
# ============================================================================

@mcp.tool()
def analyze_biological_scenario(
    biological_scenario: Annotated[str, Field(description="Description of the biological scenario or experimental setup (e.g., 'Breast cancer cells in hypoxic 3D tissue with immune infiltration').")],
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Store a biological scenario description to provide context for subsequent simulation setup.

    This context is displayed alongside signal/behavior lists and helps inform parameter choices.
    Call this before `create_simulation_domain()` to anchor the session to a specific biology.

    Returns:
        str: Confirmation message.
    """
    if not biological_scenario or not biological_scenario.strip():
        return "Error: Biological scenario description cannot be empty"
    
    session = ensure_session(session_id)
    session.scenario_context = biological_scenario.strip()
    session.mark_step_complete(WorkflowStep.SCENARIO_ANALYSIS)
    
    result = f"**Biological scenario stored:** {biological_scenario}\n"
    result += f"**Next step:** Use `create_simulation_domain()` to set up the spatial framework."
    
    return result

# ============================================================================
# SIMULATION SETUP
# ============================================================================

@mcp.tool()
def create_simulation_domain(
    domain_x: Annotated[float, Field(description="Domain width in micrometers (e.g., 2000).")],
    domain_y: Annotated[float, Field(description="Domain height in micrometers (e.g., 2000).")],
    use_2d: bool = Field(default=True, description="If True (default), creates a quasi-2D simulation. The z thickness is automatically set to the mesh spacing (dx). Set to False for a full 3D simulation and provide domain_z."),
    domain_z: Optional[float] = Field(default=None, description="Domain depth in micrometers. Required only for 3D simulations (use_2d=False). Ignored in 2D mode."),
    dx: float = Field(default=20.0, description="Mesh spacing in micrometers. Smaller values increase spatial resolution and computation time. In 2D, also sets the z thickness."),
    max_time: float = Field(default=7200.0, description="Maximum simulation time in minutes. 7200 = 5 days."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create the spatial and temporal framework for a PhysiCell simulation.

    Sets up the domain bounds, mesh resolution, and simulation duration.
    Must be called before adding substrates or cell types.
    The domain is centred at the origin: bounds run from -x/2 to +x/2 in each axis.

    In 2D mode (`use_2d=True`) the z extent is set to `dx` (one voxel thick)
    and `set_2D(True)` is applied so PhysiCell treats the simulation as planar.

    Returns:
        str: Confirmation with domain specs and next step.
    """
    # Basic validation
    if domain_x <= 0 or domain_y <= 0:
        return "Error: Domain dimensions must be positive"
    if dx <= 0:
        return "Error: Mesh spacing must be positive"
    if max_time <= 0:
        return "Error: Simulation time must be positive"

    # Resolve z dimension
    if use_2d:
        domain_z = dx  # one voxel thick
    else:
        if domain_z is None:
            return "Error: domain_z is required for 3D simulations. Set use_2d=True for a planar simulation."
        if domain_z <= 0:
            return "Error: domain_z must be positive"

    session = ensure_session(session_id)
    
    # Create new PhysiCell configuration
    session.config = PhysiCellConfig()
    session.config.domain.set_bounds(
        -domain_x/2, domain_x/2,
        -domain_y/2, domain_y/2,
        -domain_z/2, domain_z/2
    )
    session.config.domain.set_mesh(dx, dx, dx)

    if use_2d:
        session.config.domain.set_2D(True)

    session.config.options.set_max_time(max_time)
    session.config.options.set_time_steps(dt_diffusion=0.01, dt_mechanics=0.1, dt_phenotype=6.0)
    
    # Mark workflow step as complete
    session.mark_step_complete(WorkflowStep.DOMAIN_SETUP)
    
    # Format result
    mode = "2D" if use_2d else "3D"
    result = f"**Simulation domain created ({mode}):**\n"
    result += f"- Domain: {domain_x}×{domain_y}" + (f" μm (z = {domain_z} μm, one voxel)\n" if use_2d else f"×{domain_z} μm\n")
    result += f"- Mesh: {dx} μm\n"
    result += f"- Duration: {max_time/60:.1f} hours\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_substrate()` to add oxygen, nutrients, or drugs."
    
    return result

@mcp.tool()
def add_single_substrate(
    substrate_name: Annotated[str, Field(description="Name of the substrate (e.g., 'oxygen', 'glucose', 'drug').")],
    diffusion_coefficient: Annotated[float, Field(description="Diffusion rate in μm²/min. Typical: 100000 for oxygen, 30000 for glucose.")],
    decay_rate: Annotated[float, Field(description="Decay/uptake rate in 1/min. Typical: 0.01.")],
    initial_condition: Annotated[float, Field(description="Starting concentration everywhere in the domain. Typical: 38 for oxygen (mmHg).")],
    units: str = Field(default="dimensionless", description="Concentration units label (cosmetic only)."),
    dirichlet_enabled: bool = Field(default=False, description="If True, enforce a fixed boundary concentration (Dirichlet condition)."),
    dirichlet_value: Optional[float] = Field(default=None, description="Fixed boundary concentration. Defaults to initial_condition if omitted."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a chemical substrate (oxygen, glucose, drug, etc.) to the simulation environment.

    Substrates diffuse through the domain and can be consumed or secreted by cells.
    Call `create_simulation_domain()` first. Repeat for each substrate.

    Returns:
        str: Confirmation with substrate parameters and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not substrate_name or not substrate_name.strip():
        return "Error: Substrate name cannot be empty"
    if diffusion_coefficient < 0:
        return "Error: Diffusion coefficient must be non-negative"
    if decay_rate < 0:
        return "Error: Decay rate must be non-negative"
    
    if dirichlet_value is None:
        dirichlet_value = initial_condition
    
    name = substrate_name.strip()
    
    # Add substrate to configuration
    session.config.substrates.add_substrate(
        name,
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        initial_condition=initial_condition,
        dirichlet_enabled=dirichlet_enabled,
        dirichlet_value=dirichlet_value,
        units=units
    )

    # Set per-boundary Dirichlet conditions when enabled
    dirichlet_boundaries = []
    if dirichlet_enabled:
        # Determine 2D vs 3D using the domain's use_2D flag
        try:
            is_3d = not session.config.domain.data.get("use_2D", False)
        except Exception:
            is_3d = False

        boundaries = ["xmin", "xmax", "ymin", "ymax"]
        if is_3d:
            boundaries += ["zmin", "zmax"]

        name = substrate_name.strip()
        for boundary in boundaries:
            session.config.substrates.set_dirichlet_boundary(name, boundary, True, dirichlet_value)
        dirichlet_boundaries = boundaries

    # Update session counters
    session.substrates_count += 1
    session.mark_step_complete(WorkflowStep.SUBSTRATES_ADDED)
    
    # Format result
    result = f"**Substrate added:** {substrate_name}\n"
    result += f"- Diffusion: {diffusion_coefficient:g} μm²/min\n"
    result += f"- Decay: {decay_rate:g} min⁻¹\n"
    result += f"- Initial: {initial_condition:g} {units}\n"
    if dirichlet_enabled:
        result += f"- Dirichlet boundaries: {', '.join(dirichlet_boundaries)} = {dirichlet_value:g} {units}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_type()` to add cancer cells, immune cells, etc."
    
    return result

@mcp.tool()
def add_single_cell_type(
    cell_type_name: Annotated[str, Field(description="Name for this cell type (e.g., 'cancer_cell', 'immune_cell', 'fibroblast').")],
    cycle_model: str = Field(default="Ki67_basic", description="Cell cycle model. Use `get_available_cycle_models()` to list options. Common: 'Ki67_basic', 'Ki67_advanced', 'live'."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a cell type (cancer, immune, fibroblast, etc.) to the simulation.

    Cell types define the agents that populate the simulation domain.
    Call `create_simulation_domain()` first. Repeat for each cell type.
    After adding, use `configure_cell_parameters()` to set size and motility,
    and `add_single_cell_rule()` to define signal-response behaviours.

    Returns:
        str: Confirmation with cell type details and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not cell_type_name or not cell_type_name.strip():
        return "Error: Cell type name cannot be empty"
    
    cell_type_name = cell_type_name.strip()
    
    # Add cell type to configuration
    session.config.cell_types.add_cell_type(cell_type_name, template='default')
    session.config.cell_types.set_cycle_model(cell_type_name, cycle_model)
    
    # Update session counters
    session.cell_types_count += 1
    session.mark_step_complete(WorkflowStep.CELL_TYPES_ADDED)
    
    # Format result
    result = f"**Cell type added:** {cell_type_name}\n"
    result += f"- Cycle model: {cycle_model}\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    result += f"**Next step:** Use `add_single_cell_rule()` to create cell behavior rules.\n"
    result += f"First, use `list_all_available_signals()` and `list_all_available_behaviors()` to see options."
    
    return result



@mcp.tool()
def configure_cell_parameters(
    cell_type: Annotated[str, Field(description="Name of an existing cell type to configure.")],
    volume_total: float = Field(default=2500.0, description="Total cell volume in μm³."),
    volume_nuclear: float = Field(default=500.0, description="Nuclear volume in μm³."),
    fluid_fraction: float = Field(default=0.75, description="Cytoplasmic fluid fraction (0–1)."),
    motility_speed: float = Field(default=0.5, description="Cell migration speed in μm/min."),
    persistence_time: float = Field(default=5.0, description="Directional persistence time in minutes."),
    apoptosis_rate: float = Field(default=0.0001, description="Spontaneous apoptosis rate in 1/min."),
    necrosis_rate: float = Field(default=0.0001, description="Spontaneous necrosis rate in 1/min."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Modify volume, motility, and death-rate parameters for an existing cell type.

    The cell type must already exist (created by `add_single_cell_type()`).
    Call repeatedly to configure multiple cell types.

    Returns:
        str: Confirmation with the configured parameter values.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    try:
        # Set volume parameters
        session.config.cell_types.set_volume_parameters(cell_type, total=volume_total,
                                                        nuclear=volume_nuclear, fluid_fraction=fluid_fraction)

        # Set motility parameters
        session.config.cell_types.set_motility(cell_type, speed=motility_speed,
                                               persistence_time=persistence_time, enabled=True)

        # Set death rates
        session.config.cell_types.set_death_rate(cell_type, 'apoptosis', apoptosis_rate)
        session.config.cell_types.set_death_rate(cell_type, 'necrosis', necrosis_rate)

        # Track modification if loaded from XML
        if session.loaded_from_xml:
            session.mark_xml_modification()

        result = f"**Configured parameters for {cell_type}:**\n"
        result += f"- **Volume:** {volume_total:g} μm³ (nuclear: {volume_nuclear:g} μm³)\n"
        result += f"- **Motility:** {motility_speed:g} μm/min (persistence: {persistence_time:g} min)\n"
        result += f"- **Death rates:** apoptosis {apoptosis_rate:g}, necrosis {necrosis_rate:g} min⁻¹"
        
        return result
    except Exception as e:
        return f"Error configuring cell type '{cell_type}': {str(e)}"

@mcp.tool()
def set_substrate_interaction(
    cell_type: Annotated[str, Field(description="Name of an existing cell type.")],
    substrate: Annotated[str, Field(description="Name of an existing substrate (must match a name passed to add_single_substrate()).")],
    secretion_rate: float = Field(default=0.0, description="Rate at which the cell secretes the substrate (1/min)."),
    uptake_rate: float = Field(default=0.0, description="Rate at which the cell consumes the substrate (1/min). Typical oxygen uptake: 10."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Define how a cell type interacts with a substrate via secretion and uptake rates.

    Both cell type and substrate must already exist in the current session.
    Call repeatedly to configure multiple cell-substrate pairs.

    Returns:
        str: Confirmation with the configured rates.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    try:
        session.config.cell_types.add_secretion(cell_type, substrate,
                                                secretion_rate=secretion_rate,
                                                uptake_rate=uptake_rate)

        # Track modification if loaded from XML
        if session.loaded_from_xml:
            session.mark_xml_modification()
        
        return f"**Substrate interaction set:** {cell_type} ↔ {substrate} (secretion: {secretion_rate:g}, uptake: {uptake_rate:g} min⁻¹)"
    except Exception as e:
        return f"Error setting substrate interaction: {str(e)}"


@mcp.tool()
def set_cell_transformation_rate(cell_type: str, target_cell_type: str, rate: float = 0.001) -> str:
    """
When the user asks to set transformation rates, cell type transitions, or EMT/MET rates,
this function sets the base transformation rate from one cell type to another in the XML config.
This MUST be called BEFORE adding rules that target 'transition to X' behaviors,
because the XML default transformation rate is 0 and rules interpolate toward the XML default.
The function modifies an existing cell type; it does not create a new one.
To set multiple transformation rates, use this function repeatedly.

Input parameters:
cell_type (str): Name of existing cell type that will transform
target_cell_type (str): Name of existing cell type to transform into
rate (float): Transformation rate in 1/min (default: 0.001)

Returns:
str: Success message with transformation rate details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types

    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    if target_cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Target cell type '{target_cell_type}' not found. Available: {available}"

    if rate < 0:
        return "Error: Transformation rate must be non-negative"

    # Set the rate directly in the config dict
    transformations = cell_types_dict[cell_type]['phenotype']['cell_transformations']
    transformations['transformation_rates'][target_cell_type] = rate

    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)

    # Track modification if loaded from XML
    if session.loaded_from_xml:
        session.mark_xml_modification()

    return (
        f"**Transformation rate set:** {cell_type} → {target_cell_type} at {rate:g} min⁻¹\n"
        f"You can now add rules targeting `transition to {target_cell_type}` — "
        f"the XML default is {rate:g} (nonzero), so Hill function interpolation will work correctly."
    )


@mcp.tool()
def set_cell_interaction(cell_type: str, target_cell_type: str,
                         interaction_type: str, rate: float = 0.001) -> str:
    """
When the user asks to set attack rates, phagocytosis rates, or fusion rates between cell types,
this function sets the per-target-cell-type interaction rate in the XML config.
This MUST be called BEFORE adding rules that target 'attack X', 'phagocytose X', or 'fuse to X'
behaviors, because their XML defaults are 0 and rules interpolate toward the XML default.
The function modifies an existing cell type; it does not create a new one.
To set multiple interactions, use this function repeatedly.

Input parameters:
cell_type (str): Name of existing cell type that performs the interaction
target_cell_type (str): Name of existing cell type that is the target
interaction_type (str): Type of interaction - 'attack', 'phagocytose', or 'fuse'
rate (float): Interaction rate in 1/min (default: 0.001)

Returns:
str: Success message with interaction details
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types

    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    if target_cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Target cell type '{target_cell_type}' not found. Available: {available}"

    if rate < 0:
        return "Error: Rate must be non-negative"

    interactions = cell_types_dict[cell_type]['phenotype']['cell_interactions']

    type_map = {
        'attack': ('attack_rates', 'attack'),
        'phagocytose': ('live_phagocytosis_rates', 'phagocytose'),
        'fuse': ('fusion_rates', 'fuse to'),
    }
    if interaction_type not in type_map:
        return f"Error: interaction_type must be 'attack', 'phagocytose', or 'fuse'. Got: '{interaction_type}'"

    dict_key, behavior_prefix = type_map[interaction_type]
    interactions[dict_key][target_cell_type] = rate

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    behavior_name = f"{behavior_prefix} {target_cell_type}"
    return (
        f"**Interaction rate set:** {cell_type} {interaction_type}s {target_cell_type} at {rate:g} min⁻¹\n"
        f"You can now add rules targeting `{behavior_name}` — "
        f"the XML default is {rate:g} (nonzero), so Hill function interpolation will work correctly."
    )


@mcp.tool()
def configure_cell_interactions(cell_type: str,
                                apoptotic_phagocytosis_rate: Optional[float] = None,
                                necrotic_phagocytosis_rate: Optional[float] = None,
                                other_dead_phagocytosis_rate: Optional[float] = None,
                                attack_damage_rate: Optional[float] = None,
                                attack_duration: Optional[float] = None) -> str:
    """
When the user asks to set dead-cell phagocytosis rates, attack damage rate, or attack duration,
this function configures non-per-type cell interaction parameters.
This should be called BEFORE adding rules that target these behaviors when their defaults are 0.
Only parameters that are explicitly provided will be modified.

Input parameters:
cell_type (str): Name of existing cell type
apoptotic_phagocytosis_rate (float): Rate of phagocytosing apoptotic cells (1/min, default 0.0)
necrotic_phagocytosis_rate (float): Rate of phagocytosing necrotic cells (1/min, default 0.0)
other_dead_phagocytosis_rate (float): Rate of phagocytosing other dead cells (1/min, default 0.0)
attack_damage_rate (float): Rate of damage during attack (1/min, default 1.0)
attack_duration (float): Duration of attack (min, default 0.1)

Returns:
str: Success message with configured parameters
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    interactions = cell_types_dict[cell_type]['phenotype']['cell_interactions']
    changes = []

    params = {
        'apoptotic_phagocytosis_rate': apoptotic_phagocytosis_rate,
        'necrotic_phagocytosis_rate': necrotic_phagocytosis_rate,
        'other_dead_phagocytosis_rate': other_dead_phagocytosis_rate,
        'attack_damage_rate': attack_damage_rate,
        'attack_duration': attack_duration,
    }

    for key, value in params.items():
        if value is not None:
            if value < 0:
                return f"Error: {key} must be non-negative"
            interactions[key] = value
            changes.append(f"- {key}: {value:g}")

    if not changes:
        return "Error: No parameters provided. Specify at least one parameter to set."

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    return f"**Cell interactions configured for {cell_type}:**\n" + "\n".join(changes)


@mcp.tool()
def configure_cell_mechanics(cell_type: str,
                             attachment_rate: Optional[float] = None,
                             detachment_rate: Optional[float] = None,
                             cell_cell_adhesion_strength: Optional[float] = None,
                             cell_cell_repulsion_strength: Optional[float] = None,
                             relative_maximum_adhesion_distance: Optional[float] = None) -> str:
    """
When the user asks to set cell attachment rates, detachment rates, adhesion, or repulsion,
this function configures cell mechanics parameters.
This should be called BEFORE adding rules that target 'cell attachment rate' or
'cell detachment rate' behaviors, because their XML defaults are 0.
Only parameters that are explicitly provided will be modified.

Input parameters:
cell_type (str): Name of existing cell type
attachment_rate (float): Rate of cell attachment (1/min, default 0.0)
detachment_rate (float): Rate of cell detachment (1/min, default 0.0)
cell_cell_adhesion_strength (float): Adhesion strength (micron/min, default 0.4)
cell_cell_repulsion_strength (float): Repulsion strength (micron/min, default 10.0)
relative_maximum_adhesion_distance (float): Max adhesion distance (dimensionless, default 1.25)

Returns:
str: Success message with configured parameters
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    mechanics = cell_types_dict[cell_type]['phenotype']['mechanics']
    changes = []

    params = {
        'attachment_rate': attachment_rate,
        'detachment_rate': detachment_rate,
        'cell_cell_adhesion_strength': cell_cell_adhesion_strength,
        'cell_cell_repulsion_strength': cell_cell_repulsion_strength,
        'relative_maximum_adhesion_distance': relative_maximum_adhesion_distance,
    }

    for key, value in params.items():
        if value is not None:
            if value < 0:
                return f"Error: {key} must be non-negative"
            mechanics[key] = value
            changes.append(f"- {key}: {value:g}")

    if not changes:
        return "Error: No parameters provided. Specify at least one parameter to set."

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    return f"**Cell mechanics configured for {cell_type}:**\n" + "\n".join(changes)


@mcp.tool()
def configure_cell_integrity(cell_type: str,
                             damage_rate: Optional[float] = None,
                             damage_repair_rate: Optional[float] = None) -> str:
    """
When the user asks to set cell damage rates or damage repair rates,
this function configures cell integrity parameters.
This should be called BEFORE adding rules that target 'damage rate' or
'damage repair rate' behaviors, because their XML defaults are 0.
Only parameters that are explicitly provided will be modified.

Input parameters:
cell_type (str): Name of existing cell type
damage_rate (float): Rate of damage accumulation (1/min, default 0.0)
damage_repair_rate (float): Rate of damage repair (1/min, default 0.0)

Returns:
str: Success message with configured parameters
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    integrity = cell_types_dict[cell_type]['phenotype']['cell_integrity']
    changes = []

    params = {
        'damage_rate': damage_rate,
        'damage_repair_rate': damage_repair_rate,
    }

    for key, value in params.items():
        if value is not None:
            if value < 0:
                return f"Error: {key} must be non-negative"
            integrity[key] = value
            changes.append(f"- {key}: {value:g}")

    if not changes:
        return "Error: No parameters provided. Specify at least one parameter to set."

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    return f"**Cell integrity configured for {cell_type}:**\n" + "\n".join(changes)


@mcp.tool()
def set_chemotaxis(cell_type: str, substrate: str, enabled: bool = True,
                   direction: int = 1) -> str:
    """
When the user asks to set chemotaxis, enable chemotactic migration, or direct cells toward a substrate,
this function configures basic chemotaxis for a cell type.
This sets cells to follow (or flee from) a single substrate gradient.
For multi-substrate chemotaxis with per-substrate sensitivity values, use set_advanced_chemotaxis() instead.

Input parameters:
cell_type (str): Name of existing cell type
substrate (str): Name of existing substrate to chemotax toward/away from (e.g., 'oxygen')
enabled (bool): Whether chemotaxis is active (default: True)
direction (int): 1 for attraction (up gradient), -1 for repulsion (down gradient) (default: 1)

Returns:
str: Success message with chemotaxis configuration
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    if direction not in (1, -1):
        return "Error: direction must be 1 (attraction, up gradient) or -1 (repulsion, down gradient)"

    try:
        session.config.cell_types.set_chemotaxis(cell_type, substrate,
                                                  enabled=enabled, direction=direction)
    except ValueError as e:
        return f"Error: {e}"

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    dir_label = "up gradient (attraction)" if direction == 1 else "down gradient (repulsion)"
    return (f"**Chemotaxis configured for {cell_type}:**\n"
            f"- substrate: {substrate}\n"
            f"- enabled: {enabled}\n"
            f"- direction: {direction} ({dir_label})\n\n"
            f"Note: This sets basic chemotaxis. Motility must also be enabled "
            f"(use configure_cell_parameters with motility_speed > 0).")


@mcp.tool()
def set_advanced_chemotaxis(cell_type: str, substrate: str, sensitivity: float,
                            enabled: bool = True,
                            normalize_each_gradient: bool = False) -> str:
    """
When the user asks to set chemotactic sensitivity, configure multi-substrate chemotaxis,
or set how strongly cells respond to a gradient, this function configures advanced chemotaxis.
Unlike basic chemotaxis (single substrate), advanced chemotaxis allows per-substrate sensitivity
values and supports simultaneous response to multiple substrate gradients.

Call this BEFORE adding rules that target 'chemotactic response to <substrate>' behaviors,
because the XML default chemotactic sensitivity is 0 and rules interpolate toward the XML default.
To set sensitivity for multiple substrates, call this function once per substrate.

Input parameters:
cell_type (str): Name of existing cell type
substrate (str): Name of existing substrate (e.g., 'oxygen')
sensitivity (float): Chemotactic sensitivity value. Positive = attraction, negative = repulsion.
    Typical range: -1.0 to 1.0, but larger values are allowed.
enabled (bool): Whether advanced chemotaxis is active (default: True)
normalize_each_gradient (bool): Whether to normalize each substrate gradient independently (default: False)

Returns:
str: Success message with advanced chemotaxis configuration
    """
    session = get_current_session()
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"

    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        available = list(cell_types_dict.keys())
        return f"Error: Cell type '{cell_type}' not found. Available: {available}"

    motility = cell_types_dict[cell_type]['phenotype']['motility']

    # Get existing advanced_chemotaxis settings to preserve other substrates
    adv = motility.get('advanced_chemotaxis', {})
    existing_sensitivities = adv.get('chemotactic_sensitivities', {})

    # Remove placeholder if present
    existing_sensitivities.pop('substrate', None)

    # Update the target substrate's sensitivity
    existing_sensitivities[substrate] = sensitivity

    try:
        session.config.cell_types.set_advanced_chemotaxis(
            cell_type, existing_sensitivities,
            enabled=enabled, normalize_each_gradient=normalize_each_gradient)
    except ValueError as e:
        return f"Error: {e}"

    _set_legacy_config(session.config)
    if session.loaded_from_xml:
        session.mark_xml_modification()

    # Build summary of all current sensitivities
    current = motility.get('advanced_chemotaxis', {}).get('chemotactic_sensitivities', {})
    sens_lines = [f"- {sub}: {val}" for sub, val in current.items() if sub != 'substrate']

    return (f"**Advanced chemotaxis configured for {cell_type}:**\n"
            f"- enabled: {enabled}\n"
            f"- normalize_each_gradient: {normalize_each_gradient}\n"
            f"- chemotactic sensitivities:\n" + "\n".join(sens_lines) + "\n\n"
            f"Note: Motility must also be enabled "
            f"(use configure_cell_parameters with motility_speed > 0).")


# ============================================================================
# PARAMETER DISCOVERY AND DEFAULTS
# ============================================================================

@mcp.tool()
def get_available_cycle_models() -> str:
    """List all available PhysiCell cell cycle models with their identifiers.

    Returns:
        str: Markdown list of model keys and names to use in `add_single_cell_type()`.
    """

    defaults = get_default_parameters()
    cycle_models = defaults.get("cell_cycle_models", {})
    
    result = "## Available Cell Cycle Models\n\n"
    for model_key, model_data in cycle_models.items():
        model_name = model_data.get("name", model_key)
        result += f"- **{model_key}**: {model_name}\n"
    
    result += "\n**Usage:** Use exact model names in add_single_cell_type() function.\n"
    result += "**Most common:** Ki67_basic, Ki67_advanced, live"
    
    return result

# ============================================================================
# SIGNAL AND BEHAVIOR DISCOVERY
# ============================================================================

@mcp.tool()
def list_all_available_signals(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """List all PhysiCell signals that can be used in cell rules.

    Automatically expands to include substrate-specific and cell-type-specific signals
    based on components already added to the current session.
    Call `list_all_available_behaviors()` to see the corresponding behavior targets.

    Returns:
        str: Markdown table of signals grouped by type.
    """
    session = get_current_session(session_id)
    
    # Update context from current config if available
    if session and session.config:
        update_signals_behaviors_context_from_config(session.config)
        # Use expanded signals which include context-specific signals
        try:
            signals_data = {signal['name']: signal for signal in get_expanded_signals()}
        except:
            # Fall back to basic signals if expanded version fails
            signals_data = get_signals_behaviors()["signals"]
    else:
        signals_data = get_signals_behaviors()["signals"]
    
    # Get current scenario context if available
    scenario_context = session.scenario_context if session else ""
    
    result = f"## PhysiCell Signals ({len(signals_data)} total)\n"
    if scenario_context:
        result += f"**Current scenario:** {scenario_context}\n\n"
    
    # Group signals by type for better organization
    signals_by_type = {}
    for signal_name, signal_info in signals_data.items():
        signal_type = signal_info.get("type", "other")
        if signal_type not in signals_by_type:
            signals_by_type[signal_type] = []
        signals_by_type[signal_type].append(signal_info)
    
    # Display signals grouped by type
    for signal_type, signals in signals_by_type.items():
        result += f"### {signal_type.upper()}\n"
        for signal in signals:
            signal_name = signal.get('name', 'Unknown')
            signal_desc = signal.get('description', 'No description')
            result += f"- **{signal_name}**: {signal_desc}\n"
            requires = signal.get('requires', [])
            if requires:
                result += f"  - *Requires: {', '.join(requires)}*\n"
        result += "\n"
    
    result += "**Note:** Use exact signal names in add_single_cell_rule() function.\n"
    result += "**Context:** Signals are automatically expanded based on current substrates and cell types."
    
    return result

@mcp.tool()
def list_all_available_behaviors(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """List all PhysiCell behaviours that can be controlled by cell rules.

    Automatically expands to include substrate-specific and cell-type-specific behaviours
    based on components already added to the current session.
    Use exact behavior names in `add_single_cell_rule()`.

    Returns:
        str: Markdown table of behaviours grouped by type.
    """
    session = get_current_session(session_id)
    
    # Update context from current config if available
    if session and session.config:
        update_signals_behaviors_context_from_config(session.config)
        # Use expanded behaviors which include context-specific behaviors
        try:
            behaviors_data = {behavior['name']: behavior for behavior in get_expanded_behaviors()}
        except:
            # Fall back to basic behaviors if expanded version fails
            behaviors_data = get_signals_behaviors()["behaviors"]
    else:
        behaviors_data = get_signals_behaviors()["behaviors"]
    
    # Get current scenario context if available
    scenario_context = session.scenario_context if session else ""
    
    result = f"## PhysiCell Behaviors ({len(behaviors_data)} total)\n"
    if scenario_context:
        result += f"**Current scenario:** {scenario_context}\n\n"
    
    # Group behaviors by type for better organization
    behaviors_by_type = {}
    for behavior_name, behavior_info in behaviors_data.items():
        behavior_type = behavior_info.get("type", "other")
        if behavior_type not in behaviors_by_type:
            behaviors_by_type[behavior_type] = []
        behaviors_by_type[behavior_type].append(behavior_info)
    
    # Display behaviors grouped by type
    for behavior_type, behaviors in behaviors_by_type.items():
        result += f"### {behavior_type.upper()}\n"
        for behavior in behaviors:
            behavior_name = behavior.get('name', 'Unknown')
            behavior_desc = behavior.get('description', 'No description')
            result += f"- **{behavior_name}**: {behavior_desc}\n"
            requires = behavior.get('requires', [])
            if requires:
                result += f"  - *Requires: {', '.join(requires)}*\n"
        result += "\n"
    
    result += "**Note:** Use exact behavior names in add_single_cell_rule() function.\n"
    result += "**Context:** Behaviors are automatically expanded based on current substrates and cell types."
    
    return result


# ============================================================================
# CELL RULES AND PHYSIBOSS
# ============================================================================

def _get_behavior_default_from_config(session, cell_type: str, behavior: str) -> Optional[float]:
    """Look up the current XML default value for a behavior from the in-memory config.

    Returns the default value (float) if the behavior is recognized and the
    cell type exists, or None if the behavior cannot be mapped.
    """
    cell_types_dict = session.config.cell_types.cell_types
    if cell_type not in cell_types_dict:
        return None

    phenotype = cell_types_dict[cell_type]['phenotype']
    behavior = behavior.strip()

    try:
        # Death rates
        if behavior == "apoptosis":
            death = phenotype.get('death', {}).get('apoptosis', {})
            return death.get('default_rate', death.get('rate', None))
        if behavior == "necrosis":
            death = phenotype.get('death', {}).get('necrosis', {})
            return death.get('default_rate', death.get('rate', None))

        # Motility
        if behavior == "migration speed":
            return phenotype.get('motility', {}).get('speed', None)
        if behavior == "migration bias":
            return phenotype.get('motility', {}).get('migration_bias', None)
        if behavior in ("persistence time", "migration persistence time"):
            return phenotype.get('motility', {}).get('persistence_time', None)

        # Transformation: "transition to X" or "transform to X"
        for prefix in ("transition to ", "transform to "):
            if behavior.startswith(prefix):
                target = behavior[len(prefix):]
                rates = phenotype.get('cell_transformations', {}).get('transformation_rates', {})
                return rates.get(target, 0.0)

        # Secretion / uptake: "X secretion" or "X uptake"
        if behavior.endswith(" secretion"):
            substrate = behavior[:-len(" secretion")]
            return phenotype.get('secretion', {}).get(substrate, {}).get('secretion_rate', 0.0)
        if behavior.endswith(" uptake"):
            substrate = behavior[:-len(" uptake")]
            return phenotype.get('secretion', {}).get(substrate, {}).get('uptake_rate', 0.0)

        # Cycle entry (first transition rate)
        if behavior == "cycle entry":
            cycle = phenotype.get('cycle', {})
            rates = cycle.get('transition_rates', [])
            if rates:
                return rates[0].get('rate', None)
            return None

        # Exit from cycle phase N
        if behavior.startswith("exit from cycle phase "):
            try:
                phase_idx = int(behavior[len("exit from cycle phase "):])
                cycle = phenotype.get('cycle', {})
                rates = cycle.get('transition_rates', [])
                if phase_idx < len(rates):
                    return rates[phase_idx].get('rate', None)
            except (ValueError, IndexError):
                pass
            return None

        # --- Cell interactions ---
        interactions = phenotype.get('cell_interactions', {})

        # Dead-cell phagocytosis
        if behavior == "phagocytose apoptotic cell":
            return interactions.get('apoptotic_phagocytosis_rate', 0.0)
        if behavior == "phagocytose necrotic cell":
            return interactions.get('necrotic_phagocytosis_rate', 0.0)
        if behavior == "phagocytose other dead cell":
            return interactions.get('other_dead_phagocytosis_rate', 0.0)

        # Per-cell-type interactions: "attack X", "phagocytose X", "fuse to X"
        if behavior.startswith("attack "):
            target = behavior[len("attack "):]
            return interactions.get('attack_rates', {}).get(target, 0.0)
        if behavior.startswith("phagocytose "):
            target = behavior[len("phagocytose "):]
            return interactions.get('live_phagocytosis_rates', {}).get(target, 0.0)
        if behavior.startswith("fuse to "):
            target = behavior[len("fuse to "):]
            return interactions.get('fusion_rates', {}).get(target, 0.0)

        # Attack parameters
        if behavior == "attack damage rate":
            return interactions.get('attack_damage_rate', 1.0)
        if behavior == "attack duration":
            return interactions.get('attack_duration', 0.1)

        # --- Mechanics ---
        mechanics = phenotype.get('mechanics', {})

        if behavior == "cell attachment rate":
            return mechanics.get('attachment_rate', 0.0)
        if behavior == "cell detachment rate":
            return mechanics.get('detachment_rate', 0.0)
        if behavior == "cell-cell adhesion":
            return mechanics.get('cell_cell_adhesion_strength', 0.4)
        if behavior == "cell-cell repulsion":
            return mechanics.get('cell_cell_repulsion_strength', 10.0)
        if behavior == "relative maximum adhesion distance":
            return mechanics.get('relative_maximum_adhesion_distance', 1.25)

        # --- Cell integrity ---
        integrity = phenotype.get('cell_integrity', {})

        if behavior == "damage rate":
            return integrity.get('damage_rate', 0.0)
        if behavior == "damage repair rate":
            return integrity.get('damage_repair_rate', 0.0)

        # --- Chemotaxis ---
        # "chemotactic response to X" behavior maps to advanced_chemotaxis sensitivity
        if behavior.startswith("chemotactic response to "):
            substrate = behavior[len("chemotactic response to "):]
            adv = phenotype.get('motility', {}).get('advanced_chemotaxis', {})
            sensitivities = adv.get('chemotactic_sensitivities', {})
            return sensitivities.get(substrate, 0.0)

    except (KeyError, TypeError, IndexError):
        return None

    # Unrecognized behavior — skip validation
    return None


@mcp.tool()
def add_single_cell_rule(
    cell_type: Annotated[str, Field(description="Name of existing cell type.")],
    signal: Annotated[str, Field(description="Signal name. Use list_all_available_signals() to see options.")],
    direction: Annotated[str, Field(description="'increases' or 'decreases' — whether the signal promotes or suppresses the behavior.")],
    behavior: Annotated[str, Field(description="Behavior name. Use list_all_available_behaviors() to see options.")],
    saturation_value: float = Field(default=1.0, description="Value of the behavior when the signal is at saturation (maximum effect)."),
    half_max: float = Field(default=0.5, description="Signal level at which the behavior is halfway between its base value and the saturation value."),
    hill_power: float = Field(default=4.0, description="Hill coefficient controlling the sharpness of the dose-response curve (typical: 1–8)."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Add a signal-behaviour rule that makes cells respond to environmental cues.

    Rules are Hill-function dose-response relationships: when `signal` rises,
    `behavior` is increased or decreased according to a Hill curve parameterised by
    `saturation_value`, `half_max`, and `hill_power`.
    Use `list_all_available_signals()` and `list_all_available_behaviors()` first.

    Returns:
        str: Confirmation with rule summary and export readiness indicator.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    # Basic validation
    if not cell_type or not cell_type.strip():
        return "Error: Cell type name cannot be empty"
    if not signal or not signal.strip():
        return "Error: Signal name cannot be empty"
    if direction not in ['increases', 'decreases']:
        return "Error: Direction must be 'increases' or 'decreases'"
    if not behavior or not behavior.strip():
        return "Error: Behavior name cannot be empty"
    if half_max <= 0:
        return "Error: Half-max value must be positive"
    if hill_power <= 0:
        return "Error: Hill power must be positive"
    
    # Update context from current config before adding rule
    update_signals_behaviors_context_from_config(session.config)

    # --- "From 0 towards 0" detection ---
    # If both the base_value (min_signal) and the XML default are 0, the Hill
    # function evaluates to 0 everywhere, making the rule silently useless.
    xml_default = _get_behavior_default_from_config(session, cell_type.strip(), behavior.strip())
    if xml_default is not None and xml_default == 0 and min_signal == 0:
        b = behavior.strip()
        ct = cell_type.strip()
        fix = None

        # Build a specific fix suggestion depending on behavior type
        if b.startswith("transition to ") or b.startswith("transform to "):
            prefix = "transition to " if b.startswith("transition to ") else "transform to "
            target = b[len(prefix):]
            fix = (
                f"Call `set_cell_transformation_rate(cell_type=\"{ct}\", "
                f"target_cell_type=\"{target}\", rate=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        elif b.endswith(" secretion") or b.endswith(" uptake"):
            substrate = b.rsplit(" ", 1)[0]
            rate_type = "secretion_rate" if b.endswith(" secretion") else "uptake_rate"
            fix = (
                f"Call `set_substrate_interaction(cell_type=\"{ct}\", "
                f"substrate=\"{substrate}\", {rate_type}=0.1)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        elif b in ("apoptosis", "necrosis"):
            rate_kwarg = f"{b}_rate"
            fix = (
                f"Call `configure_cell_parameters(cell_type=\"{ct}\", "
                f"{rate_kwarg}=0.0001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        # Per-cell-type interactions: attack, phagocytose (live), fuse
        elif b.startswith("attack "):
            target = b[len("attack "):]
            fix = (
                f"Call `set_cell_interaction(cell_type=\"{ct}\", "
                f"target_cell_type=\"{target}\", interaction_type=\"attack\", rate=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        elif b.startswith("phagocytose ") and b not in (
            "phagocytose apoptotic cell", "phagocytose necrotic cell", "phagocytose other dead cell"
        ):
            target = b[len("phagocytose "):]
            fix = (
                f"Call `set_cell_interaction(cell_type=\"{ct}\", "
                f"target_cell_type=\"{target}\", interaction_type=\"phagocytose\", rate=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        elif b.startswith("fuse to "):
            target = b[len("fuse to "):]
            fix = (
                f"Call `set_cell_interaction(cell_type=\"{ct}\", "
                f"target_cell_type=\"{target}\", interaction_type=\"fuse\", rate=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        # Dead-cell phagocytosis
        elif b in ("phagocytose apoptotic cell", "phagocytose necrotic cell", "phagocytose other dead cell"):
            param_map = {
                "phagocytose apoptotic cell": "apoptotic_phagocytosis_rate",
                "phagocytose necrotic cell": "necrotic_phagocytosis_rate",
                "phagocytose other dead cell": "other_dead_phagocytosis_rate",
            }
            param = param_map[b]
            fix = (
                f"Call `configure_cell_interactions(cell_type=\"{ct}\", "
                f"{param}=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        # Mechanics: attachment/detachment
        elif b in ("cell attachment rate", "cell detachment rate"):
            param = "attachment_rate" if "attachment" in b else "detachment_rate"
            fix = (
                f"Call `configure_cell_mechanics(cell_type=\"{ct}\", "
                f"{param}=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        # Cell integrity: damage/repair
        elif b in ("damage rate", "damage repair rate"):
            param = "damage_rate" if b == "damage rate" else "damage_repair_rate"
            fix = (
                f"Call `configure_cell_integrity(cell_type=\"{ct}\", "
                f"{param}=0.001)` first, "
                f"then re-add this rule with `min_signal=0`."
            )
        # Chemotaxis: "chemotactic response to X"
        elif b.startswith("chemotactic response to "):
            substrate = b[len("chemotactic response to "):]
            fix = (
                f"Call `set_advanced_chemotaxis(cell_type=\"{ct}\", "
                f"substrate=\"{substrate}\", sensitivity=0.5)` first, "
                f"then re-add this rule with `min_signal=0`."
            )

        if fix is None:
            fix = (
                f"Set a nonzero XML default for `{b}` before adding this rule, "
                f"or use a nonzero `min_signal` value."
            )

        return (
            f"**Error: \"From 0 towards 0\" rule detected — this rule would have no effect.**\n\n"
            f"Both `min_signal` (base_value={min_signal}) and the XML default for "
            f"`{b}` (={xml_default}) are 0.\n"
            f"The Hill function computes: rate = 0 + (0 − 0) × H(signal) = **0 always**.\n\n"
            f"**Fix:** {fix}"
        )

    # Add rule to configuration - check if we should use the new API or legacy
    try:
        # Try new API first (from test)
        from physicell_config.modules.cell_rules import CellRulesModule
        cell_rules = CellRulesModule(session.config)
        
        rule = {
            "cell_type": cell_type.strip(),
            "signal": signal.strip(),
            "direction": direction,
            "behavior": behavior.strip(),
            "min_signal": min_signal,
            "max_signal": max_signal,
            "hill_power": hill_power,
            "half_max": half_max
        }
        cell_rules.rules.append(rule)
        
        # Also add to legacy API for export compatibility
        rules = session.config.cell_rules_csv
        rules.add_rule(
            cell_type=cell_type.strip(),
            signal=signal.strip(),
            direction=direction,
            behavior=behavior.strip(),
            base_value=min_signal,  # Map min_signal to base_value
            half_max=half_max,
            hill_power=hill_power,
            apply_to_dead=0
        )
        
    except (ImportError, AttributeError):
        # Fall back to legacy CSV API only
        rules = session.config.cell_rules_csv
        rules.add_rule(
            cell_type=cell_type.strip(),
            signal=signal.strip(),
            direction=direction,
            behavior=behavior.strip(),
            base_value=min_signal,  # Map min_signal to base_value
            half_max=half_max,
            hill_power=hill_power,
            apply_to_dead=0
        )
    
    # Update session counters
    session.rules_count += 1
    session.mark_step_complete(WorkflowStep.RULES_CONFIGURED)

    # Clear any existing validation for this rule (e.g., contradictory → revised)
    session.rule_validations = [
        rv for rv in session.rule_validations
        if not (rv.cell_type == cell_type and rv.signal == signal.strip()
                and rv.direction == direction and rv.behavior == behavior.strip())
    ]

    # Ensure the ruleset is registered as enabled in XML config
    session.config.cell_rules.add_ruleset(
        "cell_rules", folder="./config", filename="cell_rules.csv", enabled=True
    )

    # Update legacy global for backward compatibility
    _set_legacy_config(session.config)

    # Track modification if loaded from XML
    if session.loaded_from_xml:
        session.mark_xml_modification()

    # Format result with interpolation range
    # For "increases": low signal → base_value (min_signal), high signal → xml_default (saturation)
    # For "decreases": low signal → xml_default (saturation), high signal → base_value (min_signal)
    result = f"**Cell rule added:**\n"
    result += f"- Rule: {cell_type} | {signal} {direction} → {behavior}\n"
    result += f"- Saturation value: {saturation_value}\n"
    result += f"- Half-max: {half_max}\n"
    result += f"- Hill power: {hill_power}\n"
    if xml_default is not None:
        if direction == "increases":
            result += f"- At low signal: {behavior} = {min_signal:g} (base_value)\n"
            result += f"- At high signal: {behavior} → {xml_default:g} (XML default/saturation)\n"
        else:
            result += f"- At high signal: {behavior} = {min_signal:g} (base_value)\n"
            result += f"- At low signal: {behavior} → {xml_default:g} (XML default/saturation)\n"
    result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
    
    # Check if ready for export based on core components (not arbitrary percentage)
    has_domain = WorkflowStep.DOMAIN_SETUP in session.completed_steps
    has_substrates = WorkflowStep.SUBSTRATES_ADDED in session.completed_steps
    has_cell_types = WorkflowStep.CELL_TYPES_ADDED in session.completed_steps
    ready_for_export = has_domain and has_substrates and has_cell_types
    
    if ready_for_export:
        session.mark_step_complete(WorkflowStep.READY_FOR_EXPORT)
        result += f"**Ready for export!** Use `export_xml_configuration()` to generate PhysiCell files."
    else:
        result += f"**Next step:** Add more rules or use `export_xml_configuration()` to finish."
    
    return result

@mcp.tool()
def add_physiboss_model(
    cell_type: Annotated[str, Field(description="Name of an existing cell type to attach the boolean network to.")],
    bnd_file: Annotated[str, Field(description="Absolute path to the MaBoSS .bnd boolean network file.")],
    cfg_file: Annotated[str, Field(description="Absolute path to the MaBoSS .cfg configuration file.")],
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Integrate a PhysiBoSS boolean network model into a cell type.

    The cell type must already exist. Requires PhysiBoSS support in the
    installed `physicell_config` package.
    After adding, call `configure_physiboss_settings()`, then
    `add_physiboss_input_link()` and `add_physiboss_output_link()`.

    Returns:
        str: Confirmation with model file paths and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_model(
            cell_type_name=cell_type,
            model_type="maboss",
            bnd_filename=bnd_file,
            cfg_filename=cfg_file
        )
        
        # Update session tracking
        session.physiboss_models_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_MODELS_ADDED)
        
        # Auto-create MaBoSS context if not exists to enable PhysiBoSS progress tracking
        if not session.maboss_context:
            from session_manager import MaBoSSContext
            session.maboss_context = MaBoSSContext(
                model_name="auto_created",
                bnd_file_path=bnd_file,
                cfg_file_path=cfg_file,
                available_nodes=[],
                output_nodes=[],
                simulation_results="",
                target_cell_type=cell_type,
                biological_context=""
            )
        
        result = f"**PhysiBoSS model added to {cell_type}:**\n"
        result += f"- Model file: {bnd_file}\n"
        result += f"- Config file: {cfg_file}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `configure_physiboss_settings()` to set intracellular parameters."
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS model: {str(e)}"

@mcp.tool()
def configure_physiboss_settings(
    cell_type: Annotated[str, Field(description="Name of an existing cell type with a PhysiBoSS model attached.")],
    intracellular_dt: float = Field(default=6.0, description="PhysiBoSS update interval in minutes. Should be a multiple of the phenotype time step."),
    time_stochasticity: int = Field(default=0, description="Time stochasticity level (0 = deterministic)."),
    scaling: float = Field(default=1.0, description="Scaling factor applied to the intracellular dynamics."),
    start_time: float = Field(default=0.0, description="Simulation time in minutes at which the boolean network starts running."),
    inheritance_global: bool = Field(default=False, description="If True, daughter cells inherit the parent's boolean node states globally."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Configure timing, stochasticity, and inheritance parameters for a PhysiBoSS model.
    Must be called after `add_physiboss_model()`. Repeat for each cell type.
    Checked with `list_all_available_signals()` and `list_all_available_behaviors()` the available signals and behaviors to use in `add_physiboss_input_link()` and `add_physiboss_output_link()`.
    Returns:
        str: Confirmation with the configured settings and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.set_intracellular_settings(
            cell_type_name=cell_type,
            intracellular_dt=intracellular_dt,
            time_stochasticity=time_stochasticity,
            scaling=scaling,
            start_time=start_time,
            inheritance_global=inheritance_global
        )
        
        # Update session tracking
        session.physiboss_settings_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_SETTINGS_CONFIGURED)
        
        result = f"**PhysiBoSS settings configured for {cell_type}:**\n"
        result += f"- Time step: {intracellular_dt} min\n"
        result += f"- Stochasticity: {time_stochasticity}\n"
        result += f"- Scaling: {scaling}\n"
        result += f"- Start time: {start_time} min\n"
        result += f"- Global inheritance: {inheritance_global}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_input_link()` to connect PhysiCell signals to boolean nodes."
        
        return result
    except Exception as e:
        return f"Error configuring PhysiBoSS settings: {str(e)}"

@mcp.tool()
def add_physiboss_input_link(
    cell_type: Annotated[str, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    physicell_signal: Annotated[str, Field(description="PhysiCell signal name. Use list_all_available_signals() to see options.")],
    boolean_node: Annotated[str, Field(description="MaBoSS boolean node name to drive (from the .bnd file).")],
    action: str = Field(default="activation", description="'activation' (signal turns node ON) or 'inhibition' (signal turns node OFF)."),
    threshold: float = Field(default=1.0, description="Signal level at which the node is toggled."),
    smoothing: int = Field(default=0, description="Smoothing level (0 = no smoothing)."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create an input link from a PhysiCell signal to a MaBoSS boolean node.
    Requires a PhysiBoSS model already attached to the cell type.
    Call `add_physiboss_output_link()` after all input links are defined.
    Returns:
        str: Confirmation with link details and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_input(
            cell_type_name=cell_type,
            physicell_name=physicell_signal,
            intracellular_name=boolean_node,
            action=action,
            threshold=threshold,
            smoothing=smoothing
        )
        
        # Update session tracking
        session.physiboss_input_links_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_INPUTS_LINKED)
        
        result = f"**PhysiBoSS input:** {physicell_signal} → {boolean_node}\n"
        result += f"- Action: {action}\n"
        result += f"- Threshold: {threshold}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `add_physiboss_output_link()` to connect boolean nodes to cell behaviors."
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS input link: {str(e)}"

@mcp.tool()
def add_physiboss_output_link(
    cell_type: Annotated[str, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    boolean_node: Annotated[str, Field(description="MaBoSS boolean node name whose state drives the behavior.")],
    physicell_behavior: Annotated[str, Field(description="PhysiCell behavior name to control. Use list_all_available_behaviors() to see options.")],
    action: str = Field(default="activation", description="'activation' (node ON increases behavior) or 'inhibition' (node ON decreases behavior)."),
    value: float = Field(default=1000000.0, description="Behavior value applied when the node is active."),
    base_value: float = Field(default=0.0, description="Baseline behavior value when the node is inactive."),
    smoothing: int = Field(default=0, description="Smoothing level (0 = no smoothing)."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Create an output link from a MaBoSS boolean node to a PhysiCell cell behaviour.
    Requires a PhysiBoSS model already attached to the cell type.
    Use `apply_physiboss_mutation()` afterward for genetic perturbations.
    Returns:
        str: Confirmation with link details and next step.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_output(
            cell_type_name=cell_type,
            physicell_name=physicell_behavior,
            intracellular_name=boolean_node,
            action=action,
            value=value,
            base_value=base_value,
            smoothing=smoothing
        )
        
        # Update session tracking
        session.physiboss_output_links_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_OUTPUTS_LINKED)
        
        result = f"**PhysiBoSS output:** {boolean_node} → {physicell_behavior}\n"
        result += f"- Action: {action}\n"
        result += f"- Active value: {value}\n"
        result += f"- Base value: {base_value}\n"
        result += f"- Smoothing: {smoothing}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Use `apply_physiboss_mutation()` for genetic perturbations"
        
        return result
    except Exception as e:
        return f"Error adding PhysiBoSS output link: {str(e)}"

@mcp.tool()
def apply_physiboss_mutation(
    cell_type: Annotated[str, Field(description="Name of an existing cell type with a PhysiBoSS model.")],
    node_name: Annotated[str, Field(description="MaBoSS boolean node name to fix (from the .bnd file).")],
    fixed_value: Annotated[int, Field(description="Fixed node state: 0 (always OFF / loss-of-function) or 1 (always ON / gain-of-function).")],
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Fix a MaBoSS boolean node to a constant value, simulating a genetic mutation.

    Requires a PhysiBoSS model already attached to the cell type.
    Call repeatedly to apply multiple mutations.

    Returns:
        str: Confirmation of the applied mutation.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "Error: Create simulation domain first using create_simulation_domain()"
    
    if not PHYSIBOSS_AVAILABLE:
        return "Error: PhysiBoSS module not available in this PhysiCell configuration package"
    
    try:
        # Use direct config.physiboss API (simpler and more reliable)
        session.config.physiboss.add_intracellular_mutation(
            cell_type_name=cell_type,
            intracellular_name=node_name,
            value=fixed_value
        )
        
        # Update session tracking
        session.physiboss_mutations_count += 1
        session.mark_step_complete(WorkflowStep.PHYSIBOSS_MUTATIONS_APPLIED)
        
        result = f"**Mutation applied:** {cell_type}.{node_name} = {fixed_value}\n"
        result += f"- Progress: {session.get_progress_percentage():.0f}%\n"
        result += f"**Next step:** Apply additional mutations or use `export_xml_configuration()` to finish."
        
        return result
    except Exception as e:
        return f"Error applying PhysiBoSS mutation: {str(e)}"

# ============================================================================
# UTILITY AND EXPORT TOOLS
# ============================================================================

@mcp.tool()
def get_simulation_summary(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session."),
) -> str:
    """Return a comprehensive summary of the current simulation configuration.

    Shows session progress, configured components (substrates, cell types, rules,
    PhysiBoSS models), completed workflow steps, and export readiness.

    Returns:
        str: Markdown summary of current simulation state.
    """
    session = get_current_session(session_id)
    if not session:
        return "No active session. Use `create_session()` to start."
    
    if not session.config:
        return "No simulation configured yet. Use `create_simulation_domain()` to start."
    
    # Get component counts using correct PhysiCell Settings API
    try:
        substrates = list(session.config.substrates.get_substrates().keys())
    except:
        substrates = []
    
    try:
        cell_types = list(session.config.cell_types.get_cell_types().keys())
    except:
        cell_types = []
    
    # Get rules count
    rules_count = 0
    try:
        rules_count = len(session.config.cell_rules.get_rules())
    except:
        rules_count = 0
    
    # Calculate progress
    progress = session.get_progress_percentage()
    
    result = f"## Simulation Summary\n\n"
    result += f"**Session:** {session.session_id[:8]}...\n"
    result += f"**Progress:** {progress:.0f}%\n\n"
    
    if session.scenario_context:
        result += f"**Scenario:** {session.scenario_context}\n\n"
    
    # Component details
    result += f"**Components:**\n"
    result += f"- **Substrates ({len(substrates)}):** {', '.join(substrates[:3])}{'...' if len(substrates) > 3 else 'None' if not substrates else ''}\n"
    result += f"- **Cell Types ({len(cell_types)}):** {', '.join(cell_types[:3])}{'...' if len(cell_types) > 3 else 'None' if not cell_types else ''}\n"
    result += f"- **Rules:** {rules_count}\n"
    if session.rule_validations:
        validated = len(session.rule_validations)
        strong = sum(1 for rv in session.rule_validations if rv.support_level == "strong")
        moderate = sum(1 for rv in session.rule_validations if rv.support_level == "moderate")
        flagged = sum(1 for rv in session.rule_validations if rv.support_level in ("unsupported", "contradictory"))
        result += f"- **Rules Validated:** {validated} ({strong} strong, {moderate} moderate"
        if flagged:
            result += f", {flagged} flagged"
        result += ")\n"
    if session.initial_cells_count > 0:
        result += f"- **Initial Cell Positions:** {session.initial_cells_count}\n"
    result += f"- **PhysiBoSS Models:** {session.physiboss_models_count}\n\n"
    
    # Workflow status
    completed_steps = [step.value.replace('_', ' ').title() for step in session.completed_steps]
    if completed_steps:
        result += f"**Completed Steps:**\n"
        for step in completed_steps:
            result += f"{step}\n"
        result += "\n"
    
    # Next recommendations
    recommendations = session.get_next_recommended_steps()
    if recommendations:
        result += f"**Next Steps:**\n"
        for rec in recommendations[:2]:
            result += f"• {rec}\n"
    
    # Export readiness based on core components, not arbitrary percentage
    has_domain = WorkflowStep.DOMAIN_SETUP in session.completed_steps
    has_substrates = WorkflowStep.SUBSTRATES_ADDED in session.completed_steps
    has_cell_types = WorkflowStep.CELL_TYPES_ADDED in session.completed_steps
    ready_for_export = has_domain and has_substrates and has_cell_types
    
    if ready_for_export:
        result += f"\n**Ready for export!** Use `export_xml_configuration()` to generate files."
    elif substrates and cell_types:
        result += f"\n**Basic setup complete.** Add rules or export now."
    else:
        result += f"\n**Setup incomplete.** Add substrates and cell types first."

    return result

@mcp.tool()
def export_xml_configuration(
    filename: str = Field(default="PhysiCell_settings.xml", description="Output filename for the XML configuration file."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Export the complete PhysiCell configuration to an XML file in the session artifact directory.

    The generated file can be passed directly to a PhysiCell executable.
    A simulation domain must have been created before calling this.

    Returns:
        str: Export status, file path, and execution instructions.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add components first."
    
    try:
        # Get simulation info for summary using correct API
        try:
            substrates = list(session.config.substrates.get_substrates().keys())
        except:
            substrates = []
        
        try:
            cell_types = list(session.config.cell_types.get_cell_types().keys())
        except:
            cell_types = []
        
        # Fallback to session counters if config access fails
        if not substrates and session.substrates_count > 0:
            substrates = [f"substrate_{i+1}" for i in range(session.substrates_count)]
        if not cell_types and session.cell_types_count > 0:
            cell_types = [f"cell_type_{i+1}" for i in range(session.cell_types_count)]
        
        # If initial cells have been placed, ensure XML config enables cells.csv
        if session.initial_cells_count > 0:
            session.config.initial_conditions.add_csv_file("cells.csv", "./config", enabled=True)
            session.config.set_number_of_cells(0)

        # If rules have been added, ensure the ruleset is enabled in XML
        if session.rules_count > 0:
            session.config.cell_rules.add_ruleset(
                "cell_rules", folder="./config", filename="cell_rules.csv", enabled=True
            )

        # Export XML configuration
        xml_content = session.config.generate_xml()

        # Ensure we write to a writable location
        output_dir = MCP_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always save as PhysiCell_settings.xml for consistency with create_physicell_project
        output_path = output_dir / "PhysiCell_settings.xml"

        with open(output_path, 'w') as f:
            f.write(xml_content)

        xml_size = len(xml_content) // 1024

        result = f"## XML Configuration Exported\n\n"
        result += f"**File:** {output_path} ({xml_size}KB)\n"
        
        # Show XML modification info if loaded from XML
        if session.loaded_from_xml and session.original_xml_path:
            original_name = Path(session.original_xml_path).name
            if session.xml_modification_count > 0:
                result += f"**Source:** Modified {session.xml_modification_count} times from {original_name}\n"
            else:
                result += f"**Source:** Exported from {original_name} (no modifications)\n"
        else:
            result += f"**Source:** Created from scratch\n"
        
        result += f"**Substrates:** {len(substrates)} ({', '.join(substrates[:3]) if substrates else 'None'}{'...' if len(substrates) > 3 else ''})\n"
        result += f"**Cell Types:** {len(cell_types)} ({', '.join(cell_types[:3]) if cell_types else 'None'}{'...' if len(cell_types) > 3 else ''})\n"
        if session.initial_cells_count > 0:
            result += f"**Initial Cells:** {session.initial_cells_count} (from cells.csv)\n"
        result += f"**Progress:** {session.get_progress_percentage():.0f}%\n\n"

        # Warn if cells placed but CSV not yet exported
        if session.initial_cells_count > 0:
            cells_csv_path = MCP_OUTPUT_DIR / "cells.csv"
            if not cells_csv_path.exists():
                result += f"**Warning:** {session.initial_cells_count} initial cells placed but cells.csv not exported yet. Call `export_cells_csv()` before creating the project.\n\n"

        result += f"**Next step:** Copy to PhysiCell project directory and run:\n"
        result += f"```bash\n./myproject {filename}\n```"
        
        return result
        
    except Exception as e:
        return f"Error exporting XML configuration: {str(e)}"

@mcp.tool()
def export_cell_rules_csv(
    filename: str = Field(default="cell_rules.csv", description="Output filename for the cell rules CSV file."),
    session_id: Optional[str] = Field(default=None, description="Session to use. Omit to use the active session."),
) -> str:
    """Export cell signal-behaviour rules to a CSV file in the session artifact directory.

    The generated file is used alongside `PhysiCell_settings.xml` by the PhysiCell executable.
    At least one rule must exist (added via `add_single_cell_rule()`).

    Returns:
        str: Export status and file path.
    """
    session = get_current_session(session_id)
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add components first."
    
    try:
        # Check rules via CellRulesModule API
        rule_count = len(session.config.cell_rules.get_rules())
        
        if rule_count == 0:
            return "**No cell rules to export**\n\nUse add_single_cell_rule() to create signal-behavior relationships first."
        
        # For now, export using the legacy CSV API (since that's what PhysiCell expects)
        # TODO: If new API rules exist, we might need to convert them to legacy format

        # Ensure we write to a writable location
        output_dir = MCP_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always save as cell_rules.csv for consistency with create_physicell_project
        output_path = output_dir / "cell_rules.csv"

        rules.generate_csv(str(output_path))

        # Fix CRLF → LF: the library's csv.writer uses \r\n (RFC 4180),
        # but PhysiCell doesn't strip \r, causing the last field to include
        # a trailing carriage return that silently breaks rule parsing.
        content = output_path.read_bytes()
        if b'\r\n' in content:
            output_path.write_bytes(content.replace(b'\r\n', b'\n'))

        # Ensure the ruleset is registered as enabled in XML config
        session.config.cell_rules.add_ruleset(
            "cell_rules", folder="./config", filename="cell_rules.csv", enabled=True
        )

        result = f"## Cell Rules CSV Exported\n\n"
        result += f"**File:** {output_path}\n"
        result += f"**Rules:** {rule_count}\n"
        result += f"**Progress:** {session.get_progress_percentage():.0f}%\n\n"
        result += f"**Next step:** Copy to PhysiCell project directory alongside XML configuration"
        
        return result
        
    except Exception as e:
        return f"Error exporting cell rules CSV: {str(e)}"

# ============================================================================
# LITERATURE VALIDATION TOOLS
# ============================================================================

@mcp.tool()
def get_rules_for_validation() -> str:
    """
    Export current cell rules in a structured format suitable for literature validation.

    Returns all rules as a JSON-compatible list that can be passed to the
    LiteratureValidation MCP server's validate_rules_batch() tool.

    Returns:
        str: Markdown with structured rule data for validation
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add rules first."

    # Get rules from legacy CSV API
    try:
        rules_api = session.config.cell_rules_csv
        raw_rules = rules_api.get_rules()
    except Exception:
        raw_rules = []

    if not raw_rules:
        return (
            "**No cell rules to validate.**\n\n"
            "Add rules with `add_single_cell_rule()` first."
        )

    # Format rules for validation
    rules_list = []
    for r in raw_rules:
        rule_dict = {
            "cell_type": r.get("cell_type", ""),
            "signal": r.get("signal", ""),
            "direction": r.get("direction", ""),
            "behavior": r.get("behavior", ""),
            "half_max": r.get("half_max"),
            "hill_power": r.get("hill_power"),
            "base_value": r.get("base_value"),
        }
        rules_list.append(rule_dict)

    result = f"## Rules Ready for Validation\n\n"
    result += f"**Total rules:** {len(rules_list)}\n\n"

    result += "### Rules\n"
    for i, r in enumerate(rules_list, 1):
        dir_arrow = ">" if r["direction"] == "increases" else "v"
        result += (
            f"{i}. **{r['cell_type']}** | {r['signal']} {dir_arrow} {r['behavior']} "
            f"(half_max={r['half_max']}, hill={r['hill_power']})\n"
        )

    result += f"\n### Structured Data (for LiteratureValidation MCP)\n"
    result += f"```json\n{json.dumps(rules_list, indent=2)}\n```\n\n"

    result += (
        "**Validation workflow:**\n"
        "1. Call `validate_rules_batch(name, rules)` on the LiteratureValidation MCP — Edison searches 150M+ papers automatically\n"
        "2. Call `get_validation_summary(name)` to review results\n"
        "3. Call `store_validation_results()` to save results back here\n"
        "4. Call `get_validation_report()` to generate the formal report"
    )
    return result


def _extract_verdict(raw_answer: str) -> str:
    """Extract VERDICT classification from raw PaperQA answer text.

    Uses the LAST match because the answer file contains the prompt template
    (with all VERDICT options listed) followed by PaperQA's actual answer.
    re.search() would match the first template option, not the real verdict.

    CONTRADICTORY is no longer a valid VERDICT — directional contradictions
    are caught by the DIRECTION check instead. If PaperQA still writes
    CONTRADICTORY, map it to 'weak'.
    """
    import re
    matches = re.findall(r"VERDICT:\s*(STRONG|MODERATE|WEAK|CONTRADICTORY|UNSUPPORTED)", raw_answer, re.IGNORECASE)
    if matches:
        level = matches[-1].lower()  # Last match = PaperQA's actual verdict
        if level == "contradictory":
            return "weak"
        return level
    return "unsupported"


def _extract_direction(raw_answer: str) -> str:
    """Extract DIRECTION from raw PaperQA answer text.

    Uses the LAST match because the answer file contains the prompt template
    (with all DIRECTION options listed) followed by PaperQA's actual answer.
    re.search() would match the first template option, not the real direction.

    Returns 'increases', 'decreases', or 'ambiguous'.
    """
    import re
    matches = re.findall(r"DIRECTION:\s*(INCREASES|DECREASES|AMBIGUOUS)", raw_answer, re.IGNORECASE)
    if matches:
        return matches[-1].lower()  # Last match = PaperQA's actual direction
    return "ambiguous"


def _find_answer_file(cell_type: str, signal: str, direction: str, behavior: str,
                      collection_name: str | None = None) -> Path | None:
    """Find a PaperQA answer file on disk for a given rule.

    Searches LiteratureValidation answer directories for matching files.
    Checks both direction-agnostic keys (new format) and legacy keys (with direction).
    If collection_name is provided, only searches that collection.

    Returns the Path to the answer file, or None if not found.
    """
    import re as _re
    lit_val_base = Path.home() / "Documents" / "LiteratureValidation"
    if not lit_val_base.exists():
        return None

    safe_key = _re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{behavior}")
    safe_key_legacy = _re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{direction}_{behavior}")

    # Determine which directories to search
    if collection_name:
        search_dirs = [lit_val_base / collection_name.strip().lower().replace(" ", "_")]
    else:
        try:
            search_dirs = [d for d in lit_val_base.iterdir() if d.is_dir()]
        except Exception:
            return None

    for collection_dir in search_dirs:
        for key in (safe_key, safe_key_legacy):
            candidate = collection_dir / "answers" / f"{key}.md"
            if candidate.exists():
                return candidate
    return None


@mcp.tool()
def store_validation_results(
    validations: list[dict],
) -> str:
    """
    Store literature validation results for cell rules in the session.

    The server reads PaperQA answer files directly from disk — the agent does NOT
    need to pass raw answer text. VERDICT and DIRECTION are extracted server-side
    from the authoritative answer files written by validate_rule().

    Each validation dict should contain:
    - cell_type (str): Cell type name
    - signal (str): Signal name
    - direction (str): 'increases' or 'decreases'
    - behavior (str): Behavior name
    - collection_name (str, optional): LiteratureValidation collection name (searches all if omitted)
    - raw_paperqa_answer (str, DEPRECATED): No longer needed — server reads from disk
    - evidence_summary (str, optional): Summary of evidence from literature
    - suggested_half_max (float, optional): Literature-suggested half-max value
    - suggested_hill_power (float, optional): Literature-suggested Hill coefficient
    - key_citations (list[str], optional): Key paper citations

    Args:
        validations: List of validation result dicts

    Returns:
        str: Summary of stored validation results
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured."

    if not validations:
        return "**Error:** No validation results provided."

    stored = 0
    errors = []

    for v in validations:
        cell_type = v.get("cell_type", "").strip()
        signal = v.get("signal", "").strip()
        direction = v.get("direction", "").strip()
        behavior = v.get("behavior", "").strip()
        collection_name = v.get("collection_name", "").strip() or None

        if not all([cell_type, signal, direction, behavior]):
            errors.append("Skipped entry with missing required fields")
            continue

        # Find the PaperQA answer file on disk (written by validate_rule())
        answer_file = _find_answer_file(cell_type, signal, direction, behavior, collection_name)

        if answer_file is None:
            errors.append(
                f"**REJECTED** '{cell_type}/{signal}/{behavior}': No PaperQA answer file found on disk. "
                f"You MUST call `validate_rule()` or `validate_rules_batch()` via the "
                f"LiteratureValidation MCP first. Do NOT skip validation."
            )
            continue

        # Read the authoritative answer from disk (not from agent input)
        try:
            file_content = answer_file.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"Failed to read answer file for '{cell_type}/{signal}/{behavior}': {e}")
            continue

        # Extract support level from PaperQA's own VERDICT line
        support_level = _extract_verdict(file_content)

        # Extract literature direction from PaperQA's DIRECTION line
        literature_direction = _extract_direction(file_content)

        # Compute direction match
        if literature_direction == "ambiguous":
            direction_match = None
        elif literature_direction == direction:
            direction_match = True
        else:
            direction_match = False

        # Auto-flag direction mismatches as contradictory
        if direction_match is False:
            support_level = "contradictory"

        result = RuleValidationResult(
            cell_type=cell_type,
            signal=signal,
            direction=direction,
            behavior=behavior,
            support_level=support_level,
            evidence_summary=v.get("evidence_summary", ""),
            raw_paperqa_answer=file_content,
            suggested_half_max=v.get("suggested_half_max"),
            suggested_hill_power=v.get("suggested_hill_power"),
            key_citations=v.get("key_citations", []),
            literature_direction=literature_direction,
            direction_match=direction_match,
        )
        session.rule_validations.append(result)
        stored += 1

    # Check if ALL rules have been validated before marking step complete
    try:
        current_rules = session.config.cell_rules_csv.get_rules()
    except Exception:
        current_rules = []

    validated_keys = {
        (rv.cell_type, rv.signal, rv.direction, rv.behavior)
        for rv in session.rule_validations
    }
    all_rule_keys = {
        (r.get("cell_type", ""), r.get("signal", ""),
         r.get("direction", ""), r.get("behavior", ""))
        for r in current_rules
    }
    missing_rules = all_rule_keys - validated_keys

    if stored > 0 and not missing_rules:
        session.mark_step_complete(WorkflowStep.RULES_VALIDATED)

    output = f"## Validation Results Stored\n\n"
    output += f"**Stored:** {stored} / {len(validations)} results\n\n"

    if errors:
        output += "**Warnings:**\n"
        for err in errors[:5]:
            output += f"- {err}\n"
        output += "\n"

    if missing_rules:
        output += f"### Missing Validations ({len(missing_rules)} rules not yet validated)\n\n"
        output += "The following rules have not been validated yet:\n\n"
        for ct, sig, dir_, beh in sorted(missing_rules):
            output += f"- {ct} | {sig} {dir_} {beh}\n"
        output += (
            "\nCall `validate_rules_batch()` with these rules, then "
            "`store_validation_results()` again to complete validation.\n\n"
        )

    # Summary by support level
    level_counts: dict[str, int] = {}
    for rv in session.rule_validations:
        level_counts[rv.support_level] = level_counts.get(rv.support_level, 0) + 1

    output += "### Support Level Summary\n"
    for level in ["strong", "moderate", "weak", "contradictory", "unsupported"]:
        count = level_counts.get(level, 0)
        if count > 0:
            output += f"- **{level.capitalize()}:** {count}\n"

    # Flag actionable items
    flagged = [rv for rv in session.rule_validations
               if rv.support_level in ("unsupported", "contradictory")]
    if flagged:
        output += "\n### Rules Needing Attention\n"
        for rv in flagged:
            dir_arrow = ">" if rv.direction == "increases" else "v"
            output += f"- **{rv.support_level.upper()}**: {rv.cell_type} | {rv.signal} {dir_arrow} {rv.behavior}\n"

    # Show parameter suggestions
    suggestions = [rv for rv in session.rule_validations
                   if rv.suggested_half_max is not None or rv.suggested_hill_power is not None]
    if suggestions:
        output += "\n### Suggested Parameter Adjustments\n"
        for rv in suggestions:
            output += f"- **{rv.cell_type} | {rv.signal} -> {rv.behavior}:**"
            if rv.suggested_half_max is not None:
                output += f" half_max={rv.suggested_half_max}"
            if rv.suggested_hill_power is not None:
                output += f" hill_power={rv.suggested_hill_power}"
            output += "\n"

    output += f"\n**Next step:** Use `get_validation_report()` for a full report."
    return output


@mcp.tool()
def get_validation_report() -> str:
    """
    Generate a comprehensive validation report for all cell rules.

    Shows each rule's validation status, evidence summary, and any
    suggested parameter changes from published literature.

    Returns:
        str: Markdown-formatted validation report
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured."

    if not session.rule_validations:
        return (
            "**No validation results stored.**\n\n"
            "Use the LiteratureValidation MCP server to validate rules, "
            "then call `store_validation_results()` to save results here.\n\n"
            "**Quick start:**\n"
            "1. `get_rules_for_validation()` — export rules\n"
            "2. `validate_rules_batch(name, rules)` — validate against literature (LiteratureValidation MCP)\n"
            "3. `store_validation_results()` — save results here\n"
            "4. `get_validation_report()` — generate the formal report"
        )

    # Get current rules for cross-reference
    try:
        rules_api = session.config.cell_rules_csv
        current_rules = rules_api.get_rules()
    except Exception:
        current_rules = []

    report = "## Literature Validation Report\n\n"
    report += f"**Rules validated:** {len(session.rule_validations)}\n"
    report += f"**Total rules in model:** {session.rules_count}\n\n"

    # Support distribution
    level_counts: dict[str, int] = {}
    for rv in session.rule_validations:
        level_counts[rv.support_level] = level_counts.get(rv.support_level, 0) + 1

    report += "### Overall Assessment\n"
    for level in ["strong", "moderate", "weak", "contradictory", "unsupported"]:
        count = level_counts.get(level, 0)
        if count > 0:
            report += f"- **{level.capitalize()}:** {count}\n"
    report += "\n"

    # Detailed per-rule report
    report += "### Detailed Results\n\n"
    for i, rv in enumerate(session.rule_validations, 1):
        dir_arrow = ">" if rv.direction == "increases" else "v"
        report += f"#### {i}. {rv.cell_type} | {rv.signal} {dir_arrow} {rv.behavior}\n"
        report += f"**Support:** {rv.support_level.upper()}\n"

        # Show direction match status
        if rv.direction_match is False:
            report += (
                f"\n**DIRECTION MISMATCH** — Literature says {rv.signal} "
                f"**{rv.literature_direction}** {rv.behavior}, but rule proposes "
                f"{rv.signal} **{rv.direction}** {rv.behavior}. "
                f"The rule direction must be changed.\n"
            )
        elif rv.direction_match is True:
            report += f"**Direction:** Confirmed by literature ({rv.literature_direction})\n"
        elif rv.literature_direction:
            report += f"**Direction:** Could not be determined from literature\n"
        report += "\n"

        # Show raw PaperQA answer (audit trail)
        if rv.raw_paperqa_answer:
            raw = rv.raw_paperqa_answer
            if len(raw) > 800:
                raw = raw[:800] + "..."
            report += f"**PaperQA answer:**\n> {raw.replace(chr(10), chr(10) + '> ')}\n\n"
        elif rv.evidence_summary:
            # Fallback for legacy results without raw answer
            summary = rv.evidence_summary
            if len(summary) > 500:
                summary = summary[:500] + "..."
            report += f"{summary}\n\n"

        if rv.key_citations:
            report += "**Key citations:**\n"
            for cite in rv.key_citations[:5]:
                report += f"- {cite}\n"
            report += "\n"

        if rv.suggested_half_max is not None or rv.suggested_hill_power is not None:
            report += "**Suggested parameters:**"
            if rv.suggested_half_max is not None:
                report += f" half_max={rv.suggested_half_max}"
            if rv.suggested_hill_power is not None:
                report += f" hill_power={rv.suggested_hill_power}"
            report += "\n\n"

    # Unvalidated rules
    validated_keys = {
        (rv.cell_type, rv.signal, rv.direction, rv.behavior)
        for rv in session.rule_validations
    }
    unvalidated = []
    for r in current_rules:
        key = (r.get("cell_type", ""), r.get("signal", ""),
               r.get("direction", ""), r.get("behavior", ""))
        if key not in validated_keys:
            unvalidated.append(r)

    if unvalidated:
        report += f"### Unvalidated Rules ({len(unvalidated)} remaining)\n\n"
        report += "The following rules have not been validated:\n\n"
        for r in unvalidated:
            report += f"- {r.get('cell_type', '?')} | {r.get('signal', '?')} {r.get('direction', '?')} {r.get('behavior', '?')}\n"
        report += (
            "\nCall `validate_rules_batch()` with these rules, then "
            "`store_validation_results()` to complete validation.\n\n"
        )

    # Action required for contradictory rules (including direction mismatches)
    contradictory = [rv for rv in session.rule_validations if rv.support_level == "contradictory"]
    if contradictory:
        direction_mismatches = [rv for rv in contradictory if rv.direction_match is False]
        other_contradictions = [rv for rv in contradictory if rv.direction_match is not False]

        report += "\n### Contradictory Rules\n\n"
        report += (
            f"**{len(contradictory)} rule(s) are CONTRADICTORY.** Consider revising "
            "these rules based on the literature evidence.\n\n"
        )

        if direction_mismatches:
            report += (
                f"**Direction Mismatches ({len(direction_mismatches)}):** The literature-determined direction "
                "contradicts the proposed rule direction. Fix by changing the `direction` parameter in "
                "`add_single_cell_rule()`.\n\n"
            )
            for rv in direction_mismatches:
                report += (
                    f"- **{rv.cell_type}** | {rv.signal} {rv.direction} {rv.behavior} "
                    f"— literature says **{rv.literature_direction}**\n"
                )
            report += "\n"

        if other_contradictions:
            report += f"**Other Contradictions ({len(other_contradictions)}):**\n"
            for rv in other_contradictions:
                report += f"- {rv.cell_type} | {rv.signal} {rv.direction} {rv.behavior}\n"
            report += "\n"

        report += (
            "**Steps to resolve:**\n"
            "1. Review the PaperQA evidence above\n"
            "2. Modify the rule with `add_single_cell_rule()` — for direction mismatches, change the "
            "`direction` parameter; for other contradictions, adjust half_max, hill_power, or base value\n"
            "3. Re-validate the modified rule with `validate_rule()` or `validate_rules_batch()`\n"
            "4. `store_validation_results()` to update the stored results\n"
            "5. `get_validation_report()` to regenerate this report\n\n"
        )

    report += f"**Progress:** {session.get_progress_percentage():.0f}%"

    # Export report to output directory
    try:
        MCP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = MCP_OUTPUT_DIR / "validation_report.md"
        report_path.write_text(report, encoding="utf-8")
        report += f"\n\n**Report exported to:** `{report_path}`"
    except Exception:
        pass

    return report


# ============================================================================
# INITIAL CELL PLACEMENT TOOLS
# ============================================================================

@mcp.tool()
def place_initial_cells(
    cell_type: str,
    pattern: str = "random_disc",
    num_cells: int = 100,
    center_x: float = 0.0,
    center_y: float = 0.0,
    radius: float = 100.0,
    inner_radius: float = 0.0,
    x_min: float = -100.0,
    x_max: float = 100.0,
    y_min: float = -100.0,
    y_max: float = 100.0,
    spacing: float = 20.0,
    z: float = 0.0
) -> str:
    """
    Place a batch of cells with specific spatial positions for initial conditions.
    Generates x,y,z coordinates that will be written to cells.csv.
    Call multiple times with different cell types or patterns to build up the initial layout.

    PREREQUISITE: add_single_cell_type() - Cell types must exist before placing cells.
    NEXT STEP: After all placements, call export_cells_csv() to write the cells.csv file,
    then export_xml_configuration() to generate the XML (initial conditions will be enabled automatically).

    Supported patterns:
    - 'random_disc': Uniformly distributed cells in a circular area (use center_x, center_y, radius, num_cells)
    - 'random_rectangle': Uniformly distributed cells in a rectangular area (use x_min, x_max, y_min, y_max, num_cells)
    - 'single': Place one cell at a specific position (use center_x, center_y; num_cells is ignored)
    - 'grid': Evenly spaced grid of cells (use x_min, x_max, y_min, y_max, spacing; num_cells is ignored)
    - 'annular': Uniformly distributed cells in a ring (use center_x, center_y, radius, inner_radius, num_cells)

    Args:
        cell_type: Name of an existing cell type (must match a type added via add_single_cell_type)
        pattern: Spatial pattern - 'random_disc', 'random_rectangle', 'single', 'grid', or 'annular'
        num_cells: Number of cells to place (used by random_disc, random_rectangle, annular)
        center_x: Center X coordinate for disc/annular/single patterns (default: 0.0)
        center_y: Center Y coordinate for disc/annular/single patterns (default: 0.0)
        radius: Outer radius for disc/annular patterns in microns (default: 100.0)
        inner_radius: Inner radius for annular pattern in microns (default: 0.0)
        x_min: Left bound for rectangle/grid patterns (default: -100.0)
        x_max: Right bound for rectangle/grid patterns (default: 100.0)
        y_min: Bottom bound for rectangle/grid patterns (default: -100.0)
        y_max: Top bound for rectangle/grid patterns (default: 100.0)
        spacing: Cell spacing for grid pattern in microns (default: 20.0)
        z: Z coordinate for all placed cells (default: 0.0, use 0 for 2D simulations)

    Returns:
        str: Summary of placed cells with count and spatial bounds
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured. Use `create_simulation_domain()` and `add_single_cell_type()` first."

    # Validate cell type exists
    try:
        available_types = list(session.config.cell_types.get_cell_types().keys())
    except:
        available_types = []

    if not available_types:
        return "**Error:** No cell types defined. Use `add_single_cell_type()` first."

    if cell_type not in available_types:
        return f"**Error:** Cell type '{cell_type}' not found. Available types: {', '.join(available_types)}"

    # Validate pattern
    valid_patterns = ['random_disc', 'random_rectangle', 'single', 'grid', 'annular']
    if pattern not in valid_patterns:
        return f"**Error:** Invalid pattern '{pattern}'. Must be one of: {', '.join(valid_patterns)}"

    # Generate cell coordinates based on pattern
    new_cells = []

    if pattern == "single":
        new_cells.append({"x": center_x, "y": center_y, "z": z, "type": cell_type})

    elif pattern == "random_disc":
        if radius <= 0:
            return "**Error:** Radius must be positive for random_disc pattern."
        if num_cells <= 0:
            return "**Error:** num_cells must be positive."
        for _ in range(num_cells):
            theta = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.uniform(0, 1))
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            new_cells.append({"x": x, "y": y, "z": z, "type": cell_type})

    elif pattern == "random_rectangle":
        if x_min >= x_max or y_min >= y_max:
            return "**Error:** x_min must be < x_max and y_min must be < y_max."
        if num_cells <= 0:
            return "**Error:** num_cells must be positive."
        for _ in range(num_cells):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            new_cells.append({"x": x, "y": y, "z": z, "type": cell_type})

    elif pattern == "grid":
        if x_min >= x_max or y_min >= y_max:
            return "**Error:** x_min must be < x_max and y_min must be < y_max."
        if spacing <= 0:
            return "**Error:** Spacing must be positive."
        x_pos = x_min
        while x_pos <= x_max:
            y_pos = y_min
            while y_pos <= y_max:
                new_cells.append({"x": x_pos, "y": y_pos, "z": z, "type": cell_type})
                y_pos += spacing
            x_pos += spacing

    elif pattern == "annular":
        if radius <= 0:
            return "**Error:** Radius must be positive for annular pattern."
        if inner_radius < 0:
            return "**Error:** inner_radius must be non-negative."
        if inner_radius >= radius:
            return "**Error:** inner_radius must be less than radius."
        if num_cells <= 0:
            return "**Error:** num_cells must be positive."
        for _ in range(num_cells):
            theta = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.uniform(inner_radius**2, radius**2))
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
            new_cells.append({"x": x, "y": y, "z": z, "type": cell_type})

    if not new_cells:
        return "**Warning:** No cells were generated. Check your parameters."

    # Store in session
    session.initial_cells.extend(new_cells)
    session.initial_cells_count = len(session.initial_cells)
    session.mark_step_complete(WorkflowStep.INITIAL_CONDITIONS_SET)

    # Immediately update config so XML generation always reflects cell placements
    session.config.initial_conditions.add_csv_file("cells.csv", "./config", enabled=True)
    session.config.set_number_of_cells(0)

    if session.loaded_from_xml:
        session.mark_xml_modification()

    _set_legacy_config(session.config)

    # Compute bounds of newly placed cells
    xs = [c["x"] for c in new_cells]
    ys = [c["y"] for c in new_cells]
    placed_count = len(new_cells)

    result = f"**Placed {placed_count} {cell_type} cells** ({pattern})\n"
    result += f"- X range: [{min(xs):.1f}, {max(xs):.1f}] \u03bcm\n"
    result += f"- Y range: [{min(ys):.1f}, {max(ys):.1f}] \u03bcm\n"
    result += f"- Z: {z}\n"
    result += f"- **Total cells in session:** {session.initial_cells_count}\n"

    # Warn if cells are outside domain bounds
    try:
        domain_info = session.config.domain.get_info()
        dx_min, dx_max = domain_info['x_min'], domain_info['x_max']
        dy_min, dy_max = domain_info['y_min'], domain_info['y_max']
        outside = sum(1 for c in new_cells
                      if c["x"] < dx_min or c["x"] > dx_max
                      or c["y"] < dy_min or c["y"] > dy_max)
        if outside > 0:
            result += f"\n**Warning:** {outside} cells are outside the simulation domain "
            result += f"([{dx_min}, {dx_max}] x [{dy_min}, {dy_max}])\n"
    except:
        pass

    if num_cells > 10000:
        result += "\n**Note:** Large cell counts may slow simulation startup.\n"

    result += f"\n**Next step:** Place more cells or call `export_cells_csv()` to write the file."
    result += f"\n**Progress:** {session.get_progress_percentage():.0f}%"

    return result

@mcp.tool()
def remove_initial_cells(cell_type: Optional[str] = None) -> str:
    """
    Remove placed initial cells from the current session.
    Use this to clear and redo cell placements before exporting.

    Args:
        cell_type: If provided, remove only cells of this type. If omitted, remove all cells.

    Returns:
        str: Summary of removed cells and remaining count
    """
    session = get_current_session()
    if not session:
        return "**Error:** No active session."

    if not session.initial_cells:
        return "**No initial cells to remove.** Use `place_initial_cells()` to add cells first."

    previous_count = len(session.initial_cells)

    if cell_type is None:
        session.initial_cells.clear()
        removed = previous_count
    else:
        session.initial_cells = [c for c in session.initial_cells if c["type"] != cell_type]
        removed = previous_count - len(session.initial_cells)

    session.initial_cells_count = len(session.initial_cells)

    if session.initial_cells_count == 0:
        session.completed_steps.discard(WorkflowStep.INITIAL_CONDITIONS_SET)

    result = f"**Removed {removed} cells**"
    if cell_type:
        result += f" of type '{cell_type}'"
    result += f"\n- **Remaining:** {session.initial_cells_count} cells"

    if session.initial_cells_count > 0:
        # Show remaining type breakdown
        type_counts: Dict[str, int] = {}
        for c in session.initial_cells:
            type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
        for t, count in type_counts.items():
            result += f"\n  - {t}: {count}"

    return result

@mcp.tool()
def get_initial_conditions_summary() -> str:
    """
    Get a summary of current initial cell placements.
    Shows count by cell type, spatial bounds, and total cells.
    Use this to review placements before exporting cells.csv.

    Returns:
        str: Markdown-formatted summary of initial cell positions
    """
    session = get_current_session()
    if not session:
        return "**Error:** No active session."

    if not session.initial_cells:
        return "**No initial cells placed.**\n\nUse `place_initial_cells()` to add cells with spatial patterns like random_disc, grid, annular, etc."

    # Group by type
    type_counts: Dict[str, int] = {}
    for c in session.initial_cells:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1

    # Compute overall bounds
    all_x = [c["x"] for c in session.initial_cells]
    all_y = [c["y"] for c in session.initial_cells]

    result = f"## Initial Cell Placement Summary\n\n"
    result += f"**Total cells:** {session.initial_cells_count}\n\n"
    result += f"**By cell type:**\n"
    for t, count in sorted(type_counts.items()):
        result += f"- {t}: {count}\n"

    result += f"\n**Spatial bounds:**\n"
    result += f"- X: [{min(all_x):.1f}, {max(all_x):.1f}] \u03bcm\n"
    result += f"- Y: [{min(all_y):.1f}, {max(all_y):.1f}] \u03bcm\n"

    # Check against domain
    try:
        domain_info = session.config.domain.get_info()
        dx_min, dx_max = domain_info['x_min'], domain_info['x_max']
        dy_min, dy_max = domain_info['y_min'], domain_info['y_max']
        outside = sum(1 for c in session.initial_cells
                      if c["x"] < dx_min or c["x"] > dx_max
                      or c["y"] < dy_min or c["y"] > dy_max)
        result += f"\n**Domain:** [{dx_min}, {dx_max}] x [{dy_min}, {dy_max}] \u03bcm\n"
        if outside > 0:
            result += f"**Warning:** {outside} cells are outside the domain bounds.\n"
        else:
            result += f"All cells are within the domain bounds.\n"
    except:
        pass

    result += f"\n**Next step:** Call `export_cells_csv()` to write cells.csv, then `export_xml_configuration()`."

    return result

@mcp.tool()
def export_cells_csv(filename: str = "cells.csv") -> str:
    """
    When the user asks to export initial cell positions, save cell layout, or generate cells CSV,
    this function writes the cells.csv file with all placed cell positions for PhysiCell.

    PREREQUISITE: place_initial_cells() - Must have placed cells first.
    NEXT STEP: export_xml_configuration() - Export XML (initial conditions will be enabled automatically).
    Then create_physicell_project() - cells.csv will be copied to the project automatically.

    Input parameters:
    filename (str): Name for CSV file (default: 'cells.csv')

    Returns:
    str: Markdown-formatted export status with file details
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No simulation configured. Create domain and add components first."

    if not session.initial_cells:
        return "**No initial cells to export.**\n\nUse `place_initial_cells()` to add cells with spatial patterns first."

    try:
        # Write CSV file
        output_dir = MCP_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always save as cells.csv for consistency with create_physicell_project
        output_path = output_dir / "cells.csv"

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["x", "y", "z", "type"])
            for cell in session.initial_cells:
                writer.writerow([cell["x"], cell["y"], cell["z"], cell["type"]])

        # Update XML config to reference and enable cells.csv
        session.config.initial_conditions.add_csv_file("cells.csv", "./config", enabled=True)
        session.config.set_number_of_cells(0)

        session.mark_step_complete(WorkflowStep.INITIAL_CONDITIONS_SET)
        _set_legacy_config(session.config)

        # Build result summary
        type_counts: Dict[str, int] = {}
        for c in session.initial_cells:
            type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1

        result = f"## Cells CSV Exported\n\n"
        result += f"**File:** {output_path}\n"
        result += f"**Total cells:** {session.initial_cells_count}\n"
        for t, count in sorted(type_counts.items()):
            result += f"- {t}: {count}\n"
        result += f"\n**XML config updated:** initial_conditions enabled, number_of_cells set to 0\n"
        result += f"**Progress:** {session.get_progress_percentage():.0f}%\n\n"
        result += f"**Next step:** Call `export_xml_configuration()` then `create_physicell_project()` (cells.csv will be copied automatically)."

        return result

    except Exception as e:
        return f"Error exporting cells CSV: {str(e)}"

# ============================================================================
# HELPER FUNCTIONS (inspired by NeKo)
# ============================================================================

def _resolve_output_folder(simulation_id: Optional[str] = None,
                           output_folder: Optional[str] = None) -> tuple:
    """Resolve simulation output folder from simulation_id or explicit path.

    Returns:
        (Path, Optional[str]): (folder_path, error_message).
        If error_message is not None, folder_path should not be used.
    """
    if simulation_id:
        with simulations_lock:
            if simulation_id not in running_simulations:
                return None, f"Error: Simulation '{simulation_id}' not found."
            sim = running_simulations[simulation_id]
            folder = Path(sim.output_folder)
    elif output_folder:
        folder = Path(output_folder)
    else:
        folder = PHYSICELL_ROOT / "output"

    if not folder.exists():
        return None, f"Error: Output folder not found: {folder}"

    return folder, None


def clean_for_markdown(text: str) -> str:
    """
    Clean text for markdown output by removing problematic characters.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.replace("|", "\\|").replace("\n", " ").strip()

@mcp.tool()
def list_generated_files(
    session_id: Optional[str] = Field(default=None, description="Session to query. Omit to use the active session. Pass 'all' to list files across every session."),
) -> str:
    """List PhysiCell artifact files (XML/CSV) for the active session.

    Returns:
        str: Grouped list of XML and CSV artifact files.
    """
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
    else:
        session = get_current_session(session_id)
        if session is None:
            return "**No active session.** Use `create_session()` first."
        files = list_artifacts(_SERVER_ROOT, session_id=session.session_id)

    xml_files = [f for f in files if str(f).endswith(".xml")]
    csv_files = [f for f in files if str(f).endswith(".csv")]

    result = "## Generated Artifact Files\n\n"

    if xml_files:
        result += "**XML files:**\n"
        for f in xml_files:
            result += f"- {f}\n"
        result += "\n"

    if csv_files:
        result += "**CSV files:**\n"
        for f in csv_files:
            result += f"- {f}\n"
        result += "\n"

    if not xml_files and not csv_files:
        result += "No PhysiCell artifact files found."

    return result


@mcp.tool()
def clean_generated_files(
    session_id: Optional[str] = Field(default=None, description="Session to clean. Omit to use the active session."),
) -> str:
    """Remove all artifact files (XML, CSV, etc.) for the active session.

    Returns:
        str: Count of files removed.
    """
    session = get_current_session(session_id)
    if session is None:
        return "**No active session.** Use `create_session()` first."

    try:
        count = clean_artifacts(_SERVER_ROOT, session.session_id)
        return f"**Cleaned {count} artifact file(s)** for session {session.session_id[:8]}..."
    except Exception as e:
        return f"Error during cleanup: {str(e)}"

# ============================================================================
# PHYSICELL PROJECT CREATION AND EXECUTION TOOLS
# ============================================================================

@mcp.tool()
def create_physicell_project(project_name: str, copy_generated_config: bool = True) -> str:
    """
    Create a complete PhysiCell project directory from template.

    IMPORTANT: Before calling this tool, you MUST complete these steps in order:
    1. create_session() - Initialize a simulation session
    2. create_simulation_domain() - Set up spatial domain and time
    3. add_single_substrate() - Add substrates (oxygen, glucose, etc.)
    4. add_single_cell_type() - Add cell types (cancer cells, immune cells, etc.)
    5. add_single_cell_rule() - Define cell behaviors and responses
    6. export_xml_configuration() - Export the XML config file
    7. export_cell_rules_csv() - Export the cell rules CSV file

    Only AFTER completing all above steps, call this tool to create the project.

    Args:
        project_name: Name for the project (alphanumeric and underscores only)
        copy_generated_config: Whether to copy XML and CSV from current session (default: True)

    Returns:
        str: Project creation status and next steps
    """
    # Validate project name
    if not project_name or not re.match(r'^[a-zA-Z0-9_]+$', project_name):
        return "Error: Project name must contain only letters, numbers, and underscores"

    # Check if config files exist (validate workflow was followed)
    if copy_generated_config:
        xml_file = MCP_OUTPUT_DIR / "PhysiCell_settings.xml"
        csv_file = MCP_OUTPUT_DIR / "cell_rules.csv"

        if not xml_file.exists():
            return """**Error: Configuration files not found!**

You must complete the simulation setup workflow BEFORE creating a project:

1. `create_session()` - Initialize a simulation session
2. `create_simulation_domain()` - Set up spatial domain and time
3. `add_single_substrate()` - Add substrates (oxygen, glucose, etc.)
4. `add_single_cell_type()` - Add cell types (cancer, immune, etc.)
5. `add_single_cell_rule()` - Define cell behaviors
6. `export_xml_configuration()` - **Export the XML config file**
7. `export_cell_rules_csv()` - Export the cell rules CSV

Please complete these steps first, then call create_physicell_project() again."""

    # Check if project already exists
    project_dir = USER_PROJECTS_DIR / project_name
    if project_dir.exists():
        return f"Error: Project '{project_name}' already exists at {project_dir}"

    # Check if template exists
    if not TEMPLATE_DIR.exists():
        return f"Error: Template directory not found at {TEMPLATE_DIR}"

    try:
        # Create user_projects directory if it doesn't exist
        USER_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

        # Copy template to new project directory
        import shutil
        shutil.copytree(TEMPLATE_DIR, project_dir)

        result = f"**Project created:** {project_name}\n"
        result += f"**Location:** {project_dir}\n\n"

        # Copy generated config files if requested
        if copy_generated_config:
            session = get_current_session()
            if not session or not session.config:
                result += "**Warning:** No active session found. Config files not copied.\n"
                result += "Use export_xml_configuration() and export_cell_rules_csv() first.\n"
            else:
                # Check for exported files
                xml_file = MCP_OUTPUT_DIR / "PhysiCell_settings.xml"
                csv_file = MCP_OUTPUT_DIR / "cell_rules.csv"

                config_dir = project_dir / "config"
                config_dir.mkdir(exist_ok=True)

                if xml_file.exists():
                    shutil.copy(xml_file, config_dir / "PhysiCell_settings.xml")
                    result += "**Copied:** PhysiCell_settings.xml to config/\n"
                else:
                    result += "**Warning:** PhysiCell_settings.xml not found. Export it first.\n"

                if csv_file.exists():
                    shutil.copy(csv_file, config_dir / "cell_rules.csv")
                    result += "**Copied:** cell_rules.csv to config/\n"
                else:
                    result += "**Note:** cell_rules.csv not found (optional).\n"

                # Copy cells.csv if it exists (initial cell positions)
                cells_csv_file = MCP_OUTPUT_DIR / "cells.csv"
                if cells_csv_file.exists():
                    shutil.copy(cells_csv_file, config_dir / "cells.csv")
                    result += "**Copied:** cells.csv to config/\n"

        result += f"\n**Next step:** Use `compile_physicell_project('{project_name}')` to compile the project."

        return result

    except Exception as e:
        return f"Error creating project: {str(e)}"

@mcp.tool()
def compile_physicell_project(project_name: str, clean_first: bool = False) -> str:
    """
    Compile a PhysiCell project using make.

    PREREQUISITE: Call create_physicell_project() first to create the project.
    NEXT STEP: After successful compilation, call run_simulation() to start the simulation.

    Args:
        project_name: Name of the project in user_projects/
        clean_first: Whether to run 'make clean' before compiling (default: False)

    Returns:
        str: Compilation status and results
    """
    # Check if project exists
    project_dir = USER_PROJECTS_DIR / project_name
    if not project_dir.exists():
        return f"Error: Project '{project_name}' not found at {project_dir}"

    try:
        # Change to PhysiCell root directory
        os.chdir(PHYSICELL_ROOT)

        result = f"**Compiling project:** {project_name}\n"
        result += f"**Working directory:** {PHYSICELL_ROOT}\n\n"

        # Pass through environment (includes PHYSICELL_CPP if set)
        compile_env = os.environ.copy()

        # Auto-detect compiler if PHYSICELL_CPP not set
        if "PHYSICELL_CPP" not in compile_env:
            # Try common g++ versions with OpenMP support on macOS
            for compiler_candidate in ["g++-15", "g++-14", "g++-13", "g++-12"]:
                try:
                    check = subprocess.run(
                        f"which {compiler_candidate}",
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if check.returncode == 0:
                        compile_env["PHYSICELL_CPP"] = compiler_candidate
                        break
                except:
                    pass

        compiler = compile_env.get("PHYSICELL_CPP", "g++")
        result += f"**Compiler:** {compiler}\n\n"

        # Load the project
        load_cmd = f"make load PROJ={project_name}"
        load_process = subprocess.run(
            load_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=compile_env
        )

        if load_process.returncode != 0:
            return f"Error loading project:\n{load_process.stderr}"

        result += f"**Project loaded successfully**\n\n"

        # Clean if requested
        if clean_first:
            clean_process = subprocess.run(
                "make clean",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=compile_env
            )
            result += f"**Cleaned build artifacts**\n\n"

        # Compile the project
        result += f"**Compiling...**\n"
        compile_process = subprocess.run(
            "make",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            env=compile_env
        )

        if compile_process.returncode == 0:
            result += f"**Compilation successful!**\n\n"

            # Find the executable name from Makefile
            try:
                makefile_path = PHYSICELL_ROOT / "Makefile"
                with open(makefile_path, 'r') as f:
                    for line in f:
                        if line.startswith("PROGRAM_NAME"):
                            exec_name = line.split("=")[1].strip()
                            executable = PHYSICELL_ROOT / exec_name
                            if executable.exists():
                                result += f"**Executable:** {executable}\n"
                            break
            except:
                result += f"**Executable:** ./project (default)\n"

            result += f"\n**Next step:** Use `run_simulation('{project_name}')` to run the simulation."
        else:
            result += f"**Compilation failed!**\n\n"
            result += f"**Error output:**\n```\n{compile_process.stderr[:1000]}\n```"

        return result

    except subprocess.TimeoutExpired:
        return "Error: Compilation timed out after 5 minutes"
    except Exception as e:
        return f"Error compiling project: {str(e)}"

@mcp.tool()
def run_simulation(project_name: str, config_file: Optional[str] = None) -> str:
    """
    Run a PhysiCell simulation as a background process.

    PREREQUISITE: Call compile_physicell_project() first to compile the project.
    NEXT STEPS:
    - Use get_simulation_status() to monitor progress
    - When completed, use generate_simulation_gif() to create visualization

    Args:
        project_name: Name of the project to run
        config_file: Custom config file path relative to project (default: config/PhysiCell_settings.xml)

    Returns:
        str: Simulation ID and status information
    """
    global running_simulations

    # Check if project exists
    project_dir = USER_PROJECTS_DIR / project_name
    if not project_dir.exists():
        return f"Error: Project '{project_name}' not found at {project_dir}"

    # Determine config file path
    if config_file:
        config_path = config_file
    else:
        config_path = "config/PhysiCell_settings.xml"

    # Generate simulation ID
    simulation_id = str(uuid.uuid4())[:8]

    try:
        # Change to PhysiCell root directory
        os.chdir(PHYSICELL_ROOT)

        # First load the project
        load_cmd = f"make load PROJ={project_name}"
        load_process = subprocess.run(
            load_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if load_process.returncode != 0:
            return f"Error loading project:\n{load_process.stderr}"

        # Find executable name
        exec_name = "project"
        try:
            makefile_path = PHYSICELL_ROOT / "Makefile"
            with open(makefile_path, 'r') as f:
                for line in f:
                    if line.startswith("PROGRAM_NAME"):
                        exec_name = line.split("=")[1].strip()
                        break
        except:
            pass

        executable = PHYSICELL_ROOT / exec_name
        if not executable.exists():
            return f"Error: Executable '{exec_name}' not found. Run compile_physicell_project() first."

        # Set up output folder
        output_folder = PHYSICELL_ROOT / "output"
        output_folder.mkdir(exist_ok=True)

        # Clear previous output
        for f in output_folder.glob("*"):
            try:
                f.unlink()
            except:
                pass

        # Start the simulation process
        # Redirect stdout/stderr to a log file instead of PIPE to prevent
        # the process from blocking when the pipe buffer fills up.
        # PhysiCell prints a lot during initialization (one line per cell),
        # and with many cells the 64KB pipe buffer fills instantly.
        log_file = output_folder / "simulation.log"
        log_handle = open(log_file, 'w')
        cmd = f"./{exec_name} {config_path}"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(PHYSICELL_ROOT)
        )

        # Close the Python file handle — the subprocess has inherited the fd
        log_handle.close()

        # Track the simulation
        sim_run = SimulationRun(
            simulation_id=simulation_id,
            project_name=project_name,
            process=process,
            pid=process.pid,
            status="running",
            output_folder=str(output_folder),
            config_file=config_path,
            started_at=time.time(),
            log_file=str(log_file)
        )

        with simulations_lock:
            running_simulations[simulation_id] = sim_run

        result = f"**Simulation started!**\n\n"
        result += f"**Simulation ID:** {simulation_id}\n"
        result += f"**Project:** {project_name}\n"
        result += f"**Config:** {config_path}\n"
        result += f"**PID:** {process.pid}\n"
        result += f"**Output folder:** {output_folder}\n\n"
        result += f"**Next steps:**\n"
        result += f"- Poll `get_simulation_status('{simulation_id}')` every 60 seconds until complete\n"
        result += f"- Use `list_simulations()` to see all running simulations\n"
        result += f"- Use `stop_simulation('{simulation_id}')` to stop if needed"

        return result

    except Exception as e:
        return f"Error starting simulation: {str(e)}"

_last_status_check: dict[str, float] = {}  # simulation_id -> last check timestamp

@mcp.tool()
def get_simulation_status(simulation_id: str) -> str:
    """
    Check the status of a running or completed simulation.

    IMPORTANT: Do NOT call this more than once per minute for a running simulation.
    Wait at least 60 seconds between calls.

    Args:
        simulation_id: The simulation ID returned by run_simulation()

    Returns:
        str: Current status, progress, and output file information
    """
    global running_simulations

    with simulations_lock:
        if simulation_id not in running_simulations:
            return f"Error: Simulation '{simulation_id}' not found. Use list_simulations() to see available simulations."

        sim = running_simulations[simulation_id]

    # Rate limit: reject if called less than 55 seconds since last check (for running sims)
    if sim.status == "running":
        now = time.time()
        last = _last_status_check.get(simulation_id, 0)
        if now - last < 55:
            wait = int(55 - (now - last))
            return (
                f"**Too soon.** Last status check was {int(now - last)}s ago. "
                f"Wait {wait} more seconds before checking again. "
                f"Simulations typically take 5-30 minutes."
            )
        _last_status_check[simulation_id] = now

    # Check if process is still running
    if sim.process:
        return_code = sim.process.poll()
        if return_code is None:
            sim.status = "running"
        elif return_code == 0:
            sim.status = "completed"
            sim.completed_at = time.time()
            sim.return_code = return_code
        else:
            sim.status = "failed"
            sim.completed_at = time.time()
            sim.return_code = return_code

    # Count output files
    output_folder = Path(sim.output_folder)
    svg_count = len(list(output_folder.glob("snapshot*.svg")))
    xml_count = len(list(output_folder.glob("output*.xml")))
    mat_count = len(list(output_folder.glob("*.mat")))

    # Calculate elapsed time
    if sim.completed_at:
        elapsed = sim.completed_at - sim.started_at
    else:
        elapsed = time.time() - sim.started_at

    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    result = f"**Simulation Status**\n\n"
    result += f"**ID:** {sim.simulation_id}\n"
    result += f"**Project:** {sim.project_name}\n"
    result += f"**Status:** {sim.status}\n"
    result += f"**Elapsed:** {elapsed_str}\n\n"
    result += f"**Output files:**\n"
    result += f"- SVG snapshots: {svg_count}\n"
    result += f"- XML data: {xml_count}\n"
    result += f"- MAT files: {mat_count}\n"

    if sim.status == "completed":
        result += f"\n**Simulation completed successfully!**\n"
        result += f"**Next step:** Use `generate_simulation_gif('{simulation_id}')` to create visualization."
    elif sim.status == "failed":
        result += f"\n**Simulation failed** (exit code: {sim.return_code})\n"
        # Read last lines from log file for error context
        if sim.log_file:
            try:
                log_path = Path(sim.log_file)
                if log_path.exists():
                    log_text = log_path.read_text()
                    # Show last 500 chars of log for error context
                    tail = log_text[-500:] if len(log_text) > 500 else log_text
                    if tail.strip():
                        result += f"**Log tail:**\n```\n{tail.strip()}\n```"
            except Exception:
                pass
    elif sim.status == "running":
        result += f"\n**Simulation is running.** Do NOT call `get_simulation_status()` again for at least 60 seconds. Wait, then check again."

    return result

@mcp.tool()
def list_simulations() -> str:
    """
    List all tracked simulations with their current status.

    Returns:
        str: Table of all simulations
    """
    global running_simulations

    with simulations_lock:
        if not running_simulations:
            return "No simulations tracked. Use `run_simulation()` to start one."

        result = "## Tracked Simulations\n\n"
        result += "| ID | Project | Status | Started | Elapsed |\n"
        result += "|---|---|---|---|---|\n"

        for sim_id, sim in running_simulations.items():
            # Update status if process exists
            if sim.process:
                return_code = sim.process.poll()
                if return_code is None:
                    sim.status = "running"
                elif return_code == 0:
                    sim.status = "completed"
                else:
                    sim.status = "failed"

            # Calculate elapsed time
            if sim.completed_at:
                elapsed = sim.completed_at - sim.started_at
            else:
                elapsed = time.time() - sim.started_at

            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            started_str = time.strftime("%H:%M:%S", time.localtime(sim.started_at))

            result += f"| {sim_id} | {sim.project_name} | {sim.status} | {started_str} | {elapsed_str} |\n"

        result += f"\nUse `get_simulation_status(id)` for details on a specific simulation."

        return result

@mcp.tool()
def stop_simulation(simulation_id: str) -> str:
    """
    Stop a running simulation.

    Args:
        simulation_id: The simulation ID to stop

    Returns:
        str: Confirmation of termination
    """
    global running_simulations

    with simulations_lock:
        if simulation_id not in running_simulations:
            return f"Error: Simulation '{simulation_id}' not found."

        sim = running_simulations[simulation_id]

    if sim.status != "running":
        return f"Simulation '{simulation_id}' is not running (status: {sim.status})"

    if sim.process:
        try:
            # Try graceful termination first
            sim.process.terminate()
            try:
                sim.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                sim.process.kill()
                sim.process.wait()

            sim.status = "stopped"
            sim.completed_at = time.time()

            return f"**Simulation stopped**\n\n**ID:** {simulation_id}\n**Project:** {sim.project_name}"

        except Exception as e:
            return f"Error stopping simulation: {str(e)}"

    return f"No process found for simulation '{simulation_id}'"

@mcp.tool()
def generate_simulation_gif(simulation_id: Optional[str] = None,
                           output_folder: Optional[str] = None,
                           frame_delay: int = 100,
                           max_frames: Optional[int] = None) -> str:
    """
    Create an animated GIF from simulation SVG snapshots.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly (e.g., /Users/simsz/PhysiCell/output)
        frame_delay: Delay between frames in milliseconds (default: 100)
        max_frames: Maximum number of frames to include (default: all)

    Returns:
        str: Path to generated GIF file
    """
    # Determine output folder
    if simulation_id:
        with simulations_lock:
            if simulation_id not in running_simulations:
                return f"Error: Simulation '{simulation_id}' not found."
            sim = running_simulations[simulation_id]
            folder = Path(sim.output_folder)
    elif output_folder:
        folder = Path(output_folder)
    else:
        # Default to PhysiCell output folder
        folder = PHYSICELL_ROOT / "output"

    if not folder.exists():
        return f"Error: Output folder not found: {folder}"

    # Find SVG files
    svg_files = sorted(folder.glob("snapshot*.svg"), key=lambda x: x.name)

    if not svg_files:
        return f"Error: No snapshot SVG files found in {folder}"

    # Limit frames if specified
    if max_frames and len(svg_files) > max_frames:
        # Sample evenly across the simulation
        step = len(svg_files) // max_frames
        svg_files = svg_files[::step][:max_frames]

    result = f"**Generating GIF from {len(svg_files)} frames...**\n\n"

    # Try using ImageMagick (magick convert)
    gif_path = folder / "simulation.gif"

    try:
        # Build the convert command
        svg_pattern = str(folder / "snapshot*.svg")
        cmd = f"magick convert -delay {frame_delay // 10} {svg_pattern} {gif_path}"

        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )

        if process.returncode == 0 and gif_path.exists():
            file_size = gif_path.stat().st_size / (1024 * 1024)  # MB
            result += f"**GIF created successfully!**\n\n"
            result += f"**File:** {gif_path}\n"
            result += f"**Size:** {file_size:.2f} MB\n"
            result += f"**Frames:** {len(svg_files)}\n"
            return result
        else:
            result += f"**ImageMagick failed.** Trying alternative method...\n"
            result += f"Error: {process.stderr[:200]}\n\n"

    except subprocess.TimeoutExpired:
        result += "ImageMagick timed out.\n\n"
    except FileNotFoundError:
        result += "ImageMagick not found. Trying alternative method...\n\n"
    except Exception as e:
        result += f"ImageMagick error: {str(e)}\n\n"

    # Try using Python libraries as fallback
    try:
        from PIL import Image
        import cairosvg
        import io

        images = []
        for i, svg_file in enumerate(svg_files):
            # Convert SVG to PNG in memory
            png_data = cairosvg.svg2png(url=str(svg_file))
            img = Image.open(io.BytesIO(png_data))
            images.append(img)

            if (i + 1) % 10 == 0:
                result += f"Processed {i + 1}/{len(svg_files)} frames...\n"

        # Save as GIF
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=frame_delay,
                loop=0
            )

            file_size = gif_path.stat().st_size / (1024 * 1024)
            result += f"\n**GIF created successfully!**\n\n"
            result += f"**File:** {gif_path}\n"
            result += f"**Size:** {file_size:.2f} MB\n"
            result += f"**Frames:** {len(images)}\n"
            return result

    except ImportError as e:
        result += f"**Error:** Required libraries not installed.\n"
        result += f"Install with: `pip install cairosvg pillow`\n"
        result += f"Or install ImageMagick: `brew install imagemagick`\n"
        return result
    except Exception as e:
        result += f"**Error creating GIF:** {str(e)}\n"
        return result

    return result

@mcp.tool()
def get_simulation_output_files(simulation_id: Optional[str] = None,
                                output_folder: Optional[str] = None,
                                file_type: str = "all") -> str:
    """
    List available output files from a simulation.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        file_type: Filter by type - "all", "svg", "mat", "xml" (default: "all")

    Returns:
        str: List of output files with details
    """
    # Determine output folder
    if simulation_id:
        with simulations_lock:
            if simulation_id not in running_simulations:
                return f"Error: Simulation '{simulation_id}' not found."
            sim = running_simulations[simulation_id]
            folder = Path(sim.output_folder)
    elif output_folder:
        folder = Path(output_folder)
    else:
        folder = PHYSICELL_ROOT / "output"

    if not folder.exists():
        return f"Error: Output folder not found: {folder}"

    # Collect files by type
    files_by_type = {
        "svg": list(folder.glob("*.svg")),
        "xml": list(folder.glob("*.xml")),
        "mat": list(folder.glob("*.mat")),
        "gif": list(folder.glob("*.gif")),
        "other": []
    }

    # Collect other files
    all_typed = set()
    for typed_files in files_by_type.values():
        all_typed.update(f.name for f in typed_files)

    for f in folder.iterdir():
        if f.is_file() and f.name not in all_typed:
            files_by_type["other"].append(f)

    result = f"## Output Files\n\n"
    result += f"**Folder:** {folder}\n\n"

    # Filter and display
    types_to_show = [file_type] if file_type != "all" else ["svg", "xml", "mat", "gif", "other"]

    total_files = 0
    total_size = 0

    for ftype in types_to_show:
        if ftype in files_by_type and files_by_type[ftype]:
            files = sorted(files_by_type[ftype], key=lambda x: x.name)
            result += f"### {ftype.upper()} Files ({len(files)})\n"

            # Show first few and last few if many files
            if len(files) > 10:
                show_files = files[:5] + files[-3:]
                result += f"*(showing 8 of {len(files)} files)*\n"
            else:
                show_files = files

            for f in show_files:
                size_kb = f.stat().st_size / 1024
                total_size += f.stat().st_size
                total_files += 1
                if size_kb > 1024:
                    size_str = f"{size_kb/1024:.1f} MB"
                else:
                    size_str = f"{size_kb:.1f} KB"
                result += f"- {f.name} ({size_str})\n"

                if len(files) > 10 and f == files[4]:
                    result += f"- ... ({len(files) - 8} more files) ...\n"

            result += "\n"

    result += f"**Total:** {total_files} files, {total_size / (1024*1024):.2f} MB\n"

    return result

# ============================================================================
# SIMULATION DATA ANALYSIS (pcdl)
# ============================================================================

@mcp.tool()
def get_simulation_analysis_overview(simulation_id: Optional[str] = None,
                                     output_folder: Optional[str] = None) -> str:
    """
    Get an executive summary of a completed simulation.
    One-shot overview of cell populations, spatial extent, and substrate state.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly (e.g., /Users/simsz/PhysiCell/output)

    Returns:
        str: Markdown-formatted simulation overview with suggested next tools
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    # Check for output XML files
    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    try:
        # Load first, middle, and last timesteps for population snapshots
        sample_indices = [0, len(xml_files) // 2, len(xml_files) - 1]
        sample_indices = sorted(set(sample_indices))  # deduplicate if few files

        result = "## Simulation Analysis Overview\n\n"
        result += f"**Output folder:** {folder}\n"
        result += f"**Total timesteps:** {len(xml_files)}\n\n"

        snapshots = []
        for idx in sample_indices:
            xml_name = xml_files[idx].name
            ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                               microenv=False, graph=False, verbose=False)
            df = ts.get_cell_df()
            time_val = ts.get_time()

            # Count cells by type and status
            type_counts = df['cell_type'].value_counts().to_dict()
            total = len(df)

            # Separate live vs dead
            live_phases = {'Ki67_negative', 'Ki67_positive', 'S', 'G0G1', 'G0', 'G1',
                          'G1a', 'G1b', 'G1c', 'G2', 'G2M', 'M', 'live'}
            dead_phases = {'apoptotic', 'necrotic_swelling', 'necrotic_lysed', 'debris'}
            n_live = len(df[df['current_phase'].isin(live_phases)])
            n_dead = total - n_live

            # Spatial extent
            x_range = (df['position_x'].min(), df['position_x'].max())
            y_range = (df['position_y'].min(), df['position_y'].max())

            snapshots.append({
                'time': time_val, 'type_counts': type_counts,
                'total': total, 'live': n_live, 'dead': n_dead,
                'x_range': x_range, 'y_range': y_range
            })

        # Duration
        result += f"**Simulation time:** {snapshots[0]['time']:.1f} to {snapshots[-1]['time']:.1f} min"
        duration_hours = (snapshots[-1]['time'] - snapshots[0]['time']) / 60
        if duration_hours > 0:
            result += f" ({duration_hours:.1f} hours)"
        result += "\n\n"

        # Population summary table
        result += "### Cell Population Over Time\n\n"
        result += "| Time (min) | Total | Live | Dead |"
        all_types = set()
        for s in snapshots:
            all_types.update(s['type_counts'].keys())
        all_types = sorted(all_types)
        for t in all_types:
            result += f" {t} |"
        result += "\n"
        result += "|---|---|---|---|"
        for _ in all_types:
            result += "---|"
        result += "\n"

        for s in snapshots:
            result += f"| {s['time']:.0f} | {s['total']} | {s['live']} | {s['dead']} |"
            for t in all_types:
                result += f" {s['type_counts'].get(t, 0)} |"
            result += "\n"

        result += "\n"

        # Growth/decline analysis
        if len(snapshots) >= 2:
            first, last = snapshots[0], snapshots[-1]
            result += "### Population Changes\n\n"
            for t in all_types:
                n0 = first['type_counts'].get(t, 0)
                nf = last['type_counts'].get(t, 0)
                if n0 > 0:
                    change_pct = ((nf - n0) / n0) * 100
                    arrow = "+" if change_pct >= 0 else ""
                    result += f"- **{t}**: {n0} → {nf} ({arrow}{change_pct:.1f}%)\n"
                elif nf > 0:
                    result += f"- **{t}**: 0 → {nf} (new)\n"
            result += "\n"

            # Spatial expansion
            result += "### Spatial Extent\n\n"
            result += f"- **Initial:** x=[{first['x_range'][0]:.0f}, {first['x_range'][1]:.0f}] "
            result += f"y=[{first['y_range'][0]:.0f}, {first['y_range'][1]:.0f}]\n"
            result += f"- **Final:** x=[{last['x_range'][0]:.0f}, {last['x_range'][1]:.0f}] "
            result += f"y=[{last['y_range'][0]:.0f}, {last['y_range'][1]:.0f}]\n\n"

        # Substrate state from final timestep
        try:
            final_xml = xml_files[-1].name
            ts_final = pcdl_TimeStep(final_xml, output_path=str(folder),
                                     microenv=True, graph=False, verbose=False)
            substrates = ts_final.get_substrate_list()
            if substrates:
                conc_df = ts_final.get_conc_df()
                result += "### Final Substrate State\n\n"
                result += "| Substrate | Mean | Std | Min | Max |\n"
                result += "|---|---|---|---|---|\n"
                for sub in substrates:
                    if sub in conc_df.columns:
                        vals = conc_df[sub]
                        result += f"| {sub} | {vals.mean():.4g} | {vals.std():.4g} | {vals.min():.4g} | {vals.max():.4g} |\n"
                result += "\n"
        except Exception:
            pass  # Substrate analysis is optional

        # Suggest next tools
        result += "### Suggested Next Steps\n\n"
        result += "- `get_population_timeseries()` — detailed growth curves across all timesteps\n"
        result += "- `get_timestep_summary(timestep=N)` — drill into a specific timepoint\n"
        result += "- `get_substrate_summary()` — substrate concentration details\n"
        result += "- `get_cell_data()` — detailed cell attributes with filtering\n"
        result += "- `generate_analysis_plot()` — save visualizations to disk\n"
        result += "- `generate_simulation_gif()` — spatial animation from SVG snapshots\n"

        return result

    except Exception as e:
        return f"Error analyzing simulation: {str(e)}"


@mcp.tool()
def get_timestep_summary(simulation_id: Optional[str] = None,
                         output_folder: Optional[str] = None,
                         timestep: int = -1) -> str:
    """
    Get a quantitative snapshot of cell populations and spatial extent at one timestep.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        timestep: Timestep index (0-based). Use -1 for latest (default: -1)

    Returns:
        str: Markdown-formatted timestep summary
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    # Resolve timestep index
    if timestep < 0:
        timestep = len(xml_files) + timestep
    if timestep < 0 or timestep >= len(xml_files):
        return f"Error: Timestep {timestep} out of range (0-{len(xml_files)-1})"

    try:
        xml_name = xml_files[timestep].name
        ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                           microenv=False, graph=False, verbose=False)
        df = ts.get_cell_df()
        time_val = ts.get_time()

        result = f"## Timestep Summary\n\n"
        result += f"**File:** {xml_name}\n"
        result += f"**Time:** {time_val:.1f} min ({time_val/60:.1f} hours)\n"
        result += f"**Timestep index:** {timestep} of {len(xml_files)-1}\n"
        result += f"**Total cells:** {len(df)}\n\n"

        # Cell counts by type and phase
        result += "### Cell Counts by Type and Phase\n\n"
        result += "| Cell Type | Total | Live | Apoptotic | Necrotic |\n"
        result += "|---|---|---|---|---|\n"

        for ct in sorted(df['cell_type'].unique()):
            ct_df = df[df['cell_type'] == ct]
            total = len(ct_df)
            phases = ct_df['current_phase'].value_counts().to_dict()

            # Categorize phases
            apoptotic = phases.get('apoptotic', 0)
            necrotic = sum(v for k, v in phases.items()
                         if 'necrotic' in str(k).lower() or 'debris' in str(k).lower())
            live = total - apoptotic - necrotic

            result += f"| {ct} | {total} | {live} | {apoptotic} | {necrotic} |\n"

        result += "\n"

        # Phase breakdown
        result += "### Phase Distribution\n\n"
        phase_counts = df['current_phase'].value_counts()
        for phase, count in phase_counts.items():
            pct = (count / len(df)) * 100
            result += f"- **{phase}**: {count} ({pct:.1f}%)\n"
        result += "\n"

        # Spatial extent
        result += "### Spatial Extent\n\n"
        result += f"- **X range:** [{df['position_x'].min():.1f}, {df['position_x'].max():.1f}] μm\n"
        result += f"- **Y range:** [{df['position_y'].min():.1f}, {df['position_y'].max():.1f}] μm\n"
        result += f"- **Z range:** [{df['position_z'].min():.1f}, {df['position_z'].max():.1f}] μm\n\n"

        # Volume stats
        if 'total_volume' in df.columns:
            result += "### Volume Statistics\n\n"
            result += f"- **Mean volume:** {df['total_volume'].mean():.1f} μm³\n"
            result += f"- **Std volume:** {df['total_volume'].std():.1f} μm³\n"
            result += f"- **Range:** [{df['total_volume'].min():.1f}, {df['total_volume'].max():.1f}] μm³\n\n"

        # List available attributes for deeper analysis
        result += "### Available Attributes\n\n"
        result += f"Total columns: {len(df.columns)}\n\n"
        # Show a curated selection of useful columns
        useful_cols = [c for c in df.columns if any(k in c.lower() for k in
                       ['position', 'volume', 'speed', 'pressure', 'damage',
                        'cycle', 'death', 'oxygen', 'phase', 'type'])]
        if useful_cols:
            result += "Key columns: " + ", ".join(sorted(useful_cols)[:20]) + "\n\n"

        result += "Use `get_cell_data(columns='col1,col2')` to inspect specific attributes.\n"

        return result

    except Exception as e:
        return f"Error reading timestep: {str(e)}"


@mcp.tool()
def get_population_timeseries(simulation_id: Optional[str] = None,
                              output_folder: Optional[str] = None,
                              max_timesteps: int = 200) -> str:
    """
    Get cell population counts by type across all timesteps (growth curve data).
    Subsamples if more timesteps than max_timesteps.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        max_timesteps: Maximum number of timesteps to include (default: 200)

    Returns:
        str: Markdown-formatted population timeseries with growth rates
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    try:
        # Subsample if needed
        if len(xml_files) > max_timesteps:
            step = len(xml_files) / max_timesteps
            indices = [int(i * step) for i in range(max_timesteps)]
            # Always include last
            if indices[-1] != len(xml_files) - 1:
                indices[-1] = len(xml_files) - 1
            selected_files = [xml_files[i] for i in indices]
        else:
            selected_files = xml_files

        # Collect population data
        rows = []
        all_types = set()

        for xml_file in selected_files:
            ts = pcdl_TimeStep(xml_file.name, output_path=str(folder),
                               microenv=False, graph=False, verbose=False)
            df = ts.get_cell_df()
            time_val = ts.get_time()

            type_counts = df['cell_type'].value_counts().to_dict()
            all_types.update(type_counts.keys())

            # Count live vs dead
            live_phases = {'Ki67_negative', 'Ki67_positive', 'S', 'G0G1', 'G0', 'G1',
                          'G1a', 'G1b', 'G1c', 'G2', 'G2M', 'M', 'live'}
            n_live = len(df[df['current_phase'].isin(live_phases)])
            n_dead = len(df) - n_live

            rows.append({
                'time': time_val, 'total': len(df),
                'live': n_live, 'dead': n_dead,
                **type_counts
            })

        all_types = sorted(all_types)

        result = "## Population Timeseries\n\n"
        result += f"**Timesteps:** {len(rows)} (of {len(xml_files)} total)\n"
        result += f"**Duration:** {rows[0]['time']:.0f} to {rows[-1]['time']:.0f} min "
        result += f"({(rows[-1]['time'] - rows[0]['time'])/60:.1f} hours)\n"
        result += f"**Cell types:** {', '.join(all_types)}\n\n"

        # Data table
        result += "### Population Table\n\n"
        result += "| Time (min) | Total | Live | Dead |"
        for t in all_types:
            result += f" {t} |"
        result += "\n"
        result += "|---|---|---|---|"
        for _ in all_types:
            result += "---|"
        result += "\n"

        for row in rows:
            result += f"| {row['time']:.0f} | {row['total']} | {row['live']} | {row['dead']} |"
            for t in all_types:
                result += f" {row.get(t, 0)} |"
            result += "\n"

        result += "\n"

        # Growth rate analysis
        if len(rows) >= 2:
            result += "### Growth Analysis\n\n"
            first, last = rows[0], rows[-1]
            dt_hours = (last['time'] - first['time']) / 60

            for t in all_types:
                n0 = first.get(t, 0)
                nf = last.get(t, 0)
                if n0 > 0 and nf > 0 and dt_hours > 0:
                    # Net growth rate
                    import math as _math
                    growth_rate = _math.log(nf / n0) / dt_hours if nf != n0 else 0
                    if growth_rate > 0:
                        doubling_time = _math.log(2) / growth_rate
                        result += f"- **{t}**: {n0} → {nf}, "
                        result += f"growth rate = {growth_rate:.4f}/hr, "
                        result += f"doubling time ≈ {doubling_time:.1f} hr\n"
                    elif growth_rate < 0:
                        half_life = -_math.log(2) / growth_rate
                        result += f"- **{t}**: {n0} → {nf}, "
                        result += f"decline rate = {growth_rate:.4f}/hr, "
                        result += f"half-life ≈ {half_life:.1f} hr\n"
                    else:
                        result += f"- **{t}**: {n0} → {nf} (stable)\n"
                elif n0 == 0 and nf > 0:
                    result += f"- **{t}**: 0 → {nf} (appeared during simulation)\n"
                elif n0 > 0 and nf == 0:
                    result += f"- **{t}**: {n0} → 0 (eliminated)\n"

        return result

    except Exception as e:
        return f"Error generating population timeseries: {str(e)}"


@mcp.tool()
def get_substrate_summary(simulation_id: Optional[str] = None,
                          output_folder: Optional[str] = None,
                          timestep: int = -1) -> str:
    """
    Get substrate concentration statistics at one timestep.
    Shows mean/std/min/max per substrate and center-vs-edge gradient.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        timestep: Timestep index (0-based). Use -1 for latest (default: -1)

    Returns:
        str: Markdown-formatted substrate statistics
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    # Resolve timestep index
    if timestep < 0:
        timestep = len(xml_files) + timestep
    if timestep < 0 or timestep >= len(xml_files):
        return f"Error: Timestep {timestep} out of range (0-{len(xml_files)-1})"

    try:
        xml_name = xml_files[timestep].name
        ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                           microenv=True, graph=False, verbose=False)
        time_val = ts.get_time()
        substrates = ts.get_substrate_list()

        if not substrates:
            return "No substrates found in the simulation output."

        conc_df = ts.get_conc_df()

        result = f"## Substrate Summary\n\n"
        result += f"**File:** {xml_name}\n"
        result += f"**Time:** {time_val:.1f} min ({time_val/60:.1f} hours)\n"
        result += f"**Substrates:** {', '.join(substrates)}\n"
        result += f"**Voxels:** {len(conc_df)}\n\n"

        # Statistics table
        result += "### Concentration Statistics\n\n"
        result += "| Substrate | Mean | Std | Min | Max | Nonzero % |\n"
        result += "|---|---|---|---|---|---|\n"

        for sub in substrates:
            if sub in conc_df.columns:
                vals = conc_df[sub]
                nonzero_pct = (vals > 1e-10).sum() / len(vals) * 100
                result += f"| {sub} | {vals.mean():.4g} | {vals.std():.4g} "
                result += f"| {vals.min():.4g} | {vals.max():.4g} | {nonzero_pct:.1f}% |\n"

        result += "\n"

        # Center vs edge gradient analysis
        if 'mesh_center_m' in conc_df.columns and 'mesh_center_n' in conc_df.columns:
            result += "### Center vs Edge Gradient\n\n"

            m_vals = conc_df['mesh_center_m']
            n_vals = conc_df['mesh_center_n']
            m_center = (m_vals.max() + m_vals.min()) / 2
            n_center = (n_vals.max() + n_vals.min()) / 2
            m_range = m_vals.max() - m_vals.min()

            # Define center (inner 25% of domain) and edge (outer 25%)
            dist = ((m_vals - m_center)**2 + (n_vals - n_center)**2)**0.5
            center_thresh = m_range * 0.25
            edge_thresh = m_range * 0.40

            center_mask = dist <= center_thresh
            edge_mask = dist >= edge_thresh

            if center_mask.sum() > 0 and edge_mask.sum() > 0:
                result += "| Substrate | Center Mean | Edge Mean | Gradient |\n"
                result += "|---|---|---|---|\n"

                for sub in substrates:
                    if sub in conc_df.columns:
                        center_mean = conc_df.loc[center_mask, sub].mean()
                        edge_mean = conc_df.loc[edge_mask, sub].mean()
                        if edge_mean > 1e-10:
                            gradient = (edge_mean - center_mean) / edge_mean * 100
                            result += f"| {sub} | {center_mean:.4g} | {edge_mean:.4g} | {gradient:+.1f}% |\n"
                        else:
                            result += f"| {sub} | {center_mean:.4g} | {edge_mean:.4g} | N/A |\n"

                result += "\n"

        return result

    except Exception as e:
        return f"Error reading substrate data: {str(e)}"


@mcp.tool()
def get_cell_data(simulation_id: Optional[str] = None,
                  output_folder: Optional[str] = None,
                  timestep: int = -1,
                  cell_type: Optional[str] = None,
                  columns: Optional[str] = None,
                  sort_by: Optional[str] = None,
                  max_rows: int = 50) -> str:
    """
    Get detailed cell data with filtering and column selection.
    Returns a markdown table plus column statistics for the full population.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        timestep: Timestep index (0-based). Use -1 for latest (default: -1)
        cell_type: Filter to a specific cell type (default: all)
        columns: Comma-separated list of columns to include (default: essential columns)
        sort_by: Column to sort by (default: no sorting)
        max_rows: Maximum rows in the table (default: 50)

    Returns:
        str: Markdown-formatted cell data table and statistics
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    # Resolve timestep index
    if timestep < 0:
        timestep = len(xml_files) + timestep
    if timestep < 0 or timestep >= len(xml_files):
        return f"Error: Timestep {timestep} out of range (0-{len(xml_files)-1})"

    try:
        xml_name = xml_files[timestep].name
        ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                           microenv=False, graph=False, verbose=False)
        df = ts.get_cell_df()
        time_val = ts.get_time()

        # Filter by cell type
        if cell_type:
            available_types = df['cell_type'].unique().tolist()
            if cell_type not in available_types:
                return f"Error: Cell type '{cell_type}' not found. Available: {', '.join(available_types)}"
            df = df[df['cell_type'] == cell_type]

        result = f"## Cell Data\n\n"
        result += f"**Time:** {time_val:.1f} min | "
        result += f"**Cells:** {len(df)}"
        if cell_type:
            result += f" (type: {cell_type})"
        result += "\n\n"

        # Determine columns to show
        essential = ['cell_type', 'position_x', 'position_y']
        if columns:
            requested = [c.strip() for c in columns.split(',')]
            # Always include essentials at front, then requested
            show_cols = []
            for c in essential:
                if c in df.columns:
                    show_cols.append(c)
            for c in requested:
                if c in df.columns and c not in show_cols:
                    show_cols.append(c)
                elif c not in df.columns:
                    result += f"*Warning: column '{c}' not found*\n"
        else:
            # Default: essential + key attributes
            default_extra = ['position_z', 'total_volume', 'current_phase',
                           'pressure', 'total_attack_time']
            show_cols = [c for c in essential if c in df.columns]
            show_cols += [c for c in default_extra if c in df.columns]

        # Cap at 10 columns to prevent overflow
        if len(show_cols) > 10:
            show_cols = show_cols[:10]
            result += f"*Showing 10 of {len(show_cols)} requested columns*\n\n"

        # Sort if requested
        if sort_by:
            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=False)
            else:
                result += f"*Warning: sort column '{sort_by}' not found*\n"

        # Build table (capped at max_rows)
        display_df = df.head(max_rows)

        result += "### Data Table\n\n"
        if len(df) > max_rows:
            result += f"*Showing {max_rows} of {len(df)} cells*\n\n"

        # Header
        result += "| " + " | ".join(show_cols) + " |\n"
        result += "|" + "|".join(["---"] * len(show_cols)) + "|\n"

        # Rows
        for _, row in display_df.iterrows():
            vals = []
            for c in show_cols:
                v = row.get(c, 'N/A')
                if isinstance(v, float):
                    vals.append(f"{v:.4g}")
                else:
                    vals.append(str(v))
            result += "| " + " | ".join(vals) + " |\n"

        result += "\n"

        # Column statistics for numeric columns (full population, not just displayed rows)
        numeric_cols = [c for c in show_cols if c in df.columns and
                       df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        if numeric_cols:
            result += "### Column Statistics (full population)\n\n"
            result += "| Column | Mean | Std | Min | Max |\n"
            result += "|---|---|---|---|---|\n"
            for c in numeric_cols:
                vals = df[c]
                result += f"| {c} | {vals.mean():.4g} | {vals.std():.4g} "
                result += f"| {vals.min():.4g} | {vals.max():.4g} |\n"
            result += "\n"

        return result

    except Exception as e:
        return f"Error reading cell data: {str(e)}"


@mcp.tool()
def generate_analysis_plot(simulation_id: Optional[str] = None,
                           output_folder: Optional[str] = None,
                           plot_type: str = "population_timeseries",
                           timestep: int = -1) -> str:
    """
    Generate and save an analysis plot to disk.

    Args:
        simulation_id: Use output folder from this simulation ID
        output_folder: Or specify output folder directly
        plot_type: Plot type - "population_timeseries", "cell_scatter", or "substrate_contour"
        timestep: Timestep index for scatter/contour plots (default: -1 for latest)

    Returns:
        str: Path to saved plot file
    """
    _ensure_pcdl_imports()
    if not PCDL_AVAILABLE:
        return "Error: pcdl package not installed. Install with: `pip install pcdl`"

    folder, err = _resolve_output_folder(simulation_id, output_folder)
    if err:
        return err

    xml_files = sorted(folder.glob("output*.xml"))
    if not xml_files:
        return f"Error: No PhysiCell output XML files found in {folder}"

    # Create output directory for plots
    plots_dir = MCP_OUTPUT_DIR / "analysis_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        if plot_type == "population_timeseries":
            # Load all timesteps and count cells by type
            # Subsample if many timesteps
            if len(xml_files) > 200:
                step = len(xml_files) / 200
                indices = [int(i * step) for i in range(200)]
                if indices[-1] != len(xml_files) - 1:
                    indices[-1] = len(xml_files) - 1
                selected = [xml_files[i] for i in indices]
            else:
                selected = xml_files

            times = []
            type_data = {}

            for xml_file in selected:
                ts = pcdl_TimeStep(xml_file.name, output_path=str(folder),
                                   microenv=False, graph=False, verbose=False)
                df = ts.get_cell_df()
                time_val = ts.get_time()
                times.append(time_val / 60)  # Convert to hours

                counts = df['cell_type'].value_counts().to_dict()
                for ct, n in counts.items():
                    if ct not in type_data:
                        type_data[ct] = []
                # Ensure all types have entries
                for ct in type_data:
                    type_data[ct].append(counts.get(ct, 0))

            fig, ax = plt.subplots(figsize=(10, 6))
            for ct, counts in sorted(type_data.items()):
                ax.plot(times, counts, label=ct, linewidth=2)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Cell Count')
            ax.set_title('Cell Population Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = plots_dir / "population_timeseries.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return (f"## Population Timeseries Plot\n\n"
                    f"**Saved to:** {plot_path}\n"
                    f"**Timesteps plotted:** {len(times)}\n"
                    f"**Cell types:** {', '.join(sorted(type_data.keys()))}\n")

        elif plot_type == "cell_scatter":
            # Resolve timestep
            if timestep < 0:
                timestep = len(xml_files) + timestep
            if timestep < 0 or timestep >= len(xml_files):
                return f"Error: Timestep {timestep} out of range (0-{len(xml_files)-1})"

            xml_name = xml_files[timestep].name
            ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                               microenv=False, graph=False, verbose=False)
            time_val = ts.get_time()

            fig = ts.plot_scatter(focus='cell_type', ext=None)

            plot_path = plots_dir / f"cell_scatter_t{timestep}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return (f"## Cell Scatter Plot\n\n"
                    f"**Saved to:** {plot_path}\n"
                    f"**Timestep:** {timestep} (time={time_val:.0f} min)\n")

        elif plot_type == "substrate_contour":
            # Resolve timestep
            if timestep < 0:
                timestep = len(xml_files) + timestep
            if timestep < 0 or timestep >= len(xml_files):
                return f"Error: Timestep {timestep} out of range (0-{len(xml_files)-1})"

            xml_name = xml_files[timestep].name
            ts = pcdl_TimeStep(xml_name, output_path=str(folder),
                               microenv=True, graph=False, verbose=False)
            time_val = ts.get_time()
            substrates = ts.get_substrate_list()

            if not substrates:
                return "Error: No substrates found in the simulation output."

            # Plot first substrate (most commonly oxygen)
            focus_sub = substrates[0]
            fig = ts.plot_contour(focus=focus_sub, ext=None)

            plot_path = plots_dir / f"substrate_{focus_sub}_t{timestep}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return (f"## Substrate Contour Plot\n\n"
                    f"**Saved to:** {plot_path}\n"
                    f"**Substrate:** {focus_sub}\n"
                    f"**Timestep:** {timestep} (time={time_val:.0f} min)\n"
                    f"**All substrates:** {', '.join(substrates)}\n")

        else:
            return (f"Error: Unknown plot_type '{plot_type}'. "
                    f"Choose from: population_timeseries, cell_scatter, substrate_contour")

    except ImportError:
        return "Error: matplotlib is required for plotting. Install with: `pip install matplotlib`"
    except Exception as e:
        return f"Error generating plot: {str(e)}"


# ============================================================================
# UNCERTAINTY QUANTIFICATION (UQ) TOOLS
# ============================================================================

# Global UQ run tracking
uq_runs: Dict[str, Dict] = {}
uq_runs_lock = Lock()

def _generate_uq_ini(session: SessionState, uq_ctx: UQContext) -> str:
    """Generate a UQ_PhysiCell INI config file from session state."""
    config = configparser.ConfigParser()
    section = "uq_model"
    config.add_section(section)

    config.set(section, "executable", uq_ctx.executable_path or "./project")
    config.set(section, "configfile_ref", uq_ctx.xml_config_path or "config/PhysiCell_settings.xml")
    config.set(section, "numreplicates", str(uq_ctx.num_replicates))
    config.set(section, "omp_num_threads", str(uq_ctx.num_workers))

    # Build parameters dict: fixed overrides + variable params
    params_dict = {
        ".//save/SVG/enable": "'false'",
    }

    # Build rules params dict
    rules_dict = {}

    for p in uq_ctx.parameters:
        if p.param_type == "xml" and p.xpath:
            params_dict[p.xpath] = f"[None, '{p.name}']"
        elif p.param_type == "rules" and p.rule_key:
            rules_dict[p.rule_key] = f"[None, '{p.name}']"

    config.set(section, "parameters", str(params_dict).replace('"[', '[').replace(']"', ']').replace("'[", '[').replace("]'", ']'))

    if uq_ctx.rules_csv_path:
        config.set(section, "rulesfile_ref", uq_ctx.rules_csv_path)
        if rules_dict:
            config.set(section, "parameters_rules", str(rules_dict).replace('"[', '[').replace(']"', ']').replace("'[", '[').replace("]'", ']'))

    # Write INI to the UQ output directory
    ini_path = Path(uq_ctx.uq_output_dir) / "uq_config.ini"

    # Write manually to avoid configparser quoting issues
    with open(ini_path, 'w') as f:
        f.write(f"[{section}]\n")
        f.write(f"executable = {uq_ctx.executable_path or './project'}\n")
        f.write(f"configfile_ref = {uq_ctx.xml_config_path or 'config/PhysiCell_settings.xml'}\n")
        f.write(f"numreplicates = {uq_ctx.num_replicates}\n")
        f.write(f"omp_num_threads = {uq_ctx.num_workers}\n")

        # Build proper parameters dict string
        xml_params = {}
        for p in uq_ctx.parameters:
            if p.param_type == "xml" and p.xpath:
                xml_params[p.xpath] = [None, p.name]
        # Add fixed overrides
        xml_params[".//save/SVG/enable"] = "false"

        f.write(f"parameters = {repr(xml_params)}\n")

        if uq_ctx.rules_csv_path:
            f.write(f"rulesfile_ref = {uq_ctx.rules_csv_path}\n")
            rules_params = {}
            for p in uq_ctx.parameters:
                if p.param_type == "rules" and p.rule_key:
                    rules_params[p.rule_key] = [None, p.name]
            if rules_params:
                f.write(f"parameters_rules = {repr(rules_params)}\n")

    return str(ini_path)


def _build_xpath_for_cell_param(cell_type: str, param_path: str) -> str:
    """Build XPath expression for a cell type parameter."""
    # Common parameter path mappings
    mappings = {
        "cycle entry rate": "phenotype/cycle/phase_transition_rates/rate[1]",
        "cycle entry": "phenotype/cycle/phase_transition_rates/rate[1]",
        "apoptosis rate": "phenotype/death/*[@name='apoptosis']/death_rate",
        "necrosis rate": "phenotype/death/*[@name='necrosis']/death_rate",
        "migration speed": "phenotype/motility/speed",
        "migration bias": "phenotype/motility/migration_bias",
        "persistence time": "phenotype/motility/persistence_time",
    }
    resolved = mappings.get(param_path, param_path)
    return f".//cell_definitions/cell_definition[@name='{cell_type}']/{resolved}"


def _build_rule_key(cell_type: str, signal: str, direction: str, behavior: str, field: str) -> str:
    """Build a rule parameter key string for UQ_PhysiCell."""
    return f"{cell_type},{signal},{direction},{behavior},{field}"


@mcp.tool()
def setup_uq_analysis(project_name: Optional[str] = None,
                      num_replicates: int = 3,
                      num_workers: int = 4) -> str:
    """
    Initialize uncertainty quantification analysis for the current PhysiCell model.
    This auto-detects the compiled project, XML config, and rules CSV from the session,
    and prepares the UQ working directory.

    PREREQUISITES: You must have already:
    1. Created and configured a simulation (domain, substrates, cell types, rules)
    2. Exported XML and rules CSV
    3. Created and compiled a PhysiCell project

    Args:
        project_name: Name of the compiled PhysiCell project (auto-detected if omitted)
        num_replicates: Number of simulation replicates per parameter set (default: 3)
        num_workers: Number of parallel workers for simulations (default: 4)

    Returns:
        str: Setup status and next steps
    """
    _ensure_uq_imports()
    if not UQ_AVAILABLE:
        return "**Error:** uq-physicell package not installed. Run: `pip install uq-physicell`"

    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No active session. Create and configure a simulation first."

    # Auto-detect project
    if not project_name:
        # Check for recent simulations
        with simulations_lock:
            if running_simulations:
                latest = max(running_simulations.values(), key=lambda s: s.started_at)
                project_name = latest.project_name

    if not project_name:
        return "**Error:** No project found. Specify project_name or create/compile a project first."

    project_dir = USER_PROJECTS_DIR / project_name
    if not project_dir.exists():
        return f"**Error:** Project directory not found: {project_dir}"

    # Find executable
    executable = project_dir / "project"
    if not executable.exists():
        return f"**Error:** Executable not found at {executable}. Compile the project first with `compile_physicell_project('{project_name}')`."

    # Find config files
    config_dir = project_dir / "config"
    xml_path = config_dir / "PhysiCell_settings.xml"
    rules_path = config_dir / "cell_rules.csv"

    if not xml_path.exists():
        return f"**Error:** XML config not found at {xml_path}. Export configuration first."

    # Create UQ working directory
    uq_output_dir = project_dir / "uq_analysis"
    uq_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize UQ context
    uq_ctx = UQContext(
        project_name=project_name,
        executable_path=str(executable),
        xml_config_path=str(xml_path),
        rules_csv_path=str(rules_path) if rules_path.exists() else None,
        num_replicates=num_replicates,
        num_workers=num_workers,
        uq_output_dir=str(uq_output_dir),
    )
    session.uq_context = uq_ctx
    session.mark_step_complete(WorkflowStep.UQ_SETUP)

    result = "## UQ Analysis Initialized\n\n"
    result += f"**Project:** {project_name}\n"
    result += f"**Executable:** {executable}\n"
    result += f"**XML Config:** {xml_path}\n"
    result += f"**Rules CSV:** {rules_path if rules_path.exists() else 'None'}\n"
    result += f"**Replicates:** {num_replicates}\n"
    result += f"**Workers:** {num_workers}\n"
    result += f"**UQ Directory:** {uq_output_dir}\n\n"
    result += "**Next step:** Use `get_uq_parameter_suggestions()` to see calibratable parameters, "
    result += "then `define_uq_parameters()` to select which ones to vary."
    return result


@mcp.tool()
def get_uq_parameter_suggestions() -> str:
    """
    Analyze the current model configuration and suggest parameters that can be
    calibrated or analyzed with uncertainty quantification. Returns both XML-based
    parameters (cell properties) and rules-based parameters (Hill function coefficients).

    Returns:
        str: Markdown-formatted list of suggested parameters with reference values
    """
    session = get_current_session()
    if not session or not session.config:
        return "**Error:** No active session with configuration."

    if not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    import xml.etree.ElementTree as ET

    result = "## Suggested UQ Parameters\n\n"

    # === XML-based parameters ===
    result += "### XML Parameters (Cell Properties)\n\n"
    result += "| # | Cell Type | Parameter | XPath | Ref Value |\n"
    result += "|---|-----------|-----------|-------|-----------|\n"

    xml_suggestions = []
    config = session.config
    cell_types = list(config.cell_types.get_cell_types().keys())

    # Parse XML to get actual values
    xml_path = session.uq_context.xml_config_path
    try:
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
    except Exception:
        xml_root = None

    param_idx = 1
    for ct in cell_types:
        param_paths = [
            ("cycle entry rate", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/cycle/phase_transition_rates/rate[1]"),
            ("apoptosis rate", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/death/model[@name='apoptosis']/death_rate"),
            ("necrosis rate", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/death/model[@name='necrosis']/death_rate"),
            ("migration speed", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/motility/speed"),
            ("migration bias", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/motility/migration_bias"),
            ("persistence time", f".//cell_definitions/cell_definition[@name='{ct}']/phenotype/motility/persistence_time"),
        ]
        for param_name, xpath in param_paths:
            ref_val = "N/A"
            if xml_root is not None:
                elem = xml_root.find(xpath)
                if elem is not None and elem.text:
                    ref_val = elem.text.strip()
                else:
                    continue  # Skip if not found in XML
            xml_suggestions.append((ct, param_name, xpath, ref_val))
            result += f"| {param_idx} | {ct} | {param_name} | `{xpath}` | {ref_val} |\n"
            param_idx += 1

    # === Rules-based parameters ===
    result += "\n### Rules Parameters (Hill Function Coefficients)\n\n"
    result += "| # | Rule | Field | Key | Ref Value |\n"
    result += "|---|------|-------|-----|-----------|\n"

    rules_suggestions = []
    rules_csv_path = session.uq_context.rules_csv_path

    if rules_csv_path and Path(rules_csv_path).exists():
        with open(rules_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                cell_type, signal, direction, behavior = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                base_val, half_max, hill_power, apply_to_dead = row[4].strip(), row[5].strip(), row[6].strip(), row[7].strip()

                rule_desc = f"{cell_type}: {signal} {direction} {behavior}"

                for field_name, field_val in [("saturation", base_val), ("half_max", half_max), ("hill_power", hill_power)]:
                    rule_key = f"{cell_type},{signal},{direction},{behavior},{field_name}"
                    rules_suggestions.append((rule_desc, field_name, rule_key, field_val))
                    result += f"| {param_idx} | {rule_desc} | {field_name} | `{rule_key}` | {field_val} |\n"
                    param_idx += 1
    else:
        result += "*(No cell_rules.csv found)*\n"

    result += f"\n**Total suggestions:** {param_idx - 1} parameters\n\n"
    result += "**Next step:** Use `define_uq_parameters()` to select which parameters to include in your analysis.\n"
    result += "Pass parameters as a list of dicts with 'name', 'type' ('xml'/'rules'), 'xpath' or 'rule_key', "
    result += "'ref_value', 'lower_bound', 'upper_bound'."

    return result


@mcp.tool()
def define_uq_parameters(parameters: List[Dict[str, Any]]) -> str:
    """
    Define which parameters to vary in the UQ analysis. Each parameter needs a name,
    type (xml or rules), the xpath or rule_key, reference value, and bounds.

    Args:
        parameters: List of parameter definitions, each a dict with:
            - name (str): Human-readable parameter name (e.g., 'tumor_cycle_hfm')
            - type (str): 'xml' or 'rules'
            - xpath (str, for xml): XPath to the XML element
            - rule_key (str, for rules): Rule key like 'tumor,oxygen,increases,cycle entry,half_max'
            - ref_value (float): Reference/default value
            - lower_bound (float): Lower bound for sampling
            - upper_bound (float): Upper bound for sampling

    Returns:
        str: Summary of defined parameters
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    uq_ctx = session.uq_context
    uq_ctx.parameters = []

    errors = []
    for i, p in enumerate(parameters):
        name = p.get("name")
        ptype = p.get("type")
        if not name:
            errors.append(f"Parameter {i+1}: missing 'name'")
            continue
        if ptype not in ("xml", "rules"):
            errors.append(f"Parameter '{name}': type must be 'xml' or 'rules', got '{ptype}'")
            continue

        param_def = UQParameterDef(
            name=name,
            param_type=ptype,
            xpath=p.get("xpath"),
            rule_key=p.get("rule_key"),
            ref_value=p.get("ref_value"),
            lower_bound=p.get("lower_bound"),
            upper_bound=p.get("upper_bound"),
            perturbation=p.get("perturbation"),
        )

        if ptype == "xml" and not param_def.xpath:
            errors.append(f"Parameter '{name}': XML type requires 'xpath'")
            continue
        if ptype == "rules" and not param_def.rule_key:
            errors.append(f"Parameter '{name}': rules type requires 'rule_key'")
            continue

        # Auto-compute bounds if not provided (±50% of ref_value)
        if param_def.ref_value is not None:
            if param_def.lower_bound is None:
                param_def.lower_bound = param_def.ref_value * 0.5
            if param_def.upper_bound is None:
                param_def.upper_bound = param_def.ref_value * 1.5

        uq_ctx.parameters.append(param_def)

    session.mark_step_complete(WorkflowStep.UQ_PARAMETERS_DEFINED)

    result = "## UQ Parameters Defined\n\n"
    if errors:
        result += "### Warnings\n"
        for e in errors:
            result += f"- {e}\n"
        result += "\n"

    result += f"**Total parameters:** {len(uq_ctx.parameters)}\n\n"
    result += "| Name | Type | Key/XPath | Ref | Bounds |\n"
    result += "|------|------|-----------|-----|--------|\n"
    for p in uq_ctx.parameters:
        key = p.xpath if p.param_type == "xml" else p.rule_key
        # Truncate long keys for display
        key_display = key if len(key) < 50 else "..." + key[-47:]
        bounds = f"[{p.lower_bound}, {p.upper_bound}]" if p.lower_bound is not None else "auto"
        result += f"| {p.name} | {p.param_type} | `{key_display}` | {p.ref_value} | {bounds} |\n"

    result += "\n**Next step:** Use `define_quantities_of_interest()` to specify what to measure from simulations."
    return result


@mcp.tool()
def define_quantities_of_interest(qois: List[Dict[str, str]],
                                  time_column: str = "time") -> str:
    """
    Define what quantities to measure from simulation outputs. QoIs are computed
    from PhysiCell output DataFrames using lambda expressions or predefined functions.

    Args:
        qois: List of QoI definitions, each a dict with:
            - name (str): QoI identifier (e.g., 'live_tumor_cells')
            - function (str): Lambda expression on cell DataFrame, OR one of:
                'live_cells' - total live cells
                'dead_cells' - total dead cells
                'cell_count:<type>' - count of specific cell type (e.g., 'cell_count:tumor')
            - obs_column (str, optional): Column name in experimental data CSV mapping to this QoI
        time_column: Column name for time in experimental data (default: 'time')

    Returns:
        str: Summary of defined QoIs
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    uq_ctx = session.uq_context
    uq_ctx.qoi_definitions = {}
    uq_ctx.experimental_data_columns = {"time": time_column}

    # Predefined QoI templates
    predefined = {
        "live_cells": "lambda df: len(df[df['dead'] == False])",
        "dead_cells": "lambda df: len(df[df['dead'] == True])",
    }

    result_qois = []
    for q in qois:
        name = q.get("name")
        func = q.get("function", "")
        obs_col = q.get("obs_column")

        if not name:
            continue

        # Handle predefined functions
        if func in predefined:
            func = predefined[func]
        elif func.startswith("cell_count:"):
            cell_type = func.split(":", 1)[1]
            func = f"lambda df: len(df[df['cell_type'] == '{cell_type}'])"

        uq_ctx.qoi_definitions[name] = func
        if obs_col:
            uq_ctx.experimental_data_columns[name] = obs_col
        result_qois.append((name, func, obs_col))

    session.mark_step_complete(WorkflowStep.UQ_QOIS_DEFINED)

    result = "## Quantities of Interest Defined\n\n"
    result += f"**Total QoIs:** {len(result_qois)}\n\n"
    result += "| Name | Function | Obs. Column |\n"
    result += "|------|----------|-------------|\n"
    for name, func, obs_col in result_qois:
        func_display = func if len(func) < 60 else func[:57] + "..."
        result += f"| {name} | `{func_display}` | {obs_col or '-'} |\n"

    result += "\n**Next steps:**\n"
    result += "- `run_sensitivity_analysis()` - Analyze parameter sensitivity\n"
    result += "- `provide_experimental_data()` - Load reference data for calibration\n"
    return result


# ============================================================================
# UQ PHASE 2: SENSITIVITY ANALYSIS
# ============================================================================

@mcp.tool()
def run_sensitivity_analysis(method: str = "Sobol",
                             num_samples: int = 64,
                             num_workers: int = 4,
                             parallel_method: str = "inter-process") -> str:
    """
    Run sensitivity analysis on the current model to identify which parameters
    most influence the simulation outputs (QoIs).

    PREREQUISITES: setup_uq_analysis(), define_uq_parameters(), define_quantities_of_interest()

    Args:
        method: Sampling/SA method - 'Sobol', 'LHS' (Latin Hypercube), 'OAT' (One-at-a-Time),
                'Fast', 'Fractional Factorial' (default: 'Sobol')
        num_samples: Number of parameter samples (default: 64, higher = more accurate but slower)
        num_workers: Number of parallel simulation workers (default: 4)
        parallel_method: 'serial', 'inter-process', or 'inter-node' (default: 'inter-process')

    Returns:
        str: Sensitivity analysis status and run ID
    """
    _ensure_uq_imports()
    if not UQ_AVAILABLE:
        return "**Error:** uq-physicell not installed. Run: `pip install uq-physicell`"

    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    uq_ctx = session.uq_context
    if not uq_ctx.parameters:
        return "**Error:** No parameters defined. Run `define_uq_parameters()` first."

    # Generate INI config
    ini_path = _generate_uq_ini(session, uq_ctx)
    uq_ctx.ini_path = ini_path

    # Set up SA database path
    sa_db_name = f"sa_{method.lower().replace(' ', '_')}_{int(time.time())}.db"
    sa_db_path = str(Path(uq_ctx.uq_output_dir) / sa_db_name)
    uq_ctx.sa_db_path = sa_db_path
    uq_ctx.sa_method = method
    uq_ctx.sa_num_samples = num_samples

    # Build params_info for ModelAnalysisContext
    is_local = method.upper() == "OAT"
    params_info = {}
    for p in uq_ctx.parameters:
        if is_local:
            params_info[p.name] = {
                "ref_value": p.ref_value or 1.0,
                "perturbation": p.perturbation or [1.0, 5.0, 10.0],
            }
        else:
            params_info[p.name] = {
                "ref_value": p.ref_value or 1.0,
                "lower_bound": p.lower_bound or (p.ref_value * 0.5 if p.ref_value else 0.1),
                "upper_bound": p.upper_bound or (p.ref_value * 1.5 if p.ref_value else 10.0),
                "perturbation": 50.0,
            }

    model_config = {
        "ini_path": ini_path,
        "struc_name": "uq_model",
    }

    run_id = f"sa_{uuid.uuid4().hex[:8]}"

    # Run SA in a background thread
    def _run_sa():
        try:
            with uq_runs_lock:
                uq_runs[run_id] = {
                    "type": "sensitivity_analysis",
                    "status": "running",
                    "method": method,
                    "num_samples": num_samples,
                    "db_path": sa_db_path,
                    "started_at": time.time(),
                    "error": None,
                }

            context = ModelAnalysisContext(
                db_path=sa_db_path,
                model_config=model_config,
                sampler=method,
                params_info=params_info,
                qois_info=uq_ctx.qoi_definitions,
                parallel_method=parallel_method,
                num_workers=num_workers,
            )

            if is_local:
                from uq_physicell.model_analysis import run_local_sampler
                context.dic_samples = run_local_sampler(context.params_dict, method)
            else:
                from uq_physicell.model_analysis import run_global_sampler
                context.dic_samples = run_global_sampler(context.params_dict, method, N=num_samples)

            uq_run_simulations(context)

            with uq_runs_lock:
                uq_runs[run_id]["status"] = "completed"
                uq_runs[run_id]["completed_at"] = time.time()
                uq_runs[run_id]["num_simulations"] = len(context.dic_samples)

            uq_ctx.sa_results = {"run_id": run_id, "db_path": sa_db_path}
            session.mark_step_complete(WorkflowStep.SENSITIVITY_ANALYSIS_COMPLETE)

        except Exception as e:
            with uq_runs_lock:
                uq_runs[run_id]["status"] = "failed"
                uq_runs[run_id]["error"] = str(e)

    thread = threading.Thread(target=_run_sa, daemon=True)
    thread.start()

    total_sims = num_samples * uq_ctx.num_replicates * (len(uq_ctx.parameters) + 1) if is_local else num_samples * uq_ctx.num_replicates
    result = "## Sensitivity Analysis Started\n\n"
    result += f"**Run ID:** `{run_id}`\n"
    result += f"**Method:** {method}\n"
    result += f"**Samples:** {num_samples}\n"
    result += f"**Est. simulations:** ~{total_sims}\n"
    result += f"**Workers:** {num_workers}\n"
    result += f"**Database:** {sa_db_path}\n\n"
    result += "The analysis is running in the background. Use `get_sensitivity_results(run_id)` to check progress and results."
    return result


@mcp.tool()
def get_sensitivity_results(run_id: Optional[str] = None) -> str:
    """
    Get results from a sensitivity analysis run. If the run is still in progress,
    returns the current status. If complete, returns sensitivity indices and rankings.

    Args:
        run_id: SA run ID (from run_sensitivity_analysis). Uses latest if omitted.

    Returns:
        str: Sensitivity analysis results or status
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** No UQ context. Run `setup_uq_analysis()` first."

    # Find the run
    if not run_id:
        # Find latest SA run
        with uq_runs_lock:
            sa_runs = {k: v for k, v in uq_runs.items() if v["type"] == "sensitivity_analysis"}
            if not sa_runs:
                return "**Error:** No sensitivity analysis runs found. Run `run_sensitivity_analysis()` first."
            run_id = max(sa_runs.keys(), key=lambda k: sa_runs[k].get("started_at", 0))

    with uq_runs_lock:
        run_info = uq_runs.get(run_id)

    if not run_info:
        return f"**Error:** Run '{run_id}' not found."

    result = f"## Sensitivity Analysis: `{run_id}`\n\n"
    result += f"**Method:** {run_info.get('method', 'Unknown')}\n"
    result += f"**Status:** {run_info['status']}\n"

    if run_info["status"] == "running":
        elapsed = time.time() - run_info.get("started_at", time.time())
        result += f"**Elapsed:** {elapsed/60:.1f} minutes\n\n"
        result += "Analysis is still running. Check back later."
        return result

    if run_info["status"] == "failed":
        result += f"**Error:** {run_info.get('error', 'Unknown error')}\n"
        return result

    if run_info["status"] == "completed":
        elapsed = run_info.get("completed_at", time.time()) - run_info.get("started_at", time.time())
        result += f"**Completed in:** {elapsed/60:.1f} minutes\n"
        result += f"**Simulations run:** {run_info.get('num_simulations', 'N/A')}\n"
        result += f"**Database:** {run_info.get('db_path', 'N/A')}\n\n"

        # Try to load and display SA results from the database
        try:
            import sqlite3
            db_path = run_info.get("db_path")
            if db_path and Path(db_path).exists():
                conn = sqlite3.connect(db_path)
                # Check what tables exist
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                result += f"**Database tables:** {', '.join(tables)}\n\n"

                # Try to read simulation results
                if "simulations" in tables:
                    cursor = conn.execute("SELECT COUNT(*) FROM simulations")
                    count = cursor.fetchone()[0]
                    result += f"**Completed simulations:** {count}\n\n"

                conn.close()

            result += "**Analysis complete!** The results database contains all simulation outputs.\n\n"
            result += "**Next steps:**\n"
            result += "- `provide_experimental_data()` - Load reference data for calibration\n"
            result += "- `run_bayesian_calibration()` - Calibrate model parameters\n"

        except Exception as e:
            result += f"**Note:** Could not read database details: {e}\n"

    return result


# ============================================================================
# UQ PHASE 3: MODEL CALIBRATION
# ============================================================================

@mcp.tool()
def provide_experimental_data(csv_path: str,
                              column_mapping: Dict[str, str],
                              time_column: str = "time") -> str:
    """
    Load experimental/reference data for model calibration. The CSV should contain
    time-series data with columns matching the Quantities of Interest.

    Args:
        csv_path: Path to CSV file with experimental observations
        column_mapping: Dict mapping QoI names to CSV column names,
            e.g., {"live_tumor_cells": "Tumor Count", "dead_cells": "Dead Count"}
        time_column: Name of the time column in the CSV (default: 'time')

    Returns:
        str: Summary of loaded experimental data
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    csv_file = Path(csv_path)
    if not csv_file.exists():
        return f"**Error:** File not found: {csv_path}"

    uq_ctx = session.uq_context
    uq_ctx.experimental_data_path = str(csv_file)
    uq_ctx.experimental_data_columns = {"time": time_column}
    uq_ctx.experimental_data_columns.update(column_mapping)

    # Read and validate the CSV
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        rows = len(df)
        cols = list(df.columns)

        # Validate columns exist
        missing = []
        if time_column not in cols:
            missing.append(f"time column '{time_column}'")
        for qoi_name, col_name in column_mapping.items():
            if col_name not in cols:
                missing.append(f"'{col_name}' (for QoI '{qoi_name}')")

        session.mark_step_complete(WorkflowStep.EXPERIMENTAL_DATA_LOADED)

        result = "## Experimental Data Loaded\n\n"
        result += f"**File:** {csv_path}\n"
        result += f"**Rows:** {rows}\n"
        result += f"**Columns:** {', '.join(cols)}\n\n"

        if missing:
            result += "### Warnings - Missing Columns\n"
            for m in missing:
                result += f"- {m}\n"
            result += "\n"

        result += "### Column Mapping\n"
        result += "| QoI | CSV Column |\n|-----|------------|\n"
        result += f"| time | {time_column} |\n"
        for qoi_name, col_name in column_mapping.items():
            result += f"| {qoi_name} | {col_name} |\n"

        # Show data preview
        result += f"\n### Data Preview (first 5 rows)\n```\n{df.head().to_string()}\n```\n\n"

        time_range = df[time_column]
        result += f"**Time range:** {time_range.min()} to {time_range.max()}\n\n"

        result += "**Next step:** Use `run_bayesian_calibration()` or `run_abc_calibration()` to fit model parameters."
        return result

    except Exception as e:
        return f"**Error reading CSV:** {str(e)}"


@mcp.tool()
def run_bayesian_calibration(
    num_initial_samples: int = 10,
    num_iterations: int = 50,
    max_workers: int = 4,
    distance_metric: str = "sum_squared_differences",
    use_exponential_fitness: bool = True,
    search_space_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Run Bayesian Optimization calibration to find parameters that best match
    experimental data. Uses multi-objective optimization with Pareto front analysis.

    PREREQUISITES: setup_uq_analysis(), define_uq_parameters(),
    define_quantities_of_interest(), provide_experimental_data()

    Args:
        num_initial_samples: Initial random samples before optimization (default: 10)
        num_iterations: Number of BO iterations (default: 50)
        max_workers: Parallel simulation workers (default: 4)
        distance_metric: 'sum_squared_differences', 'manhattan', or 'chebyshev' (default: 'sum_squared_differences')
        use_exponential_fitness: Use exponential fitness scaling (default: True)
        search_space_overrides: Optional per-parameter bound overrides,
            e.g., {"param_name": {"lower_bound": 0.1, "upper_bound": 2.0}}

    Returns:
        str: Calibration status and run ID
    """
    _ensure_uq_imports()
    if not UQ_BO_AVAILABLE:
        return "**Error:** Bayesian optimization requires: `pip install torch botorch gpytorch`"

    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    uq_ctx = session.uq_context
    if not uq_ctx.parameters:
        return "**Error:** No parameters defined. Run `define_uq_parameters()` first."
    if not uq_ctx.experimental_data_path:
        return "**Error:** No experimental data. Run `provide_experimental_data()` first."
    if not uq_ctx.qoi_definitions:
        return "**Error:** No QoIs defined. Run `define_quantities_of_interest()` first."

    # Generate INI config
    ini_path = _generate_uq_ini(session, uq_ctx)
    uq_ctx.ini_path = ini_path

    # Set up calibration database
    cal_db_name = f"calibration_bo_{int(time.time())}.db"
    cal_db_path = str(Path(uq_ctx.uq_output_dir) / cal_db_name)
    uq_ctx.calibration_db_path = cal_db_path
    uq_ctx.calibration_method = "bayesian_optimization"

    # Select distance function
    dist_func_map = {
        "sum_squared_differences": SumSquaredDifferences,
        "manhattan": Manhattan,
        "chebyshev": Chebyshev,
    }
    dist_func = dist_func_map.get(distance_metric, SumSquaredDifferences)

    # Build search space
    search_space = {}
    for p in uq_ctx.parameters:
        overrides = (search_space_overrides or {}).get(p.name, {})
        search_space[p.name] = {
            "type": "real",
            "lower_bound": overrides.get("lower_bound", p.lower_bound or 0.01),
            "upper_bound": overrides.get("upper_bound", p.upper_bound or 10.0),
        }

    # Build distance functions dict (one per QoI with equal weights)
    distance_functions = {}
    for qoi_name in uq_ctx.qoi_definitions:
        distance_functions[qoi_name] = {
            "function": dist_func,
            "weight": 1e-5,
        }

    model_config = {
        "ini_path": ini_path,
        "struc_name": "uq_model",
        "numReplicates": uq_ctx.num_replicates,
    }

    bo_options = {
        "num_initial_samples": num_initial_samples,
        "num_iterations": num_iterations,
        "max_workers": max_workers,
        "use_exponential_fitness": use_exponential_fitness,
    }

    run_id = f"bo_{uuid.uuid4().hex[:8]}"

    # Run calibration in background thread
    def _run_bo():
        try:
            with uq_runs_lock:
                uq_runs[run_id] = {
                    "type": "bayesian_optimization",
                    "status": "running",
                    "db_path": cal_db_path,
                    "started_at": time.time(),
                    "num_iterations": num_iterations,
                    "num_initial_samples": num_initial_samples,
                    "error": None,
                }

            calib_context = BOCalibrationContext(
                db_path=cal_db_path,
                obsData=uq_ctx.experimental_data_path,
                obsData_columns=uq_ctx.experimental_data_columns,
                model_config=model_config,
                qoi_functions=uq_ctx.qoi_definitions,
                distance_functions=distance_functions,
                search_space=search_space,
                bo_options=bo_options,
            )

            uq_run_bo(calib_context)

            with uq_runs_lock:
                uq_runs[run_id]["status"] = "completed"
                uq_runs[run_id]["completed_at"] = time.time()

            uq_ctx.calibration_results = {"run_id": run_id, "db_path": cal_db_path}
            session.mark_step_complete(WorkflowStep.CALIBRATION_COMPLETE)

        except Exception as e:
            with uq_runs_lock:
                uq_runs[run_id]["status"] = "failed"
                uq_runs[run_id]["error"] = str(e)

    thread = threading.Thread(target=_run_bo, daemon=True)
    thread.start()

    total_sims = (num_initial_samples + num_iterations) * uq_ctx.num_replicates
    result = "## Bayesian Optimization Calibration Started\n\n"
    result += f"**Run ID:** `{run_id}`\n"
    result += f"**Initial samples:** {num_initial_samples}\n"
    result += f"**Iterations:** {num_iterations}\n"
    result += f"**Est. total simulations:** ~{total_sims}\n"
    result += f"**Distance metric:** {distance_metric}\n"
    result += f"**Workers:** {max_workers}\n"
    result += f"**Database:** {cal_db_path}\n\n"

    result += "### Search Space\n"
    result += "| Parameter | Lower | Upper |\n|-----------|-------|-------|\n"
    for name, space in search_space.items():
        result += f"| {name} | {space['lower_bound']} | {space['upper_bound']} |\n"

    result += "\nCalibration is running in the background. Use `get_calibration_status()` to monitor progress "
    result += "and `get_calibration_results()` when complete."
    return result


@mcp.tool()
def run_abc_calibration(
    prior_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    max_populations: int = 8,
    max_simulations: int = 500,
    min_population_size: int = 30,
    max_population_size: int = 100,
    num_workers: int = 4,
    fixed_params: Optional[Dict[str, float]] = None,
) -> str:
    """
    Run Approximate Bayesian Computation (ABC-SMC) calibration to infer posterior
    parameter distributions given experimental data. Returns full uncertainty estimates.

    PREREQUISITES: setup_uq_analysis(), define_uq_parameters(),
    define_quantities_of_interest(), provide_experimental_data()

    Args:
        prior_bounds: Per-parameter prior bounds as uniform distributions,
            e.g., {"param": {"lower": 0.1, "upper": 2.0}}. Uses parameter bounds if omitted.
        max_populations: Maximum ABC-SMC populations/generations (default: 8)
        max_simulations: Maximum total simulations (default: 500)
        min_population_size: Minimum particles per population (default: 30)
        max_population_size: Maximum particles per population (default: 100)
        num_workers: Parallel workers (default: 4)
        fixed_params: Parameters to fix at specific values (not sampled),
            e.g., {"param_name": 0.5}

    Returns:
        str: Calibration status and run ID
    """
    _ensure_uq_imports()
    if not UQ_ABC_AVAILABLE:
        return "**Error:** ABC calibration requires: `pip install pyabc`"

    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** Run `setup_uq_analysis()` first."

    uq_ctx = session.uq_context
    if not uq_ctx.parameters:
        return "**Error:** No parameters defined. Run `define_uq_parameters()` first."
    if not uq_ctx.experimental_data_path:
        return "**Error:** No experimental data. Run `provide_experimental_data()` first."

    # Generate INI config
    ini_path = _generate_uq_ini(session, uq_ctx)
    uq_ctx.ini_path = ini_path

    # Set up calibration database
    cal_db_name = f"calibration_abc_{int(time.time())}.db"
    cal_db_path = str(Path(uq_ctx.uq_output_dir) / cal_db_name)
    uq_ctx.calibration_db_path = cal_db_path
    uq_ctx.calibration_method = "abc"

    # Build prior distributions
    try:
        from pyabc import RV, Distribution
    except ImportError:
        return "**Error:** pyabc not installed. Run: `pip install pyabc`"

    prior_dict = {}
    params_to_sample = [p for p in uq_ctx.parameters if p.name not in (fixed_params or {})]

    for p in params_to_sample:
        overrides = (prior_bounds or {}).get(p.name, {})
        lower = overrides.get("lower", p.lower_bound or 0.01)
        upper = overrides.get("upper", p.upper_bound or 10.0)
        prior_dict[p.name] = RV("uniform", lower, upper - lower)

    prior = Distribution(**prior_dict)

    model_config = {
        "ini_path": ini_path,
        "struc_name": "uq_model",
    }

    # Build distance functions
    import numpy as np

    distance_functions = {}
    for qoi_name in uq_ctx.qoi_definitions:
        def make_dist_func(qname):
            def dist_func(data1, data2):
                try:
                    obs_vals = np.array(data1[qname])
                    sim_vals = np.array(data2[qname])
                    return float(np.sum((obs_vals - sim_vals) ** 2))
                except Exception:
                    return float("inf")
            return dist_func
        distance_functions[qoi_name] = {
            "function": make_dist_func(qoi_name),
            "weight": 1.0,
        }

    abc_options = {
        "max_populations": max_populations,
        "max_simulations": max_simulations,
        "population_strategy": "adaptive",
        "min_population_size": min_population_size,
        "max_population_size": max_population_size,
        "epsilon_strategy": "quantile",
        "epsilon_alpha": 0.5,
        "transition_strategy": "multivariate",
        "sampler": "multicore",
        "num_workers": num_workers,
        "mode": "local",
    }

    if fixed_params:
        abc_options["fixed_params"] = fixed_params

    run_id = f"abc_{uuid.uuid4().hex[:8]}"

    # Run in background
    def _run_abc():
        try:
            with uq_runs_lock:
                uq_runs[run_id] = {
                    "type": "abc",
                    "status": "running",
                    "db_path": cal_db_path,
                    "started_at": time.time(),
                    "max_populations": max_populations,
                    "max_simulations": max_simulations,
                    "error": None,
                }

            calib_context = ABCCalibrationContext(
                db_path=cal_db_path,
                obsData=uq_ctx.experimental_data_path,
                obsData_columns=uq_ctx.experimental_data_columns,
                model_config=model_config,
                qoi_functions=uq_ctx.qoi_definitions,
                distance_functions=distance_functions,
                prior=prior,
                abc_options=abc_options,
            )

            history = uq_run_abc(calib_context)

            with uq_runs_lock:
                uq_runs[run_id]["status"] = "completed"
                uq_runs[run_id]["completed_at"] = time.time()
                uq_runs[run_id]["n_populations"] = history.n_populations
                uq_runs[run_id]["total_simulations"] = history.total_nr_simulations

            uq_ctx.calibration_results = {"run_id": run_id, "db_path": cal_db_path}
            session.mark_step_complete(WorkflowStep.CALIBRATION_COMPLETE)

        except Exception as e:
            with uq_runs_lock:
                uq_runs[run_id]["status"] = "failed"
                uq_runs[run_id]["error"] = str(e)

    thread = threading.Thread(target=_run_abc, daemon=True)
    thread.start()

    result = "## ABC-SMC Calibration Started\n\n"
    result += f"**Run ID:** `{run_id}`\n"
    result += f"**Max populations:** {max_populations}\n"
    result += f"**Max simulations:** {max_simulations}\n"
    result += f"**Population size:** {min_population_size}-{max_population_size}\n"
    result += f"**Workers:** {num_workers}\n"
    result += f"**Database:** {cal_db_path}\n\n"

    result += "### Prior Distributions\n"
    result += "| Parameter | Distribution |\n|-----------|-------------|\n"
    for p in params_to_sample:
        overrides = (prior_bounds or {}).get(p.name, {})
        lower = overrides.get("lower", p.lower_bound or 0.01)
        upper = overrides.get("upper", p.upper_bound or 10.0)
        result += f"| {p.name} | Uniform({lower}, {upper}) |\n"

    if fixed_params:
        result += "\n### Fixed Parameters\n"
        for name, val in fixed_params.items():
            result += f"- {name} = {val}\n"

    result += "\nCalibration is running in the background. Use `get_calibration_status()` to monitor."
    return result


@mcp.tool()
def get_calibration_status(run_id: Optional[str] = None) -> str:
    """
    Check the status of a running or completed calibration job.

    Args:
        run_id: Calibration run ID. Uses latest if omitted.

    Returns:
        str: Current status of the calibration run
    """
    if not run_id:
        with uq_runs_lock:
            cal_runs = {k: v for k, v in uq_runs.items()
                        if v["type"] in ("bayesian_optimization", "abc")}
            if not cal_runs:
                return "**Error:** No calibration runs found."
            run_id = max(cal_runs.keys(), key=lambda k: cal_runs[k].get("started_at", 0))

    with uq_runs_lock:
        run_info = uq_runs.get(run_id)

    if not run_info:
        return f"**Error:** Run '{run_id}' not found."

    result = f"## Calibration Status: `{run_id}`\n\n"
    result += f"**Type:** {run_info['type'].replace('_', ' ').title()}\n"
    result += f"**Status:** {run_info['status']}\n"

    elapsed = time.time() - run_info.get("started_at", time.time())
    if run_info["status"] == "running":
        result += f"**Elapsed:** {elapsed/60:.1f} minutes\n"
        if run_info["type"] == "bayesian_optimization":
            result += f"**Target iterations:** {run_info.get('num_iterations', 'N/A')}\n"
        elif run_info["type"] == "abc":
            result += f"**Max populations:** {run_info.get('max_populations', 'N/A')}\n"
            result += f"**Max simulations:** {run_info.get('max_simulations', 'N/A')}\n"
    elif run_info["status"] == "completed":
        total_time = run_info.get("completed_at", time.time()) - run_info.get("started_at", time.time())
        result += f"**Completed in:** {total_time/60:.1f} minutes\n"
        if "n_populations" in run_info:
            result += f"**Populations completed:** {run_info['n_populations']}\n"
        if "total_simulations" in run_info:
            result += f"**Total simulations:** {run_info['total_simulations']}\n"
        result += "\nUse `get_calibration_results()` to view the results."
    elif run_info["status"] == "failed":
        result += f"**Error:** {run_info.get('error', 'Unknown')}\n"

    return result


@mcp.tool()
def get_calibration_results(run_id: Optional[str] = None) -> str:
    """
    Retrieve results from a completed calibration run, including best-fit parameters,
    Pareto front analysis (for BO), or posterior distributions (for ABC).

    Args:
        run_id: Calibration run ID. Uses latest completed if omitted.

    Returns:
        str: Calibration results with best-fit parameters
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** No UQ context."

    if not run_id:
        with uq_runs_lock:
            cal_runs = {k: v for k, v in uq_runs.items()
                        if v["type"] in ("bayesian_optimization", "abc") and v["status"] == "completed"}
            if not cal_runs:
                return "**Error:** No completed calibration runs. Check status with `get_calibration_status()`."
            run_id = max(cal_runs.keys(), key=lambda k: cal_runs[k].get("completed_at", 0))

    with uq_runs_lock:
        run_info = uq_runs.get(run_id)

    if not run_info:
        return f"**Error:** Run '{run_id}' not found."
    if run_info["status"] != "completed":
        return f"**Error:** Run '{run_id}' is {run_info['status']}, not completed."

    db_path = run_info.get("db_path")
    result = f"## Calibration Results: `{run_id}`\n\n"
    result += f"**Type:** {run_info['type'].replace('_', ' ').title()}\n"
    total_time = run_info.get("completed_at", 0) - run_info.get("started_at", 0)
    result += f"**Total time:** {total_time/60:.1f} minutes\n"
    result += f"**Database:** {db_path}\n\n"

    try:
        if run_info["type"] == "bayesian_optimization":
            from uq_physicell.database.bo_db import load_structure
            df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_path)

            result += "### Parameter Space\n"
            result += "| Parameter | Lower | Upper |\n|-----------|-------|-------|\n"
            for _, row in df_param_space.iterrows():
                result += f"| {row['ParamName']} | {row['LowerBound']} | {row['UpperBound']} |\n"

            result += f"\n### Samples Evaluated: {len(df_samples)}\n\n"

            # Pareto analysis
            try:
                from uq_physicell.bo import analyze_pareto_results
                pareto_data = analyze_pareto_results(df_qois, df_samples, df_output)

                if pareto_data and "pareto_front" in pareto_data:
                    pareto_params = pareto_data["pareto_front"].get("parameters", [])
                    pareto_ids = pareto_data["pareto_front"].get("sample_ids", [])

                    result += f"### Pareto Front: {len(pareto_params)} optimal solutions\n\n"

                    if pareto_params:
                        # Display best parameters
                        param_names = list(df_param_space["ParamName"])
                        result += "| Solution |"
                        for pn in param_names:
                            result += f" {pn} |"
                        result += "\n|----------|"
                        for _ in param_names:
                            result += "--------|"
                        result += "\n"

                        for i, params in enumerate(pareto_params[:5]):  # Show top 5
                            result += f"| {i+1} |"
                            if isinstance(params, dict):
                                for pn in param_names:
                                    result += f" {params.get(pn, 'N/A'):.4g} |"
                            else:
                                for val in (params if hasattr(params, '__iter__') else [params]):
                                    result += f" {float(val):.4g} |"
                            result += "\n"

                        # Store best-fit in session
                        if pareto_params:
                            best = pareto_params[0]
                            if isinstance(best, dict):
                                session.uq_context.best_fit_parameters = best
                            else:
                                session.uq_context.best_fit_parameters = dict(zip(param_names, best))
            except Exception as e:
                result += f"*(Pareto analysis error: {e})*\n"

        elif run_info["type"] == "abc":
            if "n_populations" in run_info:
                result += f"### ABC-SMC Summary\n"
                result += f"**Populations:** {run_info['n_populations']}\n"
                result += f"**Total simulations:** {run_info.get('total_simulations', 'N/A')}\n\n"

            # Try to read posterior from pyabc database
            try:
                import pyabc
                history = pyabc.History(f"sqlite:///{db_path}")
                df_posterior, weights = history.get_distribution(m=0, t=history.max_t)

                result += "### Posterior Parameter Estimates\n\n"
                result += "| Parameter | Mean | Std | Median | 95% CI |\n"
                result += "|-----------|------|-----|--------|--------|\n"

                best_fit = {}
                for col in df_posterior.columns:
                    mean_val = df_posterior[col].mean()
                    std_val = df_posterior[col].std()
                    median_val = df_posterior[col].median()
                    ci_low = df_posterior[col].quantile(0.025)
                    ci_high = df_posterior[col].quantile(0.975)
                    best_fit[col] = mean_val
                    result += f"| {col} | {mean_val:.4g} | {std_val:.4g} | {median_val:.4g} | [{ci_low:.4g}, {ci_high:.4g}] |\n"

                session.uq_context.best_fit_parameters = best_fit

            except Exception as e:
                result += f"*(Could not read ABC posterior: {e})*\n"

    except Exception as e:
        result += f"**Error reading results:** {str(e)}\n"

    result += "\n**Next step:** Use `apply_calibrated_parameters()` to update the model with best-fit parameters."
    return result


# ============================================================================
# UQ PHASE 4: VALIDATION & APPLICATION
# ============================================================================

@mcp.tool()
def apply_calibrated_parameters(parameter_overrides: Optional[Dict[str, float]] = None) -> str:
    """
    Apply the best-fit calibrated parameters back to the PhysiCell model configuration.
    Updates both XML and rules parameters in the session, ready for a validation run.

    Args:
        parameter_overrides: Optional manual parameter values to apply instead of
            auto-detected best-fit values. Dict of {param_name: value}.

    Returns:
        str: Summary of applied parameters and next steps
    """
    session = get_current_session()
    if not session or not session.uq_context:
        return "**Error:** No UQ context. Run calibration first."

    uq_ctx = session.uq_context
    params_to_apply = parameter_overrides or uq_ctx.best_fit_parameters

    if not params_to_apply:
        return "**Error:** No calibrated parameters available. Run calibration first or provide parameter_overrides."

    import xml.etree.ElementTree as ET

    xml_path = uq_ctx.xml_config_path
    if not xml_path or not Path(xml_path).exists():
        return f"**Error:** XML config not found at {xml_path}"

    # Parse XML
    tree = ET.parse(xml_path)
    xml_root = tree.getroot()

    applied_xml = []
    applied_rules = []
    errors = []

    for param_def in uq_ctx.parameters:
        if param_def.name not in params_to_apply:
            continue

        new_value = params_to_apply[param_def.name]

        if param_def.param_type == "xml" and param_def.xpath:
            elem = xml_root.find(param_def.xpath)
            if elem is not None:
                old_val = elem.text
                elem.text = str(new_value)
                applied_xml.append((param_def.name, old_val, new_value))
            else:
                errors.append(f"XML element not found: {param_def.xpath}")

        elif param_def.param_type == "rules" and param_def.rule_key:
            # Update rules CSV
            rules_path = uq_ctx.rules_csv_path
            if rules_path and Path(rules_path).exists():
                parts = param_def.rule_key.split(",")
                if len(parts) == 5:
                    cell_type, sig, direction, behavior, field = [p.strip() for p in parts]
                    field_map = {"saturation": 4, "half_max": 5, "hill_power": 6}
                    col_idx = field_map.get(field)

                    if col_idx is not None:
                        # Read, modify, write rules CSV
                        rows = []
                        with open(rules_path, 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                if (len(row) >= 7 and
                                    row[0].strip() == cell_type and
                                    row[1].strip() == sig and
                                    row[2].strip() == direction and
                                    row[3].strip() == behavior):
                                    old_val = row[col_idx]
                                    row[col_idx] = str(new_value)
                                    applied_rules.append((param_def.name, old_val, new_value))
                                rows.append(row)

                        with open(rules_path, 'w', newline='') as f:
                            writer = csv.writer(f, lineterminator='\n')
                            writer.writerows(rows)
                    else:
                        errors.append(f"Unknown rules field: {field}")
                else:
                    errors.append(f"Invalid rule key format: {param_def.rule_key}")
            else:
                errors.append(f"Rules CSV not found: {rules_path}")

    # Write updated XML
    tree.write(xml_path, xml_declaration=True, encoding="unicode")

    result = "## Calibrated Parameters Applied\n\n"

    if applied_xml:
        result += "### XML Parameters Updated\n"
        result += "| Parameter | Old Value | New Value |\n|-----------|-----------|----------|\n"
        for name, old, new in applied_xml:
            result += f"| {name} | {old} | {new:.6g} |\n"

    if applied_rules:
        result += "\n### Rules Parameters Updated\n"
        result += "| Parameter | Old Value | New Value |\n|-----------|-----------|----------|\n"
        for name, old, new in applied_rules:
            result += f"| {name} | {old} | {new:.6g} |\n"

    if errors:
        result += "\n### Errors\n"
        for e in errors:
            result += f"- {e}\n"

    total_applied = len(applied_xml) + len(applied_rules)
    result += f"\n**Total parameters applied:** {total_applied}\n"
    result += f"\n**Next step:** Run a validation simulation with `run_simulation('{uq_ctx.project_name}')` "
    result += "to verify the calibrated model matches experimental data."
    return result


@mcp.tool()
def get_uq_summary() -> str:
    """
    Get a complete summary of all UQ analysis work done in the current session,
    including setup status, defined parameters, SA results, and calibration results.

    Returns:
        str: Comprehensive UQ analysis summary
    """
    session = get_current_session()
    if not session:
        return "**Error:** No active session."

    uq_ctx = session.uq_context
    if not uq_ctx:
        return "**No UQ analysis configured.** Use `setup_uq_analysis()` to get started."

    result = "## UQ Analysis Summary\n\n"

    # Setup info
    result += "### Setup\n"
    result += f"- **Project:** {uq_ctx.project_name or 'Not set'}\n"
    result += f"- **Executable:** {uq_ctx.executable_path or 'Not set'}\n"
    result += f"- **XML Config:** {uq_ctx.xml_config_path or 'Not set'}\n"
    result += f"- **Rules CSV:** {uq_ctx.rules_csv_path or 'None'}\n"
    result += f"- **Replicates:** {uq_ctx.num_replicates}\n"
    result += f"- **Workers:** {uq_ctx.num_workers}\n\n"

    # Parameters
    result += f"### Parameters ({len(uq_ctx.parameters)})\n"
    if uq_ctx.parameters:
        result += "| Name | Type | Bounds |\n|------|------|--------|\n"
        for p in uq_ctx.parameters:
            bounds = f"[{p.lower_bound}, {p.upper_bound}]" if p.lower_bound is not None else "auto"
            result += f"| {p.name} | {p.param_type} | {bounds} |\n"
    else:
        result += "*(None defined)*\n"

    # QoIs
    result += f"\n### Quantities of Interest ({len(uq_ctx.qoi_definitions)})\n"
    if uq_ctx.qoi_definitions:
        for name, func in uq_ctx.qoi_definitions.items():
            result += f"- **{name}**: `{func[:60]}{'...' if len(func) > 60 else ''}`\n"
    else:
        result += "*(None defined)*\n"

    # Experimental data
    result += f"\n### Experimental Data\n"
    if uq_ctx.experimental_data_path:
        result += f"- **File:** {uq_ctx.experimental_data_path}\n"
        result += f"- **Column mapping:** {uq_ctx.experimental_data_columns}\n"
    else:
        result += "*(Not loaded)*\n"

    # SA results
    result += f"\n### Sensitivity Analysis\n"
    if uq_ctx.sa_results:
        run_id = uq_ctx.sa_results.get("run_id")
        with uq_runs_lock:
            run_info = uq_runs.get(run_id, {})
        result += f"- **Run ID:** {run_id}\n"
        result += f"- **Method:** {run_info.get('method', uq_ctx.sa_method)}\n"
        result += f"- **Status:** {run_info.get('status', 'unknown')}\n"
    else:
        result += "*(Not run)*\n"

    # Calibration results
    result += f"\n### Calibration\n"
    if uq_ctx.calibration_results:
        run_id = uq_ctx.calibration_results.get("run_id")
        with uq_runs_lock:
            run_info = uq_runs.get(run_id, {})
        result += f"- **Run ID:** {run_id}\n"
        result += f"- **Method:** {uq_ctx.calibration_method}\n"
        result += f"- **Status:** {run_info.get('status', 'unknown')}\n"
    else:
        result += "*(Not run)*\n"

    # Best-fit parameters
    if uq_ctx.best_fit_parameters:
        result += f"\n### Best-Fit Parameters\n"
        result += "| Parameter | Value |\n|-----------|-------|\n"
        for name, val in uq_ctx.best_fit_parameters.items():
            result += f"| {name} | {val:.6g} |\n"

    # Workflow status
    result += "\n### Workflow Progress\n"
    uq_steps = [
        (WorkflowStep.UQ_SETUP, "UQ Setup"),
        (WorkflowStep.UQ_PARAMETERS_DEFINED, "Parameters Defined"),
        (WorkflowStep.UQ_QOIS_DEFINED, "QoIs Defined"),
        (WorkflowStep.EXPERIMENTAL_DATA_LOADED, "Experimental Data Loaded"),
        (WorkflowStep.SENSITIVITY_ANALYSIS_COMPLETE, "Sensitivity Analysis"),
        (WorkflowStep.CALIBRATION_COMPLETE, "Calibration"),
    ]
    for step, label in uq_steps:
        status = "Done" if step in session.completed_steps else "Pending"
        marker = "[x]" if status == "Done" else "[ ]"
        result += f"- {marker} {label}\n"

    return result


@mcp.tool()
def list_uq_runs() -> str:
    """
    List all UQ analysis runs (sensitivity analysis and calibrations) with their status.

    Returns:
        str: Summary of all UQ runs
    """
    with uq_runs_lock:
        if not uq_runs:
            return "**No UQ runs found.** Start with `run_sensitivity_analysis()` or `run_bayesian_calibration()`."

        result = "## UQ Analysis Runs\n\n"
        result += "| Run ID | Type | Status | Started | Duration |\n"
        result += "|--------|------|--------|---------|----------|\n"

        for run_id, info in sorted(uq_runs.items(), key=lambda x: x[1].get("started_at", 0)):
            rtype = info["type"].replace("_", " ").title()
            status = info["status"]
            started = time.strftime("%H:%M:%S", time.localtime(info.get("started_at", 0)))

            if info.get("completed_at"):
                duration = f"{(info['completed_at'] - info['started_at'])/60:.1f} min"
            elif info["status"] == "running":
                duration = f"{(time.time() - info['started_at'])/60:.1f} min (running)"
            else:
                duration = "-"

            result += f"| `{run_id}` | {rtype} | {status} | {started} | {duration} |\n"

        return result


@mcp.tool()
def get_help() -> str:
    """
    When the user asks for help, available commands, or how to use the server,
    this function returns a guide to the available tools and their usage.

    Returns:
        str: Markdown-formatted help guide.
    """
    return """# PhysiCell MCP Server Help

## Complete Workflow (MUST follow this order!)

### Phase 1: Configure Simulation
1. **create_session()** - Initialize a simulation session (REQUIRED FIRST)
2. **analyze_biological_scenario()** - Store your biological context
3. **create_simulation_domain()** - Set up spatial/temporal framework
4. **add_single_substrate()** - Add oxygen, nutrients, drugs, etc.
5. **add_single_cell_type()** - Add cancer cells, immune cells, etc.
6. **configure_cell_parameters()** - Set cell volumes, motility, death rates
6b. **set_chemotaxis()** / **set_advanced_chemotaxis()** - Configure chemotactic migration
7. **add_single_cell_rule()** - Create realistic cell responses
7b. **place_initial_cells()** - (Optional) Place cells spatially for initial conditions

### Phase 2: Export Configuration
8. **export_xml_configuration()** - Generate PhysiCell XML (saves as PhysiCell_settings.xml)
9. **export_cell_rules_csv()** - Generate rules CSV (saves as cell_rules.csv)
9b. **export_cells_csv()** - (Optional) Export initial cell positions CSV (saves as cells.csv)

### Phase 3: Create, Compile, and Run Project
10. **create_physicell_project()** - Create project directory with config files
11. **compile_physicell_project()** - Compile the simulation executable
12. **run_simulation()** - Start the simulation (runs in background)

### Phase 4: Monitor and Visualize
13. **get_simulation_status()** - Check simulation progress
14. **generate_simulation_gif()** - Create GIF visualization when complete

### Phase 4b: Data Analysis (after simulation completes)
15. **get_simulation_analysis_overview()** - One-shot executive summary
16. **get_population_timeseries()** - Cell counts over time (growth curves)
17. **get_timestep_summary()** - Detailed snapshot at one timepoint
18. **get_substrate_summary()** - Substrate concentration statistics
19. **get_cell_data()** - Filtered cell attributes with statistics
20. **generate_analysis_plot()** - Save plots (population, scatter, contour)

### Phase 5: Uncertainty Quantification & Calibration (Optional)
15. **setup_uq_analysis()** - Initialize UQ for a compiled project
16. **get_uq_parameter_suggestions()** - See calibratable parameters
17. **define_uq_parameters()** - Select parameters to vary with bounds
18. **define_quantities_of_interest()** - Define what to measure
19. **run_sensitivity_analysis()** - Sobol/LHS/OAT sensitivity analysis
20. **get_sensitivity_results()** - View SA results
21. **provide_experimental_data()** - Load reference data CSV
22. **run_bayesian_calibration()** - Multi-objective Bayesian optimization
23. **run_abc_calibration()** - ABC-SMC posterior inference
24. **get_calibration_status()** - Check calibration progress
25. **get_calibration_results()** - View best-fit parameters
26. **apply_calibrated_parameters()** - Update model with calibrated values
27. **get_uq_summary()** - Overview of all UQ analysis work

### Phase 6: Literature Validation (Optional, requires LiteratureValidation MCP)
28. **get_rules_for_validation()** - Export rules for literature validation
29. *LiteratureValidation MCP:* validate_rules_batch() → get_validation_summary()
30. **store_validation_results()** - Save validation results in session
31. **get_validation_report()** - View full literature validation report

## Helper Functions
- **list_all_available_signals()** - See what signals cells can sense
- **list_all_available_behaviors()** - See what cells can do
- **get_simulation_summary()** - Check current setup
- **get_initial_conditions_summary()** - Review placed cell positions
- **remove_initial_cells()** - Remove placed cells
- **list_simulations()** - See all running/completed simulations
- **stop_simulation()** - Stop a running simulation
- **get_simulation_output_files()** - List output files
- **get_simulation_analysis_overview()** - Quick simulation summary
- **get_cell_data()** - Detailed cell data with filtering
- **list_uq_runs()** - See all UQ analysis runs
- **get_validation_report()** - View literature validation results

## IMPORTANT
- You MUST call tools in the order shown above
- Do NOT try to create XML files manually - use the tools
- Do NOT skip steps - each tool depends on previous steps being completed

Most parameters are optional with sensible defaults!"""

if __name__ == "__main__":
    mcp.run()
