import os
import shutil
import sys
import json
from pathlib import Path

# Make the repo root importable so we can use the shared artifact_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import maboss
import pandas as pd

from mcp.server.fastmcp import Context, FastMCP
from session_manager import session_manager, ensure_session, MaBoSSSession
from artifact_manager import get_artifact_dir, safe_artifact_path, list_artifacts, clean_artifacts

mcp = FastMCP("MaBoSS")

_SERVER_ROOT = Path(__file__).parent

# NOTE: global sim/result variables have been removed.
# All simulation state is now stored per-session in MaBoSSSession objects
# (see MaBoSS/session_manager.py).  Pass session_id to any tool that needs
# access to a simulation.

# ---------------------------------------------------------------------------
# Session management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_session(set_as_default: bool = True) -> str:
    """Create a new MaBoSS session.

    Args:
        set_as_default: If True (default), the new session becomes the active default.

    Returns:
        str: The new session ID.
    """
    sid = session_manager.create_session(set_as_default=set_as_default)
    return f"Session created: {sid}" + (" (set as default)" if set_as_default else "")


@mcp.tool()
def list_sessions() -> str:
    """List all active MaBoSS sessions and their state.

    Returns:
        str: Markdown-formatted session summary.
    """
    sessions = session_manager.list_sessions()
    if not sessions:
        return "No active sessions. Call create_session() to start one."
    lines = ["## MaBoSS Sessions\n"]
    for sid, info in sessions.items():
        default_marker = " **(default)**" if info["is_default"] else ""
        has_sim = "✓" if info["has_simulation"] else "✗"
        has_res = "✓" if info["has_result"] else "✗"
        lines.append(f"- **{sid}**{default_marker}: sim={has_sim}  result={has_res}  bnd={info['bnd_path'] or '—'}")
    return "\n".join(lines)


@mcp.tool()
def set_default_session(session_id: str) -> str:
    """Set the default (active) MaBoSS session.

    Args:
        session_id: Target session ID.

    Returns:
        str: Confirmation or error message.
    """
    if session_manager.set_default(session_id):
        return f"Default session set to: {session_id}"
    return f"Session not found: {session_id}"


@mcp.tool()
def delete_session(session_id: str, clean_files: bool = True) -> str:
    """Delete a MaBoSS session and optionally its artifact files.

    Args:
        session_id:  Session to delete.
        clean_files: If True (default), remove artifact files for the session.

    Returns:
        str: Confirmation or error message.
    """
    removed_files = 0
    if clean_files:
        removed_files = clean_artifacts(_SERVER_ROOT, session_id)
    if session_manager.delete_session(session_id):
        return f"Session {session_id} deleted." + (f" Removed {removed_files} artifact file(s)." if clean_files else "")
    return f"Session not found: {session_id}"


# ---------------------------------------------------------------------------
# Network / simulation tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_network_nodes(ctx: Context, session_id: str = None) -> str:
    """Retrieve the node names in the loaded MaBoSS network.

    Args:
        session_id: Session to query (default: active session).

    Returns:
        str: Comma-separated list of node names.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    nodes_list = list(sess.sim.network.keys())
    if not nodes_list:
        return "No nodes found in the MaBoSS network."
    ctx.info(f"Retrieved nodes: {nodes_list}")
    return f"Nodes in the MaBoSS network: {', '.join(nodes_list)}"

# tool for creating the bnd and the cfg files from a bnet file
@mcp.tool()
def bnet_to_bnd_and_cfg(bnet_path: str, ctx: Context, session_id: str = None) -> str:
    """Convert a .bnet file to MaBoSS .bnd and .cfg files.

    Files are written to the session's artifact directory
    (<server>/artifacts/<session_id>/output.bnd and output.cfg).

    Args:
        bnet_path (str): Path to the .bnet file (absolute or relative to CWD).
        session_id:      Session to write files into (default: active session).

    Returns:
        str: Paths to the generated BND and CFG files, or an error message.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
    bnd_out = str(safe_artifact_path(art_dir, "output.bnd"))
    cfg_out = str(safe_artifact_path(art_dir, "output.cfg"))

    ctx.info(f"Processing .bnet file: {bnet_path} -> {bnd_out}, {cfg_out}")
    try:
        maboss.bnet_to_bnd_and_cfg(bnet_path, bnd_out, cfg_out)
    except Exception as e:
        ctx.error(f"bnet_to_bnd_and_cfg failed: {e}")
        return f"Error converting .bnet file: {e}"

    for path, label in [(bnd_out, "BND"), (cfg_out, "CFG")]:
        if not os.path.exists(path):
            return f"Error: expected {label} file was not created at {path}."

    ctx.info(f"BND and CFG files created: {bnd_out}, {cfg_out}")
    return (
        f"MaBoSS .bnd and .cfg files created successfully.\n"
        f"  BND: {bnd_out}\n"
        f"  CFG: {cfg_out}\n\n"
        f"Next: call build_simulation(session_id='{sess.session_id}') to load the simulation."
    )


@mcp.tool()
async def build_simulation(ctx: Context, session_id: str = None,
                           bnd_path: str = None, cfg_path: str = None) -> str:
    """Load a MaBoSS simulation from BND and CFG files.

    If bnd_path/cfg_path are omitted, the files generated by the last
    bnet_to_bnd_and_cfg call for this session are used automatically.

    Usage order (recommended):
        1. bnet_to_bnd_and_cfg -> build_simulation
        2. show_maboss_parameters (inspect defaults)
        3. update_maboss_parameters (reduce sample_count, set thread_count, etc.)
        4. set_maboss_output_nodes / set_maboss_initial_state (optional)
        5. run_simulation

    Args:
        session_id: Session to load the simulation into (default: active session).
        bnd_path:   Path to .bnd file (optional; uses session artifact if omitted).
        cfg_path:   Path to .cfg file (optional; uses session artifact if omitted).

    Returns:
        str: Simulation parameters or an error message.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)

    # Fall back to session artifact paths when not provided
    if bnd_path is None:
        bnd_path = str(art_dir / "output.bnd")
    if cfg_path is None:
        cfg_path = str(art_dir / "output.cfg")

    await ctx.info(f"Loading MaBoSS simulation: BND={bnd_path}  CFG={cfg_path}")
    await ctx.info(f"PATH={os.environ.get('PATH')}  CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}")
    await ctx.info(f"which MaBoSS: {shutil.which('MaBoSS')}")

    try:
        loaded_sim = maboss.load(bnd_path, cfg_path)
    except Exception as e:
        ctx.error(f"Failed to load simulation: {e}")
        return f"Error loading MaBoSS simulation: {e}"

    if loaded_sim:
        sess.set_simulation(loaded_sim, bnd_path, cfg_path)
        ctx.info("MaBoSS simulation loaded successfully.")
        parameters_str = "\n".join(f"{k}: {v}" for k, v in loaded_sim.param.items())
        return f"MaBoSS simulation loaded successfully.\n{parameters_str}"
    else:
        ctx.error("maboss.load returned None.")
        return "Error: maboss.load returned None. Check the BND and CFG files."

@mcp.tool()
def run_simulation(ctx: Context, session_id: str = None) -> str:
    """Run the loaded MaBoSS simulation.

    Tip: Tune performance first via update_maboss_parameters (e.g. sample_count,
    thread_count) before running large simulations.

    Args:
        session_id: Session to run (default: active session).

    Returns:
        str: Result status or an error message.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        ctx.info("Running MaBoSS simulation...")
        run_result = sess.sim.run()
        sess.set_result(run_result)
        ctx.info("MaBoSS simulation completed successfully.")
        return "MaBoSS simulation run completed successfully."
    except Exception as e:
        ctx.error(f"Error during MaBoSS simulation run: {str(e)}")
        return f"Error during MaBoSS simulation run: {str(e)}"

@mcp.tool()
def get_maboss_initial_state(ctx: Context, session_id: str = None) -> str:
    """Retrieve the initial state configuration of the MaBoSS simulation.

    Args:
        session_id: Session to query (default: active session).

    Returns:
        str: Initial state of the MaBoSS simulation.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        initial_state = sess.sim.get_initial_state()
        ctx.info(f"Initial state retrieved: {initial_state}")
        return f"Initial state of the MaBoSS simulation: {initial_state}"
    except Exception as e:
        ctx.error(f"Error retrieving initial state: {str(e)}")
        return f"Error retrieving initial state: {str(e)}"


@mcp.tool()
def get_maboss_logical_rules(ctx: Context, session_id: str = None) -> str:
    """Retrieve the logical (Boolean) rules of the MaBoSS network.

    Args:
        session_id: Session to query (default: active session).

    Returns:
        str: Logical rules of the MaBoSS simulation.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        logical_rules = sess.sim.get_logical_rules()
        ctx.info(f"Logical rules retrieved: {logical_rules}")
        return f"Logical rules of the MaBoSS simulation:\n{logical_rules}"
    except Exception as e:
        ctx.error(f"Error retrieving logical rules: {str(e)}")
        return f"Error retrieving logical rules: {str(e)}"

@mcp.tool()
def get_maboss_mutations(ctx: Context, session_id: str = None) -> str:
    """Retrieve the mutation settings applied to the MaBoSS network.

    Args:
        session_id: Session to query (default: active session).

    Returns:
        str: Mutations of the MaBoSS simulation.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        mutations = sess.sim.get_mutations()
        ctx.info(f"Mutations retrieved: {mutations}")
        return f"Mutations of the MaBoSS simulation:\n{mutations}"
    except Exception as e:
        ctx.error(f"Error retrieving mutations: {str(e)}")
        return f"Error retrieving mutations: {str(e)}"

@mcp.tool()
def update_maboss_parameters(ctx: Context, parameters: dict = None, session_id: str = None) -> str:
    """Update one or more MaBoSS simulation parameters.

    Call without arguments to list current parameters and usage hints.

    Common keys:
        sample_count   (int)   Number of trajectories (reduces stochastic noise; large = slower)
        max_time       (float) Simulation time horizon
        time_tick      (float) Time discretization step
        discrete_time  (int)   0/1 toggle for discrete time mode
        thread_count   (int)   Parallel threads for faster sampling (environment dependent)

    Args:
        parameters: dict of {name: value}. Omitted or empty -> show current values.
        session_id:  Session to update (default: active session).

    Returns:
        str: Confirmation or a table of current parameters.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    try:
        if not parameters:
            current = {k: v for k, v in sess.sim.param.items()}
            ctx.info("Listing current MaBoSS parameters")
            df = pd.DataFrame([[k, v] for k, v in current.items()], columns=["parameter", "value"])
            return "Current MaBoSS parameters (call update_maboss_parameters with a parameters dict to modify):\n" + df.to_markdown(index=False, tablefmt="plain")
        allowed = set(sess.sim.param.keys())
        unknown = [k for k in parameters.keys() if k not in allowed]
        if unknown:
            return ("Unsupported parameter(s): " + ", ".join(unknown) +
                    "\nUse update_maboss_parameters() with no args to list valid keys.")
        for key, value in parameters.items():
            sess.sim.param[key] = value
        ctx.info(f"MaBoSS parameters updated: {parameters}")
        summary = ", ".join(f"{k}={v}" for k, v in parameters.items())
        return f"Parameters updated: {summary}"
    except Exception as e:
        ctx.error(f"Error updating MaBoSS parameters: {str(e)}")
        return f"Error updating MaBoSS parameters: {str(e)}"

@mcp.tool()
def show_maboss_parameters(ctx: Context, session_id: str = None) -> str:
    """Show current MaBoSS simulation parameters (read-only helper).

    Args:
        session_id: Session to query (default: active session).
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."
    current = {k: v for k, v in sess.sim.param.items()}
    df = pd.DataFrame([[k, v] for k, v in current.items()], columns=["parameter", "value"])
    return df.to_markdown(index=False, tablefmt="plain")

@mcp.tool()
def get_maboss_help_json(ctx: Context) -> str:
    """Machine-readable help for MaBoSS tools (JSON string)."""
    tools = {
        "workflow": [
            "create_session (optional – auto-created on first use)",
            "bnet_to_bnd_and_cfg",
            "build_simulation",
            "show_maboss_parameters",
            "update_maboss_parameters",
            "set_maboss_output_nodes",
            "set_maboss_initial_state",
            "run_simulation",
            "get_simulation_result",
            "visualize_network_trajectories",
            "simulate_mutation",
            "list_generated_files",
        ],
        "session_tools": [
            "create_session", "list_sessions", "set_default_session", "delete_session"
        ],
        "notes": [
            "All file I/O is scoped to <server>/artifacts/<session_id>/.",
            "Call update_maboss_parameters with no arguments to list valid keys.",
            "Reduce sample_count and set thread_count early to speed iteration.",
            "Use set_maboss_output_nodes to limit outputs and reduce result size.",
            "Pass session_id explicitly to run multiple independent simulations in parallel."
        ],
        "key_parameters": {
            "sample_count": "Number of trajectories (runtime vs precision).",
            "thread_count": "Parallel threads (if backend supports).",
            "max_time": "Simulation horizon.",
            "time_tick": "Time discretization step.",
            "discrete_time": "0/1 toggle for discrete time mode."
        }
    }
    return json.dumps(tools, ensure_ascii=False)

@mcp.tool()
def set_maboss_output_nodes(ctx: Context, output_nodes: list, session_id: str = None) -> str:
    """Set the output nodes for the MaBoSS simulation.

    Args:
        output_nodes (list): Node names to mark as output nodes.
        session_id:          Session to update (default: active session).

    Returns:
        str: Confirmation message.
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        ctx.info(f"Former MaBoSS output nodes: {sess.sim.network.get_output()}")
        sess.sim.network.set_output(output_nodes)
        ctx.info(f"MaBoSS output nodes set: {sess.sim.network.get_output()}")
        return f"MaBoSS output nodes set successfully: {sess.sim.network.get_output()}"
    except Exception as e:
        ctx.error(f"Error setting MaBoSS output nodes: {str(e)}")
        return f"Error setting MaBoSS output nodes: {str(e)}"
    
@mcp.tool()
def set_maboss_initial_state(ctx: Context, nodes, probDict, session_id: str = None) -> str:
    """Set initial state probabilities for one or more nodes in the MaBoSS simulation.

    Parameters
    ----------
    nodes : str or list/tuple of str
        Node name or list of node names.
    probDict : list, dict, or nested dict
        - Single node: list [P(0), P(1)] or dict {0: P(0), 1: P(1)}.
        - Multiple nodes: dict mapping tuples of 0/1 to probabilities,
          e.g. {(0,0): 0.4, (1,0): 0.6}.
    session_id : str, optional
        Session to update (default: active session).

    Returns
    -------
    str
        Success or error message.

    Example
    -------
    >>> set_maboss_initial_state('node1', [0.3, 0.7])
    >>> set_maboss_initial_state(['node1', 'node2'], {(0, 0): 0.4, (1, 0): 0.6, (0, 1): 0})
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        if isinstance(nodes, str):
            node_arg = nodes
        elif isinstance(nodes, (list, tuple)):
            node_arg = list(nodes)
        else:
            return "Invalid type for 'nodes'. Must be str or list/tuple of str."

        if isinstance(node_arg, str):
            if not (isinstance(probDict, list) or isinstance(probDict, dict)):
                return "For a single node, probDict must be a list or dict."
        elif isinstance(node_arg, list):
            if not isinstance(probDict, dict):
                return "For multiple nodes, probDict must be a dict mapping tuples to probabilities."

        ctx.info(f"Former MaBoSS initial state: {sess.sim.network.get_istate()}")
        sess.sim.network.set_istate(node_arg, probDict)
        ctx.info(f"Current MaBoSS initial state: {sess.sim.network.get_istate()}")
        return f"Initial state set for MaBoSS simulation: {sess.sim.network.get_istate()}"
    except Exception as e:
        ctx.error(f"Error setting MaBoSS initial state: {str(e)}")
        return f"Error setting MaBoSS initial state: {str(e)}"
    
@mcp.tool()
def simulate_mutation(ctx: Context, nodes, state="OFF", session_id: str = None) -> str:
    """Simulate the effect of one or more node mutations on the MaBoSS network.

    Creates a copy of the current simulation, applies the mutation(s), runs it,
    and returns the final state probability distribution as a Markdown table.

    Parameters
    ----------
    nodes : str or list/tuple of str
        Node name(s) to mutate.
    state : str or list/tuple of str, optional
        Mutation state(s): 'ON', 'OFF', or 'WT' (default: 'OFF').
        If a single string, all nodes are mutated to this state.
        If a list/tuple, must match the length of ``nodes``.
    session_id : str, optional
        Session to use (default: active session).

    Returns
    -------
    str
        Markdown table of the final state probability trajectory, or an error message.

    Example
    -------
    >>> simulate_mutation('FoxO3', 'OFF')
    >>> simulate_mutation(['FoxO3', 'AKT'], ['OFF', 'ON'])
    """
    sess = ensure_session(session_id)
    if sess.sim is None:
        return "No MaBoSS simulation has been built yet. Call bnet_to_bnd_and_cfg then build_simulation first."

    try:
        ctx.info("Running MaBoSS simulation with mutation analysis...")
        mutated_simulation = sess.sim.copy()

        if isinstance(nodes, str):
            node_list = [nodes]
        else:
            node_list = list(nodes)

        if isinstance(state, str):
            state_list = [state] * len(node_list)
        else:
            state_list = list(state)
            if len(state_list) != len(node_list):
                return "Length of 'state' must match length of 'nodes'."

        valid_states = {"ON", "OFF", "WT"}
        for s in state_list:
            if s not in valid_states:
                return f"Invalid mutation state: {s}. Must be one of {valid_states}."

        for node, s in zip(node_list, state_list):
            mutated_simulation.mutate(node, s)
            ctx.info(f"Applied mutation: {node} -> {s}")

        mut_result = mutated_simulation.run()
        df_prob = mut_result.get_last_states_probtraj()

        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"

        df_prob = clean_for_markdown(df_prob)
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")

        md_lines = [
            "**MaBoSS Simulation: State Probability Trajectory (with Mutation)**",
            "",
            f"_Mutations applied: {dict(zip(node_list, state_list))}_",
            "",
            md_table
        ]
        return "\n".join(md_lines)
    except Exception as e:
        ctx.error(f"Error running MaBoSS simulation with mutation: {str(e)}")
        return f"Error running MaBoSS simulation with mutation: {str(e)}"
    
@mcp.tool()
def visualize_network_trajectories(ctx: Context, session_id: str = None) -> str:
    """Plot network trajectories from the last simulation run and save to an artifact file.

    Args:
        session_id: Session to use (default: active session).

    Returns:
        str: Path to the saved PNG plot, or an error message.
    """
    ctx.info("Request to visualize network trajectories received.")
    sess = ensure_session(session_id)
    if sess.result is None:
        return "No simulation has been run yet. Call run_simulation first."

    try:
        fig = sess.result.plot_trajectory()
        if fig is None:
            fig = plt.gcf()
        art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)
        output_path = str(safe_artifact_path(art_dir, "network_trajectory.png"))
        fig.savefig(output_path)
        plt.close(fig)
        ctx.info(f"Network trajectory plot saved to {output_path}")
        return f"Network trajectory plot saved: {output_path}\nYou can open it with your image viewer."
    except Exception as e:
        ctx.error(f"Error saving network trajectory plot: {str(e)}")
        return f"Error saving network trajectory plot: {str(e)}"
    
@mcp.tool()
def get_simulation_result(session_id: str = None) -> str:
    """Retrieve the last simulation result as a Markdown table of state probabilities.

    Output Format
    -------------
    A Markdown table where:
    - **Columns** = Distinct Boolean states (sets of ON nodes joined by '--').
    - **Row** = Final timepoint probability snapshot (values sum to ~1).

    Args:
        session_id: Session to query (default: active session).

    Returns:
        str: Markdown-formatted state probability trajectory, or an info message.
    """
    sess = ensure_session(session_id)

    if sess.result is None:
        return "_No simulation has been run yet. Call run_simulation first._"

    try:
        df_prob = sess.result.get_last_states_probtraj()

        if df_prob.empty:
            return "_Simulation completed but returned no trajectory data._"

        df_prob = clean_for_markdown(df_prob)
        md_table = df_prob.to_markdown(index=False, tablefmt="plain")

        md_lines = [
            "**MaBoSS Simulation: State Probability Trajectory**",
            "",
            "_Below is the probability trajectory of each state over time:_",
            "",
            md_table
        ]
        return "\n".join(md_lines)

    except Exception as e:
        return f"**Error retrieving simulation result:** {str(e)}"


@mcp.tool()
def list_generated_files(session_id: str = None) -> str:
    """List all artifact files generated for a session (or all sessions).

    Args:
        session_id: Session to list (default: active session).
                    Pass 'all' to list across every session.

    Returns:
        str: File listing, or a message if none exist.
    """
    if session_id == "all":
        files = list_artifacts(_SERVER_ROOT, session_id=None)
    else:
        sess = ensure_session(session_id)
        files = list_artifacts(_SERVER_ROOT, session_id=sess.session_id)

    if not files:
        return "No artifact files found."
    lines = [f"- {f}" for f in files]
    return "## Generated artifact files\n\n" + "\n".join(lines)


@mcp.tool()
def check_bnd_and_cfg_name(session_id: str = None) -> str:
    """List BND and CFG files in the session artifact directory.

    Args:
        session_id: Session to check (default: active session).

    Returns:
        str: Names of the BND and CFG files, or a message if none exist.
    """
    sess = ensure_session(session_id)
    art_dir = get_artifact_dir(_SERVER_ROOT, sess.session_id)

    bnd_files = [str(f) for f in art_dir.glob("*.bnd")]
    cfg_files = [str(f) for f in art_dir.glob("*.cfg")]

    if not bnd_files and not cfg_files:
        return f"No .bnd or .cfg files found in artifact directory: {art_dir}"

    file_list = []
    if bnd_files:
        file_list.append(f"BND files: {', '.join(bnd_files)}")
    if cfg_files:
        file_list.append(f"CFG files: {', '.join(cfg_files)}")
    return "\n".join(file_list)


@mcp.tool()
def clean_bnd_and_cfg(ctx: Context, session_id: str = None) -> str:
    """Remove all artifact files (BND, CFG, plots) for the given session.

    Args:
        session_id: Session to clean (default: active session).

    Returns:
        str: Confirmation message.
    """
    sess = ensure_session(session_id)
    try:
        count = clean_artifacts(_SERVER_ROOT, sess.session_id)
        ctx.info(f"Cleaned {count} artifact file(s) for session {sess.session_id}.")
        return f"Artifact files cleaned up for session {sess.session_id}: {count} file(s) removed."
    except Exception as e:
        ctx.error(f"Error during cleanup: {str(e)}")
        return f"Error during cleanup: {str(e)}"
    
    
def clean_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Convert every cell to a string.
    2. Strip leading/trailing whitespace and collapse any internal whitespace to a single space.
    3. Replace 'nan' or entirely-blank cells with an empty string.
    4. Drop columns and rows that end up completely empty.
    """
    # 1) Make sure everything is a string so strip/regex-replace works
    df_str = df.astype(str)

    # 2) Strip leading/trailing whitespace, then collapse any run of whitespace/newlines to a single space
    df_str = df_str.applymap(lambda val: " ".join(val.split()))

    # 3) Replace the literal string 'nan' (that pandas sometimes shows for NaNs) with an actual empty string
    df_str = df_str.replace("nan", "", regex=False)

    # 4) Drop any columns that are now entirely empty
    df_str = df_str.dropna(axis=1, how="all")  # drop cols where every entry is NaN (after replacement, NaN still possible)
    df_str = df_str.loc[:, (df_str != "").any(axis=0)]  # also drop columns that are all empty strings

    # 5) Drop any rows that are now entirely empty
    df_str = df_str.dropna(axis=0, how="all")
    df_str = df_str.loc[(df_str != "").any(axis=1), :]

    return df_str


if __name__ == "__main__":
    mcp.run()
