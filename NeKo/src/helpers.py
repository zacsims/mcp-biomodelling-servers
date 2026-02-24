"""NeKo server-side helpers.

Contains constants, private helper functions, the requires_network decorator,
and standalone utility functions that support server.py tools but are not
themselves MCP-exposed tools or resources.
"""

import os
import sys
import glob
import re
import tempfile
import requests
from pathlib import Path
from typing import Optional, List
from functools import wraps

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Allow imports from NeKo/ (utils, session_manager) and the repo root
# (artifact_manager) without requiring an installed package.
_NEKO_ROOT = Path(__file__).parent.parent   # .../NeKo/
_REPO_ROOT = _NEKO_ROOT.parent              # .../mcp-biomodelling-servers/

for _p in (_NEKO_ROOT, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from neko._outputs.exports import Exports   # noqa: E402 (post-path setup)
from utils import clean_for_markdown        # noqa: E402
from session_manager import ensure_session  # noqa: E402
from artifact_manager import get_artifact_dir  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

#: Canonical error token returned by any tool that requires a network but finds none.
E_NO_NET = "E_NO_NET: No network in session. Call create_session() then create_network()."

#: One-line hint appended to summary-verbosity responses.
SUMMARY_HINT = "Set verbosity='preview' or 'full' for more details."

#: Absolute path to the NeKo server directory; used for per-session artifact dirs.
_SERVER_ROOT: Path = _NEKO_ROOT

# ── Table helpers ──────────────────────────────────────────────────────────────

def _short_table(df: "pd.DataFrame", max_rows: int = 25):
    """Return a plain Markdown table string truncated to *max_rows*, plus a boolean flag.

    Returns:
        tuple[str, bool]: (markdown_string, was_truncated)
    """
    if df is None or df.empty:
        return "(no data)", True
    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True
    return clean_for_markdown(df).to_markdown(index=False, tablefmt="plain"), truncated

# ── Session / network accessors ────────────────────────────────────────────────

def _export_dir(session_id: Optional[str] = None) -> Path:
    """Return (and create) the per-session artifact directory for NeKo exports.

    Falls back to a shared ``exports/`` directory when *session_id* is ``None``
    (legacy behaviour; should not occur in normal usage).
    """
    if session_id:
        return get_artifact_dir(_SERVER_ROOT, session_id)
    # Legacy fallback
    d = _SERVER_ROOT / "exports"
    d.mkdir(exist_ok=True)
    return d


def _session_network(session_id: Optional[str]):
    """Return ``(sess, sess.network)`` for the given (or default) session."""
    sess = ensure_session(session_id)
    return sess, sess.network


def _invalidate(sess) -> None:
    """Invalidate the edge cache on *sess* after a mutation."""
    if sess:
        sess.invalidate_edges_cache()

# ── Decorator ─────────────────────────────────────────────────────────────────

def requires_network(fn):
    """Decorator that guards tools requiring an active network.

    Injects ``sess`` and ``network`` keyword arguments into the decorated
    function.  Returns :data:`E_NO_NET` immediately if no network exists in
    the current session.
    """
    @wraps(fn)
    def inner(*args, **kwargs):
        session_id = kwargs.get("session_id")
        sess, network = _session_network(session_id)
        if network is None:
            return E_NO_NET
        kwargs["sess"] = sess
        kwargs["network"] = network
        return fn(*args, **kwargs)
    return inner

# ── Graph helpers ──────────────────────────────────────────────────────────────

def _compute_components(network) -> List[List[str]]:
    """Compute connected components (including isolated nodes) via iterative DFS.

    Returns a list of node-lists sorted from largest to smallest component.
    """
    # Seed all known nodes so that isolates appear as singleton components.
    try:
        all_nodes = set(network.nodes["Uniprot"].tolist())
    except Exception:
        all_nodes = set()

    adj: dict = {str(n): set() for n in all_nodes if pd.notna(n) and n != ""}

    # Build undirected adjacency from the edge list.
    try:
        sources = network.edges["source"].tolist()
        targets = network.edges["target"].tolist()
        for s, t in zip(sources, targets):
            if pd.notna(s) and pd.notna(t) and s != "" and t != "":
                s_str, t_str = str(s), str(t)
                adj.setdefault(s_str, set()).add(t_str)
                adj.setdefault(t_str, set()).add(s_str)
    except Exception:
        pass

    if not adj:
        return []

    visited: set = set()
    comps: List[List[str]] = []

    for n in adj:
        if n in visited:
            continue
        stack = [n]
        cur: List[str] = []
        visited.add(n)
        while stack:
            node = stack.pop()
            cur.append(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(cur)

    comps.sort(key=len, reverse=True)
    return comps

# ── File utilities ─────────────────────────────────────────────────────────────

def download_signor_database() -> str:
    """Download SIGNOR human interaction data (TSV) from the SIGNOR API and save locally."""
    url = "https://signor.uniroma2.it/API/getHumanData.php"
    try:
        r = requests.get(url)
        r.raise_for_status()
        output_file = "SIGNOR_Human.tsv"
        with open(output_file, "wb") as f:
            f.write(r.content)
        return "SIGNOR database downloaded successfully."
    except requests.RequestException as e:
        return f"Error downloading SIGNOR database: {str(e)}"


def clean_bnet_headers(folder_path: str = ".") -> str:
    """Remove boilerplate headers from every ``.bnet`` file in *folder_path*.

    Strips the first two lines when they match::

        # model in BoolNet format
        targets, factors
    """
    cleaned_files = []
    for file_path in glob.glob(os.path.join(folder_path, "*.bnet")):
        with open(file_path, "r") as fh:
            lines = fh.readlines()
        if (
            len(lines) >= 2
            and lines[0].strip() == "# model in BoolNet format"
            and lines[1].strip() == "targets, factors"
        ):
            with open(file_path, "w") as fh:
                fh.writelines(lines[2:])
            cleaned_files.append(os.path.basename(file_path))

    if cleaned_files:
        return f"Cleaned headers from: {', '.join(cleaned_files)}"
    return "No .bnet files needed cleaning."

def _get_translators(network):
    """Builds fast, two-way translation dictionaries for the network."""
    try:
        # Extract the node registry dataframe
        df = network.nodes
        
        # Ensure we only map valid rows
        clean_df = df.dropna(subset=['Uniprot', 'Genesymbol'])
        
        # Build the dictionaries
        uniprot_to_symbol = dict(zip(clean_df['Uniprot'], clean_df['Genesymbol']))
        symbol_to_uniprot = dict(zip(clean_df['Genesymbol'], clean_df['Uniprot']))
        
        return uniprot_to_symbol, symbol_to_uniprot
    except Exception as e:
        # Fallback to empty dicts if columns are missing
        return {}, {}