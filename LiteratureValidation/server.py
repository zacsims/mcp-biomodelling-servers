# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastmcp>=2.14.4",
#   "paper-qa>=5",
#   "httpx>=0.27",
# ]
# ///
"""
Literature Validation MCP Server

Validates PhysiCell cell rules against published biomedical literature
using PaperQA2 for RAG-based question answering.

The LLM orchestrates between this server, the PhysiCell MCP server,
and a PubMed MCP server to:
1. Extract rules from PhysiCell sessions
2. Search PubMed for relevant papers
3. Index papers and validate rules against literature
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

# PaperQA imports
try:
    from paperqa import Docs, Settings
    from paperqa.settings import AnswerSettings, ParsingSettings
    PAPERQA_AVAILABLE = True
except ImportError:
    PAPERQA_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

COLLECTIONS_DIR = Path(os.environ.get(
    "LITERATURE_VALIDATION_DIR",
    str(Path.home() / "Documents" / "LiteratureValidation")
))

# ============================================================================
# IN-MEMORY STATE
# ============================================================================

# Named collections: name -> Docs instance
_collections: dict[str, Any] = {}

# Validation results per collection: name -> list of result dicts
_validation_results: dict[str, list[dict]] = {}

# ============================================================================
# HELPERS
# ============================================================================

def _get_paperqa_settings() -> "Settings":
    """Build PaperQA Settings using Claude via litellm."""
    return Settings(
        llm="litellm/claude-sonnet-4-20250514",
        summary_llm="litellm/claude-haiku-4-5-20251001",
        embedding="litellm/text-embedding-3-small",
        answer=AnswerSettings(
            evidence_k=10,
            answer_max_sources=5,
        ),
        parsing=ParsingSettings(
            chunk_size=3000,
            overlap=300,
        ),
    )


def _collection_dir(name: str) -> Path:
    """Get directory for a named collection."""
    return COLLECTIONS_DIR / name


def _papers_dir(name: str) -> Path:
    """Get papers subdirectory for a named collection."""
    return _collection_dir(name) / "papers"


async def _get_or_create_docs(name: str) -> "Docs":
    """Get existing Docs instance or create a new one for the collection."""
    if name in _collections:
        return _collections[name]

    if not PAPERQA_AVAILABLE:
        raise RuntimeError(
            "paper-qa is not installed. Install with: pip install 'paper-qa>=5'"
        )

    settings = _get_paperqa_settings()
    docs = Docs()
    _collections[name] = docs
    return docs


def _build_validation_question(cell_type: str, signal: str, direction: str,
                                behavior: str, half_max: float | None = None,
                                hill_power: float | None = None) -> str:
    """Build a focused question for PaperQA to answer about a cell rule."""
    dir_word = "increase" if direction == "increases" else "decrease"
    question = (
        f"Does {signal} {dir_word} {behavior} in {cell_type} cells? "
        f"What is the experimental evidence for this relationship? "
        f"If quantitative data exists, what signal concentration causes "
        f"a half-maximal response (EC50/half-max)? "
        f"How switch-like (Hill coefficient) is the response?"
    )
    if half_max is not None:
        question += (
            f" The proposed model uses a half-max of {half_max}. "
            f"Is this consistent with published data?"
        )
    if hill_power is not None:
        question += (
            f" The proposed Hill coefficient is {hill_power}. "
            f"Is this consistent with published data?"
        )
    return question


def _parse_support_level(answer_text: str) -> str:
    """Parse PaperQA answer to determine support level."""
    lower = answer_text.lower()

    strong_positive = [
        "well established", "well-established", "strongly supported",
        "extensive evidence", "clearly demonstrates", "well documented",
        "well-documented", "confirmed by multiple", "robust evidence",
    ]
    moderate_positive = [
        "evidence suggests", "supported by", "consistent with",
        "studies show", "has been reported", "has been observed",
        "indicates that", "demonstrated that",
    ]
    weak_signals = [
        "limited evidence", "some evidence", "preliminary",
        "few studies", "not well studied", "indirect evidence",
        "may", "might", "could potentially",
    ]
    contradictory_signals = [
        "contradictory", "conflicting", "inconsistent",
        "some studies show the opposite", "debated",
    ]
    unsupported_signals = [
        "no evidence", "not supported", "no published",
        "insufficient information", "cannot determine",
        "i cannot answer",
    ]

    for phrase in unsupported_signals:
        if phrase in lower:
            return "unsupported"
    for phrase in contradictory_signals:
        if phrase in lower:
            return "contradictory"
    for phrase in strong_positive:
        if phrase in lower:
            return "strong"
    for phrase in moderate_positive:
        if phrase in lower:
            return "moderate"
    for phrase in weak_signals:
        if phrase in lower:
            return "weak"

    return "moderate"


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP(
    "LiteratureValidation",
    instructions=(
        "Literature validation server for PhysiCell cell rules. "
        "Uses PaperQA2 to validate rules against published biomedical literature. "
        "Workflow: create_paper_collection → add_papers_to_collection → "
        "validate_rule/validate_rules_batch → get_validation_summary."
    ),
)


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
def create_paper_collection(name: str) -> str:
    """
    Create a named paper collection for literature validation.

    Each collection is an independent PaperQA document store.
    Use one collection per validation session or biological topic.

    Args:
        name: Collection name (e.g., "hypoxia_migration", "tumor_immune")

    Returns:
        str: Success message with collection details
    """
    if not PAPERQA_AVAILABLE:
        return (
            "**Error:** paper-qa is not installed.\n\n"
            "Install with: `pip install 'paper-qa>=5'`\n"
            "The ANTHROPIC_API_KEY environment variable must also be set."
        )

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "**Error:** ANTHROPIC_API_KEY environment variable is not set.\n\n"
            "PaperQA uses Claude via litellm for question answering. "
            "Set your Anthropic API key before using this server."
        )

    name = name.strip().lower().replace(" ", "_")
    if not name:
        return "**Error:** Collection name cannot be empty."

    # Create directory structure
    papers_path = _papers_dir(name)
    papers_path.mkdir(parents=True, exist_ok=True)

    # Initialize empty Docs instance
    _collections[name] = None  # Lazy init on first use
    _validation_results[name] = []

    return (
        f"## Collection Created: `{name}`\n\n"
        f"**Papers directory:** `{papers_path}`\n"
        f"**Status:** Empty — add papers with `add_papers_to_collection()`\n\n"
        f"**Next step:** Use the PubMed MCP server to search for relevant papers, "
        f"then add their abstracts here."
    )


@mcp.tool()
async def add_papers_to_collection(
    name: str,
    papers: list[dict[str, str]],
) -> str:
    """
    Add papers (abstracts or full text) to a collection for indexing by PaperQA.

    Each paper dict should have:
    - "title": Paper title
    - "text": Abstract or full text content
    - "pmid" (optional): PubMed ID
    - "doi" (optional): DOI
    - "authors" (optional): Author list string
    - "year" (optional): Publication year

    Args:
        name: Collection name (must exist)
        papers: List of paper dicts with at minimum "title" and "text"

    Returns:
        str: Summary of papers added and indexed
    """
    if not PAPERQA_AVAILABLE:
        return "**Error:** paper-qa is not installed."

    name = name.strip().lower().replace(" ", "_")
    papers_path = _papers_dir(name)
    if not papers_path.exists():
        return f"**Error:** Collection `{name}` does not exist. Create it first with `create_paper_collection()`."

    if not papers:
        return "**Error:** No papers provided."

    # Get or create Docs instance
    try:
        docs = await _get_or_create_docs(name)
    except RuntimeError as e:
        return f"**Error:** {e}"

    added_count = 0
    errors = []

    for paper in papers:
        title = paper.get("title", "").strip()
        text = paper.get("text", "").strip()

        if not title or not text:
            errors.append(f"Skipped paper with missing title or text")
            continue

        # Build metadata header
        meta_lines = [f"Title: {title}"]
        if paper.get("authors"):
            meta_lines.append(f"Authors: {paper['authors']}")
        if paper.get("year"):
            meta_lines.append(f"Year: {paper['year']}")
        if paper.get("pmid"):
            meta_lines.append(f"PMID: {paper['pmid']}")
        if paper.get("doi"):
            meta_lines.append(f"DOI: {paper['doi']}")
        meta_lines.append("")  # blank line before content
        meta_lines.append(text)

        full_text = "\n".join(meta_lines)

        # Write paper to file for PaperQA
        safe_name = hashlib.md5(title.encode()).hexdigest()[:12]
        paper_file = papers_path / f"{safe_name}.txt"
        paper_file.write_text(full_text, encoding="utf-8")

        # Add to PaperQA Docs
        try:
            await docs.aadd(
                paper_file,
                citation=title,
                docname=title[:80],
            )
            added_count += 1
        except Exception as e:
            errors.append(f"Failed to index '{title[:40]}...': {e}")

    result = f"## Papers Added to `{name}`\n\n"
    result += f"**Indexed:** {added_count} / {len(papers)} papers\n"
    result += f"**Papers directory:** `{papers_path}`\n"

    if errors:
        result += f"\n**Warnings:**\n"
        for err in errors[:5]:
            result += f"- {err}\n"
        if len(errors) > 5:
            result += f"- ... and {len(errors) - 5} more\n"

    result += (
        f"\n**Next step:** Use `validate_rule()` or `validate_rules_batch()` "
        f"to check rules against this literature."
    )
    return result


@mcp.tool()
async def validate_rule(
    name: str,
    cell_type: str,
    signal: str,
    direction: str,
    behavior: str,
    half_max: float | None = None,
    hill_power: float | None = None,
    min_signal: float | None = None,
    max_signal: float | None = None,
) -> str:
    """
    Validate a single cell rule against literature in the collection.

    Asks PaperQA whether the proposed signal-behavior relationship is
    supported by published evidence, and checks parameter plausibility.

    Args:
        name: Collection name with indexed papers
        cell_type: Cell type name (e.g., "cancer", "macrophage")
        signal: Signal name (e.g., "oxygen", "TNF")
        direction: 'increases' or 'decreases'
        behavior: Behavior name (e.g., "migration_speed", "apoptosis")
        half_max: Proposed half-maximum signal level (optional)
        hill_power: Proposed Hill coefficient (optional)
        min_signal: Minimum signal value (optional)
        max_signal: Maximum signal value (optional)

    Returns:
        str: Validation result with support level, evidence summary, and citations
    """
    if not PAPERQA_AVAILABLE:
        return "**Error:** paper-qa is not installed."

    name = name.strip().lower().replace(" ", "_")
    if name not in _collections and not _papers_dir(name).exists():
        return f"**Error:** Collection `{name}` does not exist."

    if direction not in ("increases", "decreases"):
        return "**Error:** direction must be 'increases' or 'decreases'."

    try:
        docs = await _get_or_create_docs(name)
    except RuntimeError as e:
        return f"**Error:** {e}"

    # Check that the collection has papers
    papers_path = _papers_dir(name)
    paper_files = list(papers_path.glob("*.txt"))
    if not paper_files:
        return (
            f"**Error:** Collection `{name}` has no papers indexed.\n"
            f"Add papers with `add_papers_to_collection()` first."
        )

    # Build and ask the question
    question = _build_validation_question(
        cell_type, signal, direction, behavior, half_max, hill_power
    )

    try:
        response = await docs.aquery(question)
        answer_text = response.answer
        references = response.references
    except Exception as e:
        return f"**Error querying PaperQA:** {e}"

    # Determine support level
    support_level = _parse_support_level(answer_text)

    # Store result
    validation = {
        "cell_type": cell_type,
        "signal": signal,
        "direction": direction,
        "behavior": behavior,
        "half_max": half_max,
        "hill_power": hill_power,
        "support_level": support_level,
        "evidence_summary": answer_text,
        "references": references,
    }
    if name not in _validation_results:
        _validation_results[name] = []
    _validation_results[name].append(validation)

    # Format result
    dir_arrow = "↑" if direction == "increases" else "↓"
    support_emoji = {
        "strong": "**STRONG**",
        "moderate": "**MODERATE**",
        "weak": "**WEAK**",
        "contradictory": "**CONTRADICTORY**",
        "unsupported": "**UNSUPPORTED**",
    }

    result = f"## Rule Validation: {cell_type} | {signal} {dir_arrow} {behavior}\n\n"
    result += f"**Support Level:** {support_emoji.get(support_level, support_level)}\n\n"
    result += f"### Evidence Summary\n{answer_text}\n\n"

    if references:
        result += f"### Key References\n{references}\n\n"

    if half_max is not None:
        result += f"**Proposed half-max:** {half_max}\n"
    if hill_power is not None:
        result += f"**Proposed Hill power:** {hill_power}\n"

    return result


@mcp.tool()
async def validate_rules_batch(
    name: str,
    rules: list[dict[str, Any]],
) -> str:
    """
    Validate multiple cell rules against literature in a single call.

    Each rule dict should contain: cell_type, signal, direction, behavior.
    Optional: half_max, hill_power, min_signal, max_signal.

    Args:
        name: Collection name with indexed papers
        rules: List of rule dicts to validate

    Returns:
        str: Combined validation results for all rules
    """
    if not PAPERQA_AVAILABLE:
        return "**Error:** paper-qa is not installed."

    if not rules:
        return "**Error:** No rules provided."

    results = []
    for i, rule in enumerate(rules):
        cell_type = rule.get("cell_type", "")
        signal = rule.get("signal", "")
        direction = rule.get("direction", "")
        behavior = rule.get("behavior", "")

        if not all([cell_type, signal, direction, behavior]):
            results.append(
                f"**Rule {i+1}:** Skipped — missing required fields "
                f"(need cell_type, signal, direction, behavior)"
            )
            continue

        result = await validate_rule(
            name=name,
            cell_type=cell_type,
            signal=signal,
            direction=direction,
            behavior=behavior,
            half_max=rule.get("half_max"),
            hill_power=rule.get("hill_power"),
            min_signal=rule.get("min_signal"),
            max_signal=rule.get("max_signal"),
        )
        results.append(result)

    header = f"## Batch Validation Results ({len(rules)} rules)\n\n"
    return header + "\n---\n\n".join(results)


@mcp.tool()
def get_validation_summary(name: str) -> str:
    """
    Get a summary of all validation results for a collection.

    Shows support level distribution and highlights any unsupported
    or contradictory rules that may need attention.

    Args:
        name: Collection name

    Returns:
        str: Markdown summary of validation results
    """
    name = name.strip().lower().replace(" ", "_")

    validations = _validation_results.get(name, [])
    if not validations:
        return (
            f"**No validation results** for collection `{name}`.\n\n"
            f"Use `validate_rule()` or `validate_rules_batch()` first."
        )

    # Count by support level
    level_counts: dict[str, int] = {}
    for v in validations:
        level = v.get("support_level", "unknown")
        level_counts[level] = level_counts.get(level, 0) + 1

    result = f"## Validation Summary: `{name}`\n\n"
    result += f"**Total rules validated:** {len(validations)}\n\n"

    result += "### Support Level Distribution\n"
    for level in ["strong", "moderate", "weak", "contradictory", "unsupported"]:
        count = level_counts.get(level, 0)
        if count > 0:
            bar = "=" * count
            result += f"- **{level.capitalize()}:** {count} {bar}\n"

    # Flag rules needing attention
    flagged = [v for v in validations if v["support_level"] in ("unsupported", "contradictory", "weak")]
    if flagged:
        result += "\n### Rules Needing Attention\n"
        for v in flagged:
            dir_arrow = ">" if v["direction"] == "increases" else "v"
            result += (
                f"- **{v['support_level'].upper()}**: "
                f"{v['cell_type']} | {v['signal']} {dir_arrow} {v['behavior']}\n"
            )

    # Well-supported rules
    supported = [v for v in validations if v["support_level"] in ("strong", "moderate")]
    if supported:
        result += "\n### Well-Supported Rules\n"
        for v in supported:
            dir_arrow = ">" if v["direction"] == "increases" else "v"
            result += (
                f"- **{v['support_level'].upper()}**: "
                f"{v['cell_type']} | {v['signal']} {dir_arrow} {v['behavior']}\n"
            )

    result += (
        f"\n**Next step:** Use `store_validation_results()` in PhysiCell MCP "
        f"to persist these results in your simulation session."
    )
    return result


@mcp.tool()
def suggest_search_queries(
    cell_type: str,
    signal: str,
    direction: str,
    behavior: str,
) -> str:
    """
    Generate optimized PubMed search queries for validating a cell rule.

    Produces multiple query variants to maximize recall from PubMed
    literature search.

    Args:
        cell_type: Cell type name (e.g., "cancer", "macrophage")
        signal: Signal name (e.g., "oxygen", "TNF")
        direction: 'increases' or 'decreases'
        behavior: Behavior name (e.g., "migration_speed", "apoptosis")

    Returns:
        str: List of suggested PubMed queries
    """
    # Build synonym maps for common PhysiCell terms
    signal_synonyms: dict[str, list[str]] = {
        "oxygen": ["oxygen", "hypoxia", "O2", "pO2", "normoxia"],
        "glucose": ["glucose", "glycolysis", "nutrient deprivation"],
        "pressure": ["mechanical pressure", "cell density", "contact inhibition", "compression"],
        "TNF": ["TNF", "TNF-alpha", "tumor necrosis factor", "TNFα"],
        "IFN-gamma": ["IFN-gamma", "interferon gamma", "IFNγ", "IFN-γ"],
        "TGF-beta": ["TGF-beta", "TGFβ", "TGF-β", "transforming growth factor beta"],
        "VEGF": ["VEGF", "vascular endothelial growth factor"],
    }
    behavior_synonyms: dict[str, list[str]] = {
        "migration_speed": ["migration", "motility", "cell migration speed", "invasion"],
        "cycle_entry": ["proliferation", "cell cycle", "cell division", "growth rate"],
        "apoptosis": ["apoptosis", "programmed cell death", "cell death"],
        "necrosis": ["necrosis", "necrotic cell death"],
        "chemotaxis": ["chemotaxis", "directed migration", "chemotactic"],
        "adhesion": ["cell adhesion", "cell-cell adhesion", "attachment"],
        "secretion": ["secretion", "cytokine release", "paracrine signaling"],
    }
    cell_type_synonyms: dict[str, list[str]] = {
        "cancer": ["cancer cells", "tumor cells", "carcinoma", "malignant cells"],
        "tumor": ["tumor cells", "cancer cells", "neoplastic cells"],
        "macrophage": ["macrophage", "macrophages", "tumor-associated macrophages", "TAM"],
        "T cell": ["T cell", "T lymphocyte", "T cells", "CD8+ T cell"],
        "fibroblast": ["fibroblast", "cancer-associated fibroblast", "CAF", "stromal cells"],
        "endothelial": ["endothelial cell", "vascular endothelial", "endothelium"],
        "epithelial": ["epithelial cell", "epithelium"],
    }

    # Get synonyms or use the term itself
    sig_terms = signal_synonyms.get(signal, [signal])
    beh_terms = behavior_synonyms.get(behavior, [behavior.replace("_", " ")])
    cell_terms = cell_type_synonyms.get(cell_type, [cell_type])

    dir_word = "increases" if direction == "increases" else "decreases"

    queries = []

    # Primary query: most specific
    queries.append(
        f"{cell_terms[0]} {sig_terms[0]} {dir_word} {beh_terms[0]}"
    )

    # Broader queries with OR groups
    sig_or = " OR ".join(f'"{t}"' for t in sig_terms[:3])
    beh_or = " OR ".join(f'"{t}"' for t in beh_terms[:3])
    cell_or = " OR ".join(f'"{t}"' for t in cell_terms[:3])
    queries.append(f"({cell_or}) AND ({sig_or}) AND ({beh_or})")

    # MeSH-style query
    queries.append(
        f'"{sig_terms[0]}" AND "{beh_terms[0]}" AND ({cell_or})'
    )

    # Quantitative query for parameter values
    queries.append(
        f'"{sig_terms[0]}" AND "{beh_terms[0]}" AND '
        f'(EC50 OR "half-maximal" OR "dose-response" OR "Hill coefficient")'
    )

    # Review articles for overview
    queries.append(
        f"({sig_or}) AND ({beh_or}) AND review[pt]"
    )

    result = f"## Suggested PubMed Queries\n\n"
    result += f"**Rule:** {cell_type} | {signal} {direction} {behavior}\n\n"
    for i, q in enumerate(queries, 1):
        label = ["Primary", "Broad (OR groups)", "MeSH-style", "Quantitative", "Reviews"][i-1]
        result += f"### {i}. {label}\n```\n{q}\n```\n\n"

    result += (
        "**Usage:** Pass these queries to the PubMed MCP server's "
        "`search_pubmed()` tool, then add the returned abstracts to "
        "this collection with `add_papers_to_collection()`."
    )
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mcp.run()
