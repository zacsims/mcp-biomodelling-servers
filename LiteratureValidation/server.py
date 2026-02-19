# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastmcp>=2.14.4",
#   "edison-client",
#   "httpx>=0.27",
# ]
# ///
"""
Literature Validation MCP Server

Validates PhysiCell cell rules against published biomedical literature
using Edison Scientific's PaperQA3 API, which searches 150M+ papers
automatically — no manual paper curation needed.

The LLM orchestrates between this server and the PhysiCell MCP server to:
1. Extract rules from PhysiCell sessions
2. Validate rules against literature via Edison API
3. Store results back in the PhysiCell session
"""

import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Edison client imports
try:
    from edison_client import EdisonClient, JobNames, TaskRequest
    EDISON_AVAILABLE = True
except ImportError:
    EDISON_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

COLLECTIONS_DIR = Path(os.environ.get(
    "LITERATURE_VALIDATION_DIR",
    str(Path.home() / "Documents" / "LiteratureValidation")
))

# Global Edison client (initialized lazily)
_edison_client: "EdisonClient | None" = None


def _get_edison_client() -> "EdisonClient":
    """Get or create the global Edison client."""
    global _edison_client
    if _edison_client is None:
        api_key = os.environ.get("EDISON_PLATFORM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EDISON_PLATFORM_API_KEY environment variable is not set. "
                "Get your API key from https://platform.edisonscientific.com/profile"
            )
        _edison_client = EdisonClient(api_key=api_key)
    return _edison_client


# ============================================================================
# IN-MEMORY STATE
# ============================================================================

# Validation results per collection: name -> list of result dicts
_validation_results: dict[str, list[dict]] = {}

# ============================================================================
# HELPERS
# ============================================================================


def _collection_dir(name: str) -> Path:
    """Get directory for a named collection."""
    return COLLECTIONS_DIR / name


_SIGNAL_UNITS: dict[str, str] = {
    "oxygen": "mmHg",
    "glucose": "mM",
    "pressure": "dimensionless (0-1)",
    "TNF": "pg/mL",
    "IFN-gamma": "pg/mL",
    "TGF-beta": "ng/mL",
    "VEGF": "ng/mL",
}


def _build_validation_question(cell_type: str, signal: str, direction: str,
                                behavior: str, half_max: float | None = None,
                                hill_power: float | None = None,
                                signal_units: str | None = None) -> str:
    """Build a direction-agnostic question for PaperQA to answer about a cell rule.

    The question does NOT state the proposed direction, so PaperQA independently
    determines the direction from literature. This prevents leading-question bias.
    """
    # Auto-detect units if not provided
    if signal_units is None:
        signal_units = _SIGNAL_UNITS.get(signal.lower())

    units_note = f" (measured in {signal_units})" if signal_units else ""
    question = (
        f"What is the effect of {signal}{units_note} on {behavior} in {cell_type} cells? "
        f"Does increasing {signal} concentration lead to increased or decreased {behavior}? "
        f"What is the experimental evidence for this relationship? "
        f"If quantitative data exists, what signal concentration causes "
        f"a half-maximal response (EC50/half-max)? "
        f"How switch-like (Hill coefficient) is the response?"
    )
    if half_max is not None:
        units_str = f" {signal_units}" if signal_units else ""
        question += (
            f" For reference, a proposed model uses a half-max of {half_max}{units_str}. "
            f"Is this consistent with published data?"
        )
    if hill_power is not None:
        question += (
            f" For reference, a proposed Hill coefficient is {hill_power}. "
            f"Is this consistent with published data?"
        )
    question += (
        "\n\nConclude your answer with BOTH of the following verdict lines:\n\n"
        "First, state the direction of the relationship:\n"
        "DIRECTION: INCREASES — increasing signal concentration increases the behavior\n"
        "DIRECTION: DECREASES — increasing signal concentration decreases the behavior\n"
        "DIRECTION: AMBIGUOUS — evidence is mixed or insufficient to determine direction\n\n"
        "Then, state the strength of evidence:\n"
        "VERDICT: STRONG — extensive quantitative experimental support for the relationship\n"
        "VERDICT: MODERATE — qualitative evidence supports it but quantitative parameterization lacking\n"
        "VERDICT: WEAK — limited or indirect evidence only\n"
        "VERDICT: UNSUPPORTED — no relevant experimental evidence found"
    )
    return question


def _parse_support_level(answer_text: str) -> str:
    """Extract VERDICT classification from PaperQA answer text.

    Uses the LAST match because the answer file contains the prompt template
    (with all VERDICT options listed) followed by PaperQA's actual answer.
    re.search() would match the first template option, not the real verdict.

    CONTRADICTORY is no longer a valid VERDICT — directional contradictions
    are caught by the DIRECTION check instead. If PaperQA still writes
    CONTRADICTORY, map it to 'weak' (evidence exists but contradicts).
    """
    matches = re.findall(r"VERDICT:\s*(STRONG|MODERATE|WEAK|CONTRADICTORY|UNSUPPORTED)", answer_text, re.IGNORECASE)
    if matches:
        level = matches[-1].lower()  # Last match = PaperQA's actual verdict
        if level == "contradictory":
            return "weak"
        return level
    return "unsupported"


def _parse_direction(answer_text: str) -> str:
    """Extract DIRECTION from PaperQA answer.

    Uses the LAST match because the answer file contains the prompt template
    (with all DIRECTION options listed) followed by PaperQA's actual answer.
    re.search() would match the first template option, not the real direction.

    Returns 'increases', 'decreases', or 'ambiguous'.
    """
    matches = re.findall(r"DIRECTION:\s*(INCREASES|DECREASES|AMBIGUOUS)", answer_text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()  # Last match = PaperQA's actual direction
    return "ambiguous"


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP(
    "LiteratureValidation",
    instructions=(
        "Literature validation server for PhysiCell cell rules. "
        "Uses Edison Scientific's PaperQA3 API to validate rules against "
        "150M+ published papers. "
        "Workflow: validate_rule/validate_rules_batch → get_validation_summary. "
        "No paper collection management needed — Edison searches automatically."
    ),
)


# ============================================================================
# TOOLS
# ============================================================================

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
    signal_units: str | None = None,
) -> str:
    """
    Validate a single cell rule against literature using Edison PaperQA3.

    Edison automatically searches 150M+ papers — no paper collection
    management needed. Just provide the rule details and get an
    evidence-based verdict.

    Args:
        name: Collection name for organizing results (e.g., "tumor_model_v1")
        cell_type: Cell type name (e.g., "cancer", "macrophage")
        signal: Signal name (e.g., "oxygen", "TNF")
        direction: 'increases' or 'decreases'
        behavior: Behavior name (e.g., "migration_speed", "apoptosis")
        half_max: Proposed half-maximum signal level (optional)
        hill_power: Proposed Hill coefficient (optional)
        min_signal: Minimum signal value (optional)
        max_signal: Maximum signal value (optional)
        signal_units: Units for signal values (e.g., "mmHg", "mM"). Auto-detected
                      for common signals (oxygen=mmHg, glucose=mM) if not provided.

    Returns:
        str: Validation result with support level, evidence summary, and citations
    """
    if not EDISON_AVAILABLE:
        return (
            "**Error:** edison-client is not installed.\n\n"
            "Install with: `pip install edison-client`"
        )

    # Check for API key
    if not os.environ.get("EDISON_PLATFORM_API_KEY"):
        return (
            "**Error:** EDISON_PLATFORM_API_KEY environment variable is not set.\n\n"
            "Get your API key from https://platform.edisonscientific.com/profile "
            "and set it in the MCP server env config."
        )

    if direction not in ("increases", "decreases"):
        return "**Error:** direction must be 'increases' or 'decreases'."

    name = name.strip().lower().replace(" ", "_")

    # Ensure answers directory exists
    answers_dir = _collection_dir(name) / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached answer file (direction-agnostic key)
    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{behavior}")
    answer_file = answers_dir / f"{safe_key}.md"
    cached = False

    if answer_file.exists():
        # Reuse existing answer file instead of calling Edison again
        cached = True
        formatted_answer = answer_file.read_text(encoding="utf-8")
        answer_text = formatted_answer
        has_answer = True
    else:
        # Build direction-agnostic question
        question = _build_validation_question(
            cell_type, signal, direction, behavior, half_max, hill_power, signal_units
        )

        # Query Edison API
        try:
            client = _get_edison_client()
            task_data = TaskRequest(
                name=JobNames.LITERATURE,
                query=question,
            )
            responses = await client.arun_tasks_until_done(task_data)
            response = responses[0]

            answer_text = response.answer or ""
            formatted_answer = response.formatted_answer or answer_text
            has_answer = response.has_successful_answer
        except Exception as e:
            return f"**Error querying Edison API:** {e}"

        # Save full answer to output file
        answer_file.write_text(formatted_answer, encoding="utf-8")

    # Determine support level and literature direction
    # Try answer_text first, fall back to formatted_answer
    support_level = _parse_support_level(answer_text)
    if support_level == "unsupported":
        alt = _parse_support_level(formatted_answer)
        if alt != "unsupported":
            support_level = alt

    literature_direction = _parse_direction(answer_text)
    if literature_direction == "ambiguous":
        alt = _parse_direction(formatted_answer)
        if alt != "ambiguous":
            literature_direction = alt

    # Compare literature direction against proposed direction
    if literature_direction == "ambiguous":
        direction_match = None
    elif literature_direction == direction:
        direction_match = True
    else:
        direction_match = False

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
        "answer_file": str(answer_file),
        "literature_direction": literature_direction,
        "direction_match": direction_match,
    }
    if name not in _validation_results:
        _validation_results[name] = []
    _validation_results[name].append(validation)

    # Format result
    dir_arrow = "\u2191" if direction == "increases" else "\u2193"
    support_labels = {
        "strong": "**STRONG**",
        "moderate": "**MODERATE**",
        "weak": "**WEAK**",
        "unsupported": "**UNSUPPORTED**",
    }

    result = f"## Rule Validation: {cell_type} | {signal} {dir_arrow} {behavior}\n\n"

    if cached:
        result += "**Source:** Cached from previous validation (no Edison API call)\n\n"

    # Show direction mismatch warning prominently
    if direction_match is False:
        result += (
            f"### DIRECTION MISMATCH\n\n"
            f"Literature says {signal} **{literature_direction}** {behavior}, "
            f"but rule proposes {signal} **{direction}** {behavior}.\n\n"
        )

    result += f"**Support Level:** {support_labels.get(support_level, support_level)}\n"
    result += f"**Literature Direction:** {literature_direction}\n"
    if direction_match is True:
        result += f"**Direction:** Confirmed by literature\n"
    elif direction_match is False:
        result += f"**Direction:** MISMATCH \u2014 proposed '{direction}', literature says '{literature_direction}'\n"
    else:
        result += f"**Direction:** Could not be determined from literature\n"

    if not has_answer:
        result += f"**Note:** Edison could not find a definitive answer for this query\n"
    result += "\n"

    result += f"### Evidence Summary\n{answer_text}\n\n"

    if formatted_answer and formatted_answer != answer_text:
        result += f"### Full Response with References\n{formatted_answer}\n\n"

    if half_max is not None:
        result += f"**Proposed half-max:** {half_max}\n"
    if hill_power is not None:
        result += f"**Proposed Hill power:** {hill_power}\n"

    result += f"\n**Full answer saved to:** `{answer_file}`\n"

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
        name: Collection name for organizing results (e.g., "tumor_model_v1")
        rules: List of rule dicts to validate

    Returns:
        str: Combined validation results for all rules
    """
    if not EDISON_AVAILABLE:
        return "**Error:** edison-client is not installed."

    if not rules:
        return "**Error:** No rules provided."

    # Check for API key
    if not os.environ.get("EDISON_PLATFORM_API_KEY"):
        return (
            "**Error:** EDISON_PLATFORM_API_KEY environment variable is not set.\n\n"
            "Get your API key from https://platform.edisonscientific.com/profile "
            "and set it in the MCP server env config."
        )

    name = name.strip().lower().replace(" ", "_")
    answers_dir = _collection_dir(name) / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Separate cached (answer file exists) from uncached rules
    cached_rules: dict[int, tuple[dict[str, Any], Path]] = {}  # index -> (rule, answer_file)
    uncached_rules: list[tuple[int, dict[str, Any], str, Path]] = []  # (index, rule, question, answer_file)
    skipped: dict[int, str] = {}  # index -> skip reason

    for i, rule in enumerate(rules):
        cell_type = rule.get("cell_type", "")
        signal = rule.get("signal", "")
        direction = rule.get("direction", "")
        behavior = rule.get("behavior", "")

        if not all([cell_type, signal, direction, behavior]):
            skipped[i] = (
                f"**Rule {i+1}:** Skipped \u2014 missing required fields "
                f"(need cell_type, signal, direction, behavior)"
            )
            continue

        if direction not in ("increases", "decreases"):
            skipped[i] = (
                f"**Rule {i+1}:** Skipped \u2014 direction must be 'increases' or 'decreases'."
            )
            continue

        # Check for cached answer file
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{behavior}")
        answer_file = answers_dir / f"{safe_key}.md"

        if answer_file.exists():
            # Cache hit — read existing answer, no Edison API call needed
            cached_rules[i] = (rule, answer_file)
        else:
            # Cache miss — need to query Edison
            question = _build_validation_question(
                cell_type, signal, direction, behavior,
                rule.get("half_max"), rule.get("hill_power"),
                rule.get("signal_units"),
            )
            uncached_rules.append((i, rule, question, answer_file))

    # Phase 2: Send only UNCACHED rules to Edison in parallel
    edison_responses: dict[int, Any] = {}  # index -> response
    edison_errors: dict[int, str] = {}  # index -> error message

    if uncached_rules:
        try:
            client = _get_edison_client()
            task_requests = [
                TaskRequest(name=JobNames.LITERATURE, query=q)
                for _, _, q, _ in uncached_rules
            ]
            responses = await client.arun_tasks_until_done(
                task_requests, concurrency=10
            )
            for (idx, _, _, _), resp in zip(uncached_rules, responses):
                edison_responses[idx] = resp
        except Exception as e:
            for idx, _, _, _ in uncached_rules:
                edison_errors[idx] = f"**Error querying Edison API:** {e}"

    # Phase 3: Process results for each rule (cached + fresh)
    ordered_results: list[tuple[int, str]] = []

    for i, rule in enumerate(rules):
        if i in skipped:
            ordered_results.append((i, skipped[i]))
            continue

        cell_type = rule["cell_type"]
        signal = rule["signal"]
        direction = rule["direction"]
        behavior = rule["behavior"]
        half_max = rule.get("half_max")
        hill_power = rule.get("hill_power")

        # Determine answer source: cache or Edison response
        if i in cached_rules:
            _, answer_file = cached_rules[i]
            formatted_answer = answer_file.read_text(encoding="utf-8")
            answer_text = formatted_answer
            has_answer = True
            cached = True
        elif i in edison_errors:
            ordered_results.append((i, edison_errors[i]))
            continue
        elif i in edison_responses:
            response = edison_responses[i]
            answer_text = response.answer or ""
            formatted_answer = response.formatted_answer or answer_text
            has_answer = response.has_successful_answer
            cached = False

            # Save new answer file
            safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{behavior}")
            answer_file = answers_dir / f"{safe_key}.md"
            answer_file.write_text(formatted_answer, encoding="utf-8")
        else:
            ordered_results.append((i, f"**Rule {i+1}:** No response received."))
            continue

        # Parse support level and direction
        support_level = _parse_support_level(answer_text)
        if support_level == "unsupported":
            alt = _parse_support_level(formatted_answer)
            if alt != "unsupported":
                support_level = alt

        literature_direction = _parse_direction(answer_text)
        if literature_direction == "ambiguous":
            alt = _parse_direction(formatted_answer)
            if alt != "ambiguous":
                literature_direction = alt

        # Direction comparison
        if literature_direction == "ambiguous":
            direction_match = None
        elif literature_direction == direction:
            direction_match = True
        else:
            direction_match = False

        # Store validation result
        validation = {
            "cell_type": cell_type,
            "signal": signal,
            "direction": direction,
            "behavior": behavior,
            "half_max": half_max,
            "hill_power": hill_power,
            "support_level": support_level,
            "evidence_summary": answer_text,
            "answer_file": str(answer_file),
            "literature_direction": literature_direction,
            "direction_match": direction_match,
        }
        if name not in _validation_results:
            _validation_results[name] = []
        _validation_results[name].append(validation)

        # Format result (same as validate_rule output)
        dir_arrow = "\u2191" if direction == "increases" else "\u2193"
        support_labels = {
            "strong": "**STRONG**",
            "moderate": "**MODERATE**",
            "weak": "**WEAK**",
            "unsupported": "**UNSUPPORTED**",
        }

        result = f"## Rule Validation: {cell_type} | {signal} {dir_arrow} {behavior}\n\n"

        if cached:
            result += "**Source:** Cached from previous validation (no Edison API call)\n\n"

        if direction_match is False:
            result += (
                f"### DIRECTION MISMATCH\n\n"
                f"Literature says {signal} **{literature_direction}** {behavior}, "
                f"but rule proposes {signal} **{direction}** {behavior}.\n\n"
            )

        result += f"**Support Level:** {support_labels.get(support_level, support_level)}\n"
        result += f"**Literature Direction:** {literature_direction}\n"
        if direction_match is True:
            result += f"**Direction:** Confirmed by literature\n"
        elif direction_match is False:
            result += f"**Direction:** MISMATCH \u2014 proposed '{direction}', literature says '{literature_direction}'\n"
        else:
            result += f"**Direction:** Could not be determined from literature\n"

        if not has_answer:
            result += f"**Note:** Edison could not find a definitive answer for this query\n"
        result += "\n"

        result += f"### Evidence Summary\n{answer_text}\n\n"

        if formatted_answer and formatted_answer != answer_text:
            result += f"### Full Response with References\n{formatted_answer}\n\n"

        if half_max is not None:
            result += f"**Proposed half-max:** {half_max}\n"
        if hill_power is not None:
            result += f"**Proposed Hill power:** {hill_power}\n"

        result += f"\n**Full answer saved to:** `{answer_file}`\n"

        ordered_results.append((i, result))

    # Sort by original index and combine
    ordered_results.sort(key=lambda x: x[0])
    all_results = [r for _, r in ordered_results]

    n_cached = len(cached_rules)
    n_fresh = len(edison_responses)
    n_errors = len(edison_errors) + len(skipped)
    header = f"## Batch Validation Results ({len(rules)} rules)\n\n"
    if n_cached > 0:
        header += f"**{n_cached} cached** (reused from previous validation), **{n_fresh} queried** via Edison API"
        if n_errors > 0:
            header += f", **{n_errors} skipped/errored**"
        header += "\n\n"
    return header + "\n---\n\n".join(all_results)


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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mcp.run()
