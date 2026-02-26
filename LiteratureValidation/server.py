# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastmcp>=2.14.4",
#   "edison-client",
#   "httpx>=0.27",
# ]
# ///
"""
Literature Search MCP Server

Searches published biomedical literature using Edison Scientific's PaperQA3
API, which queries 150M+ papers automatically — no manual paper curation needed.

The agent uses this server to ask any biological question during model building
and receives evidence-based answers with citations.
"""

import hashlib
import os
import re
from pathlib import Path

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
# HELPERS
# ============================================================================


def _collection_dir(name: str) -> Path:
    """Get directory for a named collection."""
    return COLLECTIONS_DIR / name


def _query_cache_key(query: str) -> str:
    """Generate a filesystem-safe cache key from a query string."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _extract_answer_from_response(response: object) -> tuple[str, str, bool | None]:
    """Extract answer fields from an Edison API response.

    The PQATaskResponse.validate_pqa_fields() silently returns None for all
    answer fields when environment_frame is missing or empty (empty dict is
    falsy in Python). This function robustly extracts from both the typed
    PQA fields AND the raw environment_frame as a fallback.

    Returns (answer_text, formatted_answer, has_successful_answer).
    """
    # Try standard PQA fields first
    answer_text = getattr(response, "answer", None) or ""
    formatted_answer = getattr(response, "formatted_answer", None) or ""
    has_answer = getattr(response, "has_successful_answer", None)

    # Fallback: extract from environment_frame (verbose responses preserve it)
    if not answer_text:
        env_frame = getattr(response, "environment_frame", None) or {}
        state = env_frame.get("state", {}).get("state", {})
        pqa_response = state.get("response", {})
        answer_obj = pqa_response.get("answer", {})
        if isinstance(answer_obj, dict):
            answer_text = answer_obj.get("answer", "") or ""
            formatted_answer = answer_obj.get("formatted_answer", "") or formatted_answer
            if has_answer is None:
                has_answer = answer_obj.get("has_successful_answer")
        elif isinstance(answer_obj, str):
            answer_text = answer_obj

    return answer_text, formatted_answer or answer_text, has_answer


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP(
    "LiteratureValidation",
    instructions=(
        "Literature search server for biomedical questions. "
        "Uses Edison Scientific's PaperQA3 API to search 150M+ published papers. "
        "Use search_literature() to ask any biological question and get an "
        "evidence-based answer with citations. Results are cached for reuse."
    ),
)


# ============================================================================
# TOOLS
# ============================================================================

@mcp.tool()
async def search_literature(
    query: str,
    collection_name: str = "default",
) -> str:
    """
    Search published biomedical literature for answers to any question.

    Uses Edison Scientific's PaperQA3 API to automatically search 150M+
    papers. Ask any biological question — parameter values, mechanisms,
    dose-response relationships, experimental evidence, etc.

    Results are cached so repeated queries return instantly.

    Args:
        query: Any question about biology, cell behavior, parameters, etc.
               Examples:
               - "What oxygen concentration causes tumor cell necrosis?"
               - "Do macrophages increase or decrease tumor proliferation?"
               - "What is the Hill coefficient for oxygen-dependent cell cycle arrest?"
               - "What is a typical migration speed for breast cancer cells?"
        collection_name: Cache namespace for organizing results (default: "default")

    Returns:
        str: Evidence-based answer with citations from published literature
    """
    if not EDISON_AVAILABLE:
        return (
            "**Error:** edison-client is not installed.\n\n"
            "Install with: `pip install edison-client`"
        )

    if not os.environ.get("EDISON_PLATFORM_API_KEY"):
        return (
            "**Error:** EDISON_PLATFORM_API_KEY environment variable is not set.\n\n"
            "Get your API key from https://platform.edisonscientific.com/profile "
            "and set it in the MCP server env config."
        )

    if not query.strip():
        return "**Error:** query cannot be empty."

    collection_name = collection_name.strip().lower().replace(" ", "_")

    # Ensure answers directory exists
    answers_dir = _collection_dir(collection_name) / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_key = _query_cache_key(query)
    answer_file = answers_dir / f"{cache_key}.md"

    if answer_file.exists():
        cached_answer = answer_file.read_text(encoding="utf-8")
        if cached_answer.strip():  # Skip empty cached files (previous extraction failures)
            return (
                f"## Literature Search Result (cached)\n\n"
                f"**Query:** {query}\n\n"
                f"---\n\n"
                f"{cached_answer}\n\n"
                f"---\n\n"
                f"**Source:** Cached answer from `{answer_file}`"
            )

    # Query Edison API
    try:
        client = _get_edison_client()
        task_data = TaskRequest(
            name=JobNames.LITERATURE,
            query=query,
        )
        # Use verbose=True so the response preserves environment_frame,
        # which we need as a fallback when PQATaskResponse field extraction
        # fails (its validator silently returns None for all answer fields
        # when environment_frame is missing or empty).
        responses = await client.arun_tasks_until_done(task_data, verbose=True)
        response = responses[0]

        answer_text, formatted_answer, has_answer = _extract_answer_from_response(response)
    except Exception as e:
        return f"**Error querying Edison API:** {e}"

    # Only cache non-empty answers (so failed extractions can be retried)
    if formatted_answer.strip():
        answer_file.write_text(formatted_answer, encoding="utf-8")

    # Format result
    result = f"## Literature Search Result\n\n"
    result += f"**Query:** {query}\n\n"

    if not has_answer:
        result += "**Note:** Edison could not find a definitive answer for this query.\n\n"

    result += f"---\n\n"
    result += f"{answer_text}\n\n"

    if formatted_answer and formatted_answer != answer_text:
        result += f"### Full Response with References\n\n{formatted_answer}\n\n"

    result += f"---\n\n"
    result += f"**Answer cached to:** `{answer_file}`"

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mcp.run()
