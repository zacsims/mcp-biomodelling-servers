# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastmcp>=2.14.4",
#   "paper-qa>=5",
#   "httpx>=0.27",
#   "coredis>=4",
#   "metapub>=0.5",
#   "pypdf[image]",
#   "setuptools<72",
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
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import httpx

from mcp.server.fastmcp import FastMCP

# PaperQA imports
try:
    from paperqa import Docs, Settings
    from paperqa.settings import AnswerSettings, ParsingSettings
    PAPERQA_AVAILABLE = True
except ImportError:
    PAPERQA_AVAILABLE = False

# metapub for PDF discovery (68+ publishers, 97% coverage)
try:
    from metapub import FindIt
    METAPUB_AVAILABLE = True
except ImportError:
    METAPUB_AVAILABLE = False

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
    """Build PaperQA Settings using OpenAI o4-mini."""
    return Settings(
        llm="o4-mini",
        summary_llm="o4-mini",
        temperature=1,  # Required for reasoning models (o4-mini)
        embedding="text-embedding-3-small",
        parsing=ParsingSettings(
            # Chunk size larger than any paper so each summary = one paper
            reader_config={"chunk_chars": 500_000, "overlap": 0},
        ),
        answer=AnswerSettings(
            evidence_k=10,
            answer_max_sources=5,
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
    if name in _collections and _collections[name] is not None:
        return _collections[name]

    if not PAPERQA_AVAILABLE:
        raise RuntimeError(
            "paper-qa is not installed. Install with: pip install 'paper-qa>=5'"
        )

    settings = _get_paperqa_settings()
    docs = Docs()
    _collections[name] = docs
    return docs


async def _download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF from a URL to a local file path.

    Returns True on success, False on failure.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
            if len(content) < 1000:
                # Too small to be a real PDF
                return False
            dest.write_bytes(content)
            return True
    except Exception:
        return False


def _find_pdf_url(pmid: str | None = None, doi: str | None = None) -> str | None:
    """Use metapub FindIt to locate a PDF URL for a paper.

    FindIt covers 68+ publishers (97% coverage) including Elsevier, Wiley,
    Springer, Nature, PLOS, BMC, PMC, and many more.
    Returns a direct PDF URL or None.
    """
    if not METAPUB_AVAILABLE:
        return None
    try:
        import time
        time.sleep(0.4)  # Avoid NCBI rate limiting (3 req/s without API key)
        kwargs = {}
        if pmid:
            kwargs["pmid"] = pmid
        if doi:
            kwargs["doi"] = doi
        if not kwargs:
            return None
        src = FindIt(**kwargs)
        return src.url or None
    except Exception:
        return None


async def _unpaywall_pdf_url(doi: str) -> str | None:
    """Query Unpaywall API for an open-access PDF URL.

    Unpaywall aggregates OA locations including preprint repositories
    (bioRxiv, medRxiv), institutional repositories, and green OA copies.
    Falls back through all oa_locations if best_oa_location has no PDF.
    """
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email=mcp_literature_validation@users.noreply.github.com"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("is_oa"):
            return None

        # Try best_oa_location first
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            return best["url_for_pdf"]

        # Fall through all oa_locations looking for a PDF URL
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
            # Some locations only have url (landing page), skip those

    except Exception:
        pass
    return None


async def _try_fetch_pdf(paper: dict, papers_path: Path) -> Path | None:
    """Attempt to fetch a full PDF for a paper.

    Strategy (in order):
    1. bioRxiv direct download (if bioRxiv DOI)
    2. metapub FindIt — 68+ publisher-specific strategies
    3. Unpaywall — finds preprints/OA copies of paywalled papers

    Returns the PDF file path on success, None on failure.
    """
    title = paper.get("title", "unknown")
    safe_name = hashlib.md5(title.encode()).hexdigest()[:12]
    pdf_path = papers_path / f"{safe_name}.pdf"

    # Already downloaded
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    pmid = paper.get("pmid")
    doi = paper.get("doi", "")
    biorxiv_doi = paper.get("biorxiv_doi")

    # 1. bioRxiv DOI — direct PDF download (always available)
    if biorxiv_doi or (doi and doi.startswith("10.1101/")):
        biorxiv_doi = biorxiv_doi or doi
        pdf_url = f"https://www.biorxiv.org/content/{biorxiv_doi}v1.full.pdf"
        if await _download_pdf(pdf_url, pdf_path):
            return pdf_path

    # 2. metapub FindIt — covers 68+ publishers (Elsevier, Wiley, Springer, etc.)
    pdf_url = _find_pdf_url(pmid=pmid, doi=doi if doi else None)
    if pdf_url:
        if await _download_pdf(pdf_url, pdf_path):
            return pdf_path

    # 3. Unpaywall — finds preprints & OA copies of paywalled papers
    pdf_url = await _unpaywall_pdf_url(doi)
    if pdf_url:
        if await _download_pdf(pdf_url, pdf_path):
            return pdf_path

    return None


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
        "Uses PaperQA2 to validate rules against published biomedical literature. "
        "Workflow: create_paper_collection → add_papers_to_collection (with fetch_pdfs=True) "
        "or add_papers_by_id → validate_rule/validate_rules_batch → get_validation_summary. "
        "Supports full PDF indexing from PubMed Central and bioRxiv."
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
            "The OPENAI_API_KEY environment variable must also be set."
        )

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        return (
            "**Error:** OPENAI_API_KEY environment variable is not set.\n\n"
            "PaperQA uses OpenAI models for question answering. "
            "Set your OpenAI API key in the MCP server env config."
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
    fetch_pdfs: bool = False,
) -> str:
    """
    Add papers to a collection for indexing by PaperQA (PDF-only).

    Only papers with downloadable full PDFs are indexed. Papers without
    available PDFs are skipped entirely (no abstract-only fallback).

    Each paper dict should have:
    - "title": Paper title
    - "text": Abstract or full text content
    - "pmid" (optional): PubMed ID
    - "doi" (optional): DOI
    - "biorxiv_doi" (optional): bioRxiv DOI for direct PDF download
    - "authors" (optional): Author list string
    - "year" (optional): Publication year

    Args:
        name: Collection name (must exist)
        papers: List of paper dicts with at minimum "title" and "text"
        fetch_pdfs: If True, attempt to download full PDFs from PMC/bioRxiv.
                    Papers without PDFs are skipped. (default: False)

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

    settings = _get_paperqa_settings()
    added_count = 0
    skipped_no_pdf = 0
    errors = []

    for paper in papers:
        title = paper.get("title", "").strip()
        text = paper.get("text", "").strip()

        if not title or not text:
            errors.append(f"Skipped paper with missing title or text")
            continue

        paper_file = None

        # Try PDF download
        if fetch_pdfs:
            try:
                pdf_path = await _try_fetch_pdf(paper, papers_path)
                if pdf_path:
                    paper_file = pdf_path
            except Exception:
                pass

        # Skip papers without full PDF
        if paper_file is None:
            skipped_no_pdf += 1
            errors.append(f"Skipped '{title[:60]}' — no full PDF available")
            continue

        # Add to PaperQA Docs
        try:
            await docs.aadd(
                paper_file,
                citation=title,
                docname=title[:80],
                settings=settings,
            )
            added_count += 1
        except Exception as e:
            errors.append(f"Failed to index '{title[:40]}...': {e}")

    result = f"## Papers Added to `{name}`\n\n"
    result += f"**Indexed:** {added_count} / {len(papers)} papers (PDFs only)\n"
    if skipped_no_pdf > 0:
        result += f"**Skipped (no PDF):** {skipped_no_pdf}\n"
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
async def add_papers_by_id(
    name: str,
    pmids: list[str] | None = None,
    biorxiv_dois: list[str] | None = None,
    fetch_pdfs: bool = True,
) -> str:
    """
    Add papers by PubMed ID or bioRxiv DOI to a collection (PDF-only).

    Automatically fetches metadata (title, abstract) and downloads full PDFs.
    Papers without available PDFs are skipped entirely (no abstract-only fallback).
    For PubMed papers, PDFs are fetched via metapub FindIt and Unpaywall.
    For bioRxiv preprints, PDFs are always available and fetched by default.

    Args:
        name: Collection name (must exist)
        pmids: List of PubMed IDs (e.g., ["35486828", "33264437"])
        biorxiv_dois: List of bioRxiv DOIs (e.g., ["10.1101/2024.01.15.123456"])
        fetch_pdfs: If True, download full PDFs. Papers without PDFs are skipped. (default: True)

    Returns:
        str: Summary of papers added
    """
    if not PAPERQA_AVAILABLE:
        return "**Error:** paper-qa is not installed."

    name = name.strip().lower().replace(" ", "_")
    papers_path = _papers_dir(name)
    if not papers_path.exists():
        return f"**Error:** Collection `{name}` does not exist. Create it first with `create_paper_collection()`."

    if not pmids and not biorxiv_dois:
        return "**Error:** Provide at least one PMID or bioRxiv DOI."

    try:
        docs = await _get_or_create_docs(name)
    except RuntimeError as e:
        return f"**Error:** {e}"

    settings = _get_paperqa_settings()
    added_count = 0
    skipped_no_pdf = 0
    errors = []

    # Process PubMed IDs
    if pmids:
        # Fetch metadata from NCBI E-utilities efetch
        ids_str = ",".join(pmids)
        efetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={ids_str}&rettype=xml&retmode=xml"
        )
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(efetch_url)
                resp.raise_for_status()
                xml_text = resp.text
        except Exception as e:
            errors.append(f"Failed to fetch PubMed metadata: {e}")
            xml_text = ""

        if xml_text:
            # Simple XML parsing for title, abstract, PMID
            # Split into individual articles
            articles = re.findall(
                r"<PubmedArticle>(.*?)</PubmedArticle>", xml_text, re.DOTALL
            )
            for article_xml in articles:
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article_xml)
                title_match = re.search(
                    r"<ArticleTitle>(.*?)</ArticleTitle>", article_xml, re.DOTALL
                )
                # Abstract may have multiple AbstractText elements
                abstract_parts = re.findall(
                    r"<AbstractText[^>]*>(.*?)</AbstractText>", article_xml, re.DOTALL
                )

                pmid = pmid_match.group(1) if pmid_match else "unknown"
                title = title_match.group(1).strip() if title_match else f"PMID:{pmid}"
                # Strip any remaining XML tags from title
                title = re.sub(r"<[^>]+>", "", title)
                abstract = " ".join(
                    re.sub(r"<[^>]+>", "", part).strip() for part in abstract_parts
                )

                if not abstract:
                    errors.append(f"PMID {pmid}: No abstract available")
                    continue

                # Extract DOI if present
                doi_match = re.search(
                    r'<ArticleId IdType="doi">([^<]+)</ArticleId>', article_xml
                )
                doi = doi_match.group(1) if doi_match else ""

                # Extract authors
                author_parts = re.findall(
                    r"<LastName>([^<]+)</LastName>\s*<ForeName>([^<]+)</ForeName>",
                    article_xml,
                )
                authors = ", ".join(f"{ln} {fn}" for ln, fn in author_parts[:5])
                if len(author_parts) > 5:
                    authors += " et al."

                # Extract year
                year_match = re.search(
                    r"<PubDate>\s*<Year>(\d{4})</Year>", article_xml
                )
                year = year_match.group(1) if year_match else ""

                paper = {
                    "title": title,
                    "text": abstract,
                    "pmid": pmid,
                    "doi": doi,
                    "authors": authors,
                    "year": year,
                }

                paper_file = None

                if fetch_pdfs:
                    try:
                        pdf_path = await _try_fetch_pdf(paper, papers_path)
                        if pdf_path:
                            paper_file = pdf_path
                    except Exception:
                        pass

                # Skip papers without full PDF
                if paper_file is None:
                    skipped_no_pdf += 1
                    errors.append(f"Skipped PMID {pmid} — no full PDF available")
                    continue

                try:
                    await docs.aadd(
                        paper_file,
                        citation=title,
                        docname=title[:80],
                        settings=settings,
                    )
                    added_count += 1
                except Exception as e:
                    errors.append(f"Failed to index PMID {pmid}: {e}")

    # Process bioRxiv DOIs
    if biorxiv_dois:
        for biorxiv_doi in biorxiv_dois:
            # Normalize DOI
            biorxiv_doi = biorxiv_doi.strip()
            if biorxiv_doi.startswith("https://doi.org/"):
                biorxiv_doi = biorxiv_doi[len("https://doi.org/"):]

            # Fetch metadata from bioRxiv API
            api_url = f"https://api.biorxiv.org/details/biorxiv/{biorxiv_doi}"
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(api_url)
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                errors.append(f"bioRxiv {biorxiv_doi}: Failed to fetch metadata: {e}")
                continue

            collection = data.get("collection", [])
            if not collection:
                errors.append(f"bioRxiv {biorxiv_doi}: No metadata found")
                continue

            # Use the latest version
            entry = collection[-1]
            title = entry.get("title", f"bioRxiv:{biorxiv_doi}")
            abstract = entry.get("abstract", "")
            authors = entry.get("authors", "")
            date = entry.get("date", "")
            year = date[:4] if date else ""
            version = entry.get("version", "1")

            if not abstract:
                errors.append(f"bioRxiv {biorxiv_doi}: No abstract available")
                continue

            paper = {
                "title": title,
                "text": abstract,
                "biorxiv_doi": biorxiv_doi,
                "doi": biorxiv_doi,
                "authors": authors,
                "year": year,
            }

            paper_file = None

            if fetch_pdfs:
                # bioRxiv PDFs use the version number
                safe_name = hashlib.md5(title.encode()).hexdigest()[:12]
                pdf_path = papers_path / f"{safe_name}.pdf"
                pdf_url = f"https://www.biorxiv.org/content/{biorxiv_doi}v{version}.full.pdf"
                try:
                    if await _download_pdf(pdf_url, pdf_path):
                        paper_file = pdf_path
                except Exception:
                    pass

            # Skip papers without full PDF
            if paper_file is None:
                skipped_no_pdf += 1
                errors.append(f"Skipped bioRxiv {biorxiv_doi} — no full PDF available")
                continue

            try:
                await docs.aadd(
                    paper_file,
                    citation=title,
                    docname=title[:80],
                    settings=settings,
                )
                added_count += 1
            except Exception as e:
                errors.append(f"Failed to index bioRxiv {biorxiv_doi}: {e}")

    total_requested = len(pmids or []) + len(biorxiv_dois or [])
    result = f"## Papers Added to `{name}`\n\n"
    result += f"**Indexed:** {added_count} / {total_requested} papers (PDFs only)\n"
    if skipped_no_pdf > 0:
        result += f"**Skipped (no PDF):** {skipped_no_pdf}\n"
    result += f"**Papers directory:** `{papers_path}`\n"

    if errors:
        result += f"\n**Warnings:**\n"
        for err in errors[:10]:
            result += f"- {err}\n"
        if len(errors) > 10:
            result += f"- ... and {len(errors) - 10} more\n"

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
    signal_units: str | None = None,
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
        signal_units: Units for signal values (e.g., "mmHg", "mM"). Auto-detected
                      for common signals (oxygen=mmHg, glucose=mM) if not provided.

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

    # Check that the collection has papers (PDFs only)
    papers_path = _papers_dir(name)
    paper_files = list(papers_path.glob("*.pdf"))
    if not paper_files:
        return (
            f"**Error:** Collection `{name}` has no papers indexed.\n"
            f"Add papers with `add_papers_to_collection()` first."
        )

    # Build and ask the question
    settings = _get_paperqa_settings()
    question = _build_validation_question(
        cell_type, signal, direction, behavior, half_max, hill_power, signal_units
    )

    try:
        response = await docs.aquery(question, settings=settings)
        answer_text = response.answer
        references = response.references
        formatted_answer = response.formatted_answer
    except Exception as e:
        return f"**Error querying PaperQA:** {e}"

    # Save full PaperQA answer to output file (direction-agnostic key)
    answers_dir = _collection_dir(name) / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{cell_type}_{signal}_{behavior}")
    answer_file = answers_dir / f"{safe_key}.md"
    answer_file.write_text(formatted_answer, encoding="utf-8")

    # Determine support level and literature direction
    support_level = _parse_support_level(answer_text)
    literature_direction = _parse_direction(answer_text)

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
        "references": references,
        "answer_file": str(answer_file),
        "literature_direction": literature_direction,
        "direction_match": direction_match,
    }
    if name not in _validation_results:
        _validation_results[name] = []
    _validation_results[name].append(validation)

    # Format result
    dir_arrow = "↑" if direction == "increases" else "↓"
    support_labels = {
        "strong": "**STRONG**",
        "moderate": "**MODERATE**",
        "weak": "**WEAK**",
        "unsupported": "**UNSUPPORTED**",
    }

    result = f"## Rule Validation: {cell_type} | {signal} {dir_arrow} {behavior}\n\n"

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
        result += f"**Direction:** MISMATCH — proposed '{direction}', literature says '{literature_direction}'\n"
    else:
        result += f"**Direction:** Could not be determined from literature\n"
    result += "\n"

    result += f"### Evidence Summary\n{answer_text}\n\n"

    if references:
        result += f"### Key References\n{references}\n\n"

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
            signal_units=rule.get("signal_units"),
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
        "`search_articles()` tool, then add the returned papers to "
        "this collection with `add_papers_to_collection(fetch_pdfs=True)` "
        "or `add_papers_by_id()` with their PMIDs.\n"
    )

    # bioRxiv category suggestions
    biorxiv_categories: dict[str, list[str]] = {
        "cancer": ["cancer biology", "cell biology"],
        "tumor": ["cancer biology", "cell biology"],
        "macrophage": ["immunology", "cell biology"],
        "T cell": ["immunology"],
        "immune": ["immunology"],
        "fibroblast": ["cell biology", "cancer biology"],
        "endothelial": ["cell biology", "physiology"],
        "epithelial": ["cell biology", "developmental biology"],
        "neuron": ["neuroscience"],
        "stem cell": ["developmental biology", "cell biology"],
    }
    signal_categories: dict[str, list[str]] = {
        "oxygen": ["cell biology", "physiology", "cancer biology"],
        "glucose": ["cell biology", "biochemistry"],
        "VEGF": ["cancer biology", "cell biology"],
        "TNF": ["immunology", "cell biology"],
        "TGF-beta": ["cancer biology", "cell biology", "developmental biology"],
        "IFN-gamma": ["immunology"],
    }

    # Collect relevant categories
    categories: set[str] = set()
    for key, cats in biorxiv_categories.items():
        if key.lower() in cell_type.lower():
            categories.update(cats)
    for key, cats in signal_categories.items():
        if key.lower() in signal.lower():
            categories.update(cats)
    if not categories:
        categories = {"cell biology"}

    result += "\n### bioRxiv Preprint Search\n\n"
    result += (
        "The bioRxiv MCP server uses **category + date range** filtering "
        "(no keyword search). Suggested categories:\n"
    )
    for cat in sorted(categories):
        result += f"- `{cat}`\n"
    result += (
        "\nUse `search_preprints(category=\"...\", recent_days=90)` from the "
        "bioRxiv MCP server to find recent preprints, then add them with "
        "`add_papers_by_id(biorxiv_dois=[...])`."
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mcp.run()
