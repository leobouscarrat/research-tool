"""
MCP server exposing the research_tool functions to Claude Code.
Run via: uv run python mcp_server.py
"""

import os
import sys

# Ensure working directory is always the project root (theme_manager uses relative paths)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from core.paper_fetcher import download_and_index
from core.rag_agent import ask
from core.research_agent import run_research_agent_turn
from core.theme_manager import (
    create_theme,
    get_paper_count,
    get_papers,
    list_themes,
    sync_papers_from_disk,
)

mcp = FastMCP("research-tool")


# ── Themes ────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_research_themes() -> dict:
    """List all existing research themes with their metadata (paper count, last search date, etc.)."""
    themes = list_themes()
    result = {}
    for slug in themes:
        sync_papers_from_disk(slug)
        t = themes[slug]
        result[slug] = {
            "name": t.get("name"),
            "slug": t.get("slug"),
            "created_at": t.get("created_at"),
            "last_search": t.get("last_search"),
            "paper_count": get_paper_count(slug),
            "search_count": len(t.get("searches", [])),
        }
    return result


@mcp.tool()
def create_research_theme(name: str) -> str:
    """
    Create a new research theme (collection of indexed papers).

    Args:
        name: Human-readable name for the theme (e.g. 'EBS Systems & Tools').

    Returns:
        The slug (identifier) of the created theme, or an error message if it already exists.
    """
    slug = name.lower().replace(" ", "_")
    success = create_theme(name)
    if success:
        return f"Theme '{name}' created with slug '{slug}'."
    return f"Theme with slug '{slug}' already exists."


# ── Papers ────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_indexed_papers(theme_slug: str) -> list[dict]:
    """
    List all papers already indexed in a theme.

    Args:
        theme_slug: The theme identifier (e.g. 'ebs_systems_&_tools').
    """
    sync_papers_from_disk(theme_slug)
    papers = get_papers(theme_slug)
    return [
        {
            "paper_id": p["paper_id"],
            "title": p["title"],
            "authors": p["authors"],
            "published": p["published"],
            "source": p.get("source", "arxiv"),
            "external_url": p.get("external_url", ""),
        }
        for p in papers
    ]


@mcp.tool()
def search_and_index_papers(
    theme_slug: str,
    query: str,
    sources: list[str] | None = None,
    min_score: int = 5,
) -> str:
    """
    Search for academic papers on a topic, score them for relevance, download the PDFs,
    and index them in a theme's vector store.

    This operation can take several minutes (LLM scoring per paper).

    Args:
        theme_slug: The theme identifier to index papers into.
        query: Research topic or question (e.g. 'event-based surveillance infectious disease NLP').
        sources: List of sources to search. Available: ['arxiv', 'semantic_scholar'].
                 Defaults to ['arxiv'].
        min_score: Minimum relevance score (1-10) to keep a paper. Defaults to 5.

    Returns:
        A summary of how many papers were found and indexed.
    """
    if sources is None:
        sources = ["arxiv"]

    # Retrieve already-indexed paper IDs to avoid duplicates
    existing_ids = {p["paper_id"] for p in get_papers(theme_slug)}

    messages = [HumanMessage(content=query)]

    clarification, papers, _ = run_research_agent_turn(
        messages=messages,
        existing_paper_ids=existing_ids,
        min_score=min_score,
        sources=sources,
    )

    if clarification:
        return f"The search agent needs clarification: {clarification}"

    if not papers:
        return "No relevant papers found for this query."

    indexed = download_and_index(theme_slug, papers, subject=query)
    titles = "\n".join(f"  - {p.title[:80]}" for p in papers[:10])
    more = f"\n  ... and {len(papers) - 10} more" if len(papers) > 10 else ""
    return (
        f"Indexed {indexed} paper(s) into theme '{theme_slug}'.\n\n"
        f"Papers found:\n{titles}{more}"
    )


# ── Q&A ───────────────────────────────────────────────────────────────────────


@mcp.tool()
def ask_question(theme_slug: str, question: str) -> str:
    """
    Ask a question about the papers indexed in a theme.
    Uses RAG (Retrieval-Augmented Generation) to find relevant passages
    and answer with citations.

    Args:
        theme_slug: The theme identifier to search in.
        question: The question to ask (in any language).

    Returns:
        An answer grounded in the indexed papers, with citations.
    """
    paper_count = get_paper_count(theme_slug)
    if paper_count == 0:
        return f"No papers indexed in theme '{theme_slug}'. Run search_and_index_papers first."

    history = [{"role": "user", "content": question}]
    return ask(question=question, slug=theme_slug, history=history)


if __name__ == "__main__":
    mcp.run()
