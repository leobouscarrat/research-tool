"""Tests for LLM score parsing, theme persistence and the MCP tool surface."""

import asyncio

import pytest

from core import theme_manager
from core.research_agent import _extract_score_from_response


@pytest.mark.parametrize(
    "content, expected",
    [
        ('```json\n{"relevance_score": 8, "justification": "ok"}\n```', 8),
        ('Sure! {"relevance_score": 3, "justification": "off topic"} Hope it helps.', 3),
        ('{"score": 9}', None),
        ("no json here", None),
    ],
)
def test_extract_score_from_llm_response(content, expected):
    parsed = _extract_score_from_response(content)
    if expected is None:
        assert parsed is None
    else:
        assert parsed["relevance_score"] == expected


@pytest.fixture
def isolated_themes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(theme_manager, "THEMES_FILE", tmp_path / "data/themes.json")
    monkeypatch.setattr(theme_manager, "THEMES_DIR", tmp_path / "data/themes")
    return tmp_path


def test_create_theme_is_idempotent(isolated_themes):
    assert theme_manager.create_theme("LLM Evaluation") is True
    assert theme_manager.create_theme("LLM Evaluation") is False
    assert "llm_evaluation" in theme_manager.list_themes()
    assert (isolated_themes / "data/themes/llm_evaluation/pdfs").is_dir()


def test_add_paper_deduplicates(isolated_themes):
    theme_manager.create_theme("RAG")
    paper = {"paper_id": "p1", "title": "T", "authors": "A", "published": "2024", "abstract": ""}
    theme_manager.add_paper("rag", paper)
    theme_manager.add_paper("rag", paper)
    assert theme_manager.get_paper_ids("rag") == {"p1"}


def test_sync_papers_from_disk_drops_entries_without_pdf(isolated_themes):
    theme_manager.create_theme("RAG")
    for pid in ("kept", "missing"):
        theme_manager.add_paper(
            "rag", {"paper_id": pid, "title": pid, "authors": "", "published": "", "abstract": ""}
        )
    (isolated_themes / "data/themes/rag/pdfs/kept.pdf").write_bytes(b"%PDF")

    theme_manager.sync_papers_from_disk("rag")

    assert theme_manager.get_paper_ids("rag") == {"kept"}
    assert theme_manager.get_paper_count("rag") == 1


def test_mcp_server_exposes_expected_tools(monkeypatch):
    import os

    cwd = os.getcwd()
    try:
        import mcp_server

        tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    finally:
        os.chdir(cwd)

    assert tools == {
        "list_research_themes",
        "create_research_theme",
        "list_indexed_papers",
        "search_and_index_papers",
        "ask_question",
    }
