"""Tests for the paper sources (no network: HTTP calls are mocked)."""

import json
from unittest.mock import MagicMock

import pytest

from core.sources import base, semantic_scholar
from core.sources.base import NormalizedPaper, PaperSource


def _response(status: int, payload: dict | None = None, headers: dict | None = None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json.return_value = payload or {}
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda _s: None)


def test_request_with_retry_retries_on_429_then_succeeds(monkeypatch):
    calls = iter([_response(429, headers={"Retry-After": "1"}), _response(200)])
    get = MagicMock(side_effect=lambda *a, **k: next(calls))
    monkeypatch.setattr(base.requests, "get", get)

    resp = PaperSource.request_with_retry("https://example.org")

    assert resp.status_code == 200
    assert get.call_count == 2


def test_request_with_retry_returns_non_retryable_errors_immediately(monkeypatch):
    get = MagicMock(return_value=_response(404))
    monkeypatch.setattr(base.requests, "get", get)

    assert PaperSource.request_with_retry("https://example.org").status_code == 404
    assert get.call_count == 1


def test_normalized_paper_llm_dict_truncates_abstract():
    paper = NormalizedPaper(
        paper_id="2401.00001",
        title="T",
        authors="A",
        published="2024",
        abstract="x" * 1000,
        source="arxiv",
        pdf_url=None,
        external_url="https://arxiv.org/abs/2401.00001",
        raw=None,
    )
    llm = paper.to_llm_dict()
    assert llm["id"] == "2401.00001"
    assert len(llm["s"]) == 200


def test_semantic_scholar_search_prefers_arxiv_id_and_fills_cache(monkeypatch):
    payload = {
        "data": [
            {
                "paperId": "abc",
                "title": "With ArXiv id",
                "authors": [{"name": "Ada Lovelace"}],
                "year": 2024,
                "abstract": "Abstract one",
                "externalIds": {"ArXiv": "2401.00001"},
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.00001"},
            },
            {
                "paperId": "def",
                "title": "Without ArXiv id",
                "authors": [],
                "year": None,
                "abstract": None,
                "externalIds": {},
                "openAccessPdf": None,
            },
        ]
    }
    monkeypatch.setattr(
        PaperSource, "request_with_retry", staticmethod(lambda *a, **k: _response(200, payload))
    )
    semantic_scholar._cache.clear()

    results = json.loads(semantic_scholar.semantic_scholar_search.invoke({"query": "rag"}))

    assert [r["id"] for r in results] == ["2401.00001", "ss_def"]
    source = semantic_scholar.SemanticScholarSource()
    paper = source.get_paper("ss_def")
    assert paper is not None
    assert paper.pdf_url is None
    assert paper.external_url == "https://semanticscholar.org/paper/def"


def test_semantic_scholar_download_skips_papers_without_open_access_pdf(tmp_path):
    paper = NormalizedPaper("ss_x", "T", "A", "2024", "", "semantic_scholar", None, "", None)
    assert semantic_scholar.SemanticScholarSource().download_pdf(paper, str(tmp_path)) is None
