import json
import time
from pathlib import Path

import arxiv
from langchain_core.tools import tool

from core.logger import logger
from core.sources.base import _BACKOFF_SECONDS, NormalizedPaper, PaperSource

_cache: dict[str, arxiv.Result] = {}


@tool
def arxiv_search(query: str, max_results: int = 15) -> str:
    """Search ArXiv for academic papers matching a query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    raw_results = None
    for attempt, wait in enumerate(_BACKOFF_SECONDS, start=1):
        try:
            raw_results = list(client.results(search))
            break
        except Exception as e:
            logger.warning(f"ArXiv erreur (tentative {attempt}/{len(_BACKOFF_SECONDS)}) : {e} — attente {wait}s")
            time.sleep(wait)
    if raw_results is None:
        logger.error("ArXiv : échec après toutes les tentatives")
        return json.dumps([])
    for paper in raw_results:
        arxiv_id = paper.entry_id.split("/")[-1]
        _cache[arxiv_id] = paper
        results.append({
            "id": arxiv_id,
            "t": paper.title,
            "a": ", ".join(a.name.split()[-1] for a in paper.authors[:2]),
            "y": str(paper.published.year),
            "s": paper.summary[:1000],
            "source": "arxiv"
        })
    logger.info(f"ArXiv : {len(results)} papiers récupérés")
    return json.dumps(results, ensure_ascii=False, separators=(",", ":"))


class ArxivSource(PaperSource):

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def display_name(self) -> str:
        return "ArXiv"

    @property
    def tool(self):
        return arxiv_search

    def get_paper(self, paper_id: str) -> NormalizedPaper | None:
        cached = _cache.get(paper_id)
        if not cached:
            return None
        return NormalizedPaper(
            paper_id=paper_id,
            title=cached.title,
            authors=", ".join(a.name for a in cached.authors[:3]),
            published=str(cached.published.date()),
            abstract=cached.summary[:500] + ("..." if len(cached.summary) > 500 else ""),
            source="arxiv",
            pdf_url=cached.pdf_url,
            external_url=f"https://arxiv.org/abs/{paper_id}",
            raw=cached
        )

    def download_pdf(self, paper: NormalizedPaper, dest_dir: str) -> str | None:
        pdf_path = Path(dest_dir) / f"{paper.paper_id}.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        for attempt, wait in enumerate(_BACKOFF_SECONDS, start=1):
            try:
                if pdf_path.exists():
                    pdf_path.unlink()
                paper.raw.download_pdf(dirpath=dest_dir, filename=f"{paper.paper_id}.pdf")
                logger.info(f"PDF téléchargé : {pdf_path}")
                return str(pdf_path)
            except Exception as e:
                logger.warning(
                    f"Échec téléchargement ArXiv PDF {paper.paper_id} "
                    f"(tentative {attempt}/{len(_BACKOFF_SECONDS)}) : {e} — attente {wait}s"
                )
                time.sleep(wait)
        logger.error(f"Abandon téléchargement ArXiv PDF {paper.paper_id} après {len(_BACKOFF_SECONDS)} tentatives")
        return None

    def clear_cache(self):
        _cache.clear()
