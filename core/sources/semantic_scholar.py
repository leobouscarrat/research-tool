import json
import requests
from pathlib import Path
from langchain_core.tools import tool
from core.sources.base import PaperSource, NormalizedPaper
from core.logger import logger

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,authors,year,abstract,externalIds,openAccessPdf"

_cache: dict[str, dict] = {}


@tool
def semantic_scholar_search(query: str, max_results: int = 15) -> str:
    """Search Semantic Scholar for academic papers matching a query."""
    params = {"query": query, "limit": max_results, "fields": FIELDS}
    try:
        response = requests.get(SEMANTIC_SCHOLAR_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Semantic Scholar API error: {e}")
        return json.dumps([])

    results = []
    for paper in data.get("data", []):
        paper_id = paper.get("paperId", "")
        arxiv_id = paper.get("externalIds", {}).get("ArXiv")
        # Utilise arxiv_id si dispo, sinon préfixe ss_ pour éviter les collisions
        cache_id = arxiv_id if arxiv_id else f"ss_{paper_id}"
        pdf_url = (paper.get("openAccessPdf") or {}).get("url")

        _cache[cache_id] = {
            "paperId": paper_id,
            "arxiv_id": arxiv_id,
            "cache_id": cache_id,
            "title": paper.get("title", ""),
            "authors": ", ".join(a["name"] for a in paper.get("authors", [])[:3]),
            "year": str(paper.get("year") or ""),
            "abstract": (paper.get("abstract") or "")[:500] + "...",
            "pdf_url": pdf_url,
            "source": "semantic_scholar"
        }

        results.append({
            "id": cache_id,
            "t": paper.get("title", ""),
            "a": ", ".join(a["name"].split()[-1] for a in paper.get("authors", [])[:2]),
            "y": str(paper.get("year") or ""),
            "s": (paper.get("abstract") or "")[:1000],
            "source": "semantic_scholar"
        })

    logger.info(f"Semantic Scholar : {len(results)} papiers récupérés")
    return json.dumps(results, ensure_ascii=False, separators=(",", ":"))


class SemanticScholarSource(PaperSource):

    @property
    def name(self) -> str:
        return "semantic_scholar"

    @property
    def display_name(self) -> str:
        return "Semantic Scholar"

    @property
    def tool(self):
        return semantic_scholar_search

    def get_paper(self, paper_id: str) -> NormalizedPaper | None:
        cached = _cache.get(paper_id)
        if not cached:
            return None

        ss_id = cached["paperId"]
        return NormalizedPaper(
            paper_id=paper_id,
            title=cached["title"],
            authors=cached["authors"],
            published=cached["year"],
            abstract=cached["abstract"],
            source="semantic_scholar",
            pdf_url=cached.get("pdf_url"),
            external_url=f"https://semanticscholar.org/paper/{ss_id}",
            raw=cached
        )

    def download_pdf(self, paper: NormalizedPaper, dest_dir: str) -> str | None:
        if not paper.pdf_url:
            logger.warning(f"Pas de PDF open access pour {paper.paper_id}")
            return None

        pdf_path = Path(dest_dir) / f"{paper.paper_id}.pdf"
        if pdf_path.exists():
            return str(pdf_path)

        try:
            response = requests.get(paper.pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"PDF téléchargé : {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            logger.error(f"Échec téléchargement Semantic Scholar PDF {paper.paper_id}: {e}")
            return None

    def clear_cache(self):
        _cache.clear()
