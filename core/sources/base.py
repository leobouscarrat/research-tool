import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests
from langchain_core.tools import BaseTool


@dataclass
class NormalizedPaper:
    """Format unifié pour tous les papiers, quelle que soit la source."""
    paper_id: str          # ID unique (arxiv_id ou ss_<paperId>)
    title: str
    authors: str
    published: str
    abstract: str
    source: str            # "arxiv" | "semantic_scholar"
    pdf_url: str | None    # URL du PDF si disponible
    external_url: str      # Lien vers la page du papier
    raw: object            # Objet brut original (pour téléchargement)

    # Champs ajoutés par le scoring
    relevance_score: int = 0
    justification: str = "—"

    def to_llm_dict(self) -> dict:
        """Version compressée pour le contexte LLM."""
        return {
            "id": self.paper_id,
            "t": self.title,
            "a": self.authors,
            "y": self.published,
            "s": self.abstract[:200],
            "source": self.source
        }

    def to_display_dict(self) -> dict:
        """Version complète pour l'affichage Streamlit."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "abstract": self.abstract,
            "source": self.source,
            "pdf_url": self.pdf_url,
            "external_url": self.external_url,
            "relevance_score": self.relevance_score,
            "justification": self.justification,
        }


_RETRY_STATUS = {429, 503}
_BACKOFF_SECONDS = [5, 15, 30, 60]


class PaperSource(ABC):
    """Interface commune pour toutes les sources de papiers."""

    @staticmethod
    def request_with_retry(url: str, **kwargs) -> requests.Response:
        """GET avec backoff exponentiel sur 429 et 503."""
        from core.logger import logger
        for attempt, wait in enumerate(_BACKOFF_SECONDS, start=1):
            response = requests.get(url, **kwargs)
            if response.status_code not in _RETRY_STATUS:
                return response
            retry_after = int(response.headers.get("Retry-After", wait))
            logger.warning(
                f"HTTP {response.status_code} depuis {url} "
                f"(tentative {attempt}/{len(_BACKOFF_SECONDS)}) — attente {retry_after}s"
            )
            time.sleep(retry_after)
        return requests.get(url, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant de la source ex: 'arxiv'"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Nom affiché ex: 'ArXiv'"""
        pass

    @property
    @abstractmethod
    def tool(self) -> BaseTool:
        """Tool LangChain exposé à l'agent."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> NormalizedPaper | None:
        """Récupère un papier depuis le cache local."""
        pass

    @abstractmethod
    def download_pdf(self, paper: NormalizedPaper, dest_dir: str) -> str | None:
        """Télécharge le PDF et retourne le chemin local."""
        pass

    def clear_cache(self):
        """Vide le cache local."""
        pass
