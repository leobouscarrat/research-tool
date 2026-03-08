from core.sources.arxiv import ArxivSource
from core.sources.semantic_scholar import SemanticScholarSource

AVAILABLE_SOURCES = {
    "arxiv": ArxivSource(),
    "semantic_scholar": SemanticScholarSource(),
}