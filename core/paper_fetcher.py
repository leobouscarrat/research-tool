from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import logger
from core.sources import AVAILABLE_SOURCES
from core.sources.base import NormalizedPaper
from core.theme_manager import add_paper, update_theme_stats
from core.vector_store import get_vectorstore


def download_and_index(slug: str, papers: list[NormalizedPaper]) -> int:
    pdf_dir = Path(f"./data/themes/{slug}/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = get_vectorstore(slug)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    indexed_count = 0

    for paper in papers:
        # Tente le téléchargement PDF
        source = AVAILABLE_SOURCES.get(paper.source)
        pdf_path = source.download_pdf(paper, str(pdf_dir)) if source else None

        if pdf_path:
            # Indexe depuis le PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        else:
            # Fallback : indexe uniquement les métadonnées + abstract
            logger.warning(f"Pas de PDF pour {paper.paper_id}, indexation de l'abstract uniquement")
            docs = [Document(
                page_content=f"{paper.title}\n\n{paper.abstract}",
                metadata={}
            )]

        for doc in docs:
            doc.metadata.update({
                "title": paper.title,
                "authors": paper.authors,
                "paper_id": paper.paper_id,
                "published": paper.published,
                "source": paper.source,
            })

        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

        add_paper(slug, paper.to_display_dict())
        indexed_count += 1
        logger.info(f"Indexé : {paper.title[:60]} ({paper.source})")

    if indexed_count > 0:
        keywords = ", ".join(p.title for p in papers[:3])
        update_theme_stats(slug, keywords, indexed_count)

    return indexed_count
