import arxiv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.vector_store import get_vectorstore
from core.theme_manager import update_theme_stats

def search_and_index(slug: str, keywords: str, max_results: int = 10) -> list[dict]:
    pdf_dir = Path(f"./data/themes/{slug}/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=keywords,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    vectorstore = get_vectorstore(slug)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    results = []

    for paper in client.results(search):
        arxiv_id = paper.entry_id.split("/")[-1]
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            paper.download_pdf(dirpath=str(pdf_dir), filename=f"{arxiv_id}.pdf")

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors[:3]),
                "arxiv_id": arxiv_id,
                "published": str(paper.published.date()),
                "theme": slug,
            })

        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

        results.append({
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:3]),
            "published": str(paper.published.date()),
            "arxiv_id": arxiv_id,
            "summary": paper.summary[:300] + "...",
        })

    update_theme_stats(slug, keywords, len(results))
    return results