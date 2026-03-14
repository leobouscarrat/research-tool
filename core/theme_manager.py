import json
import shutil
from datetime import datetime
from pathlib import Path

from core.logger import logger

THEMES_FILE = Path("./data/themes.json")
THEMES_DIR = Path("./data/themes")

def _load() -> dict:
    if not THEMES_FILE.exists():
        THEMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        THEMES_FILE.write_text("{}")
        return {}
    content = THEMES_FILE.read_text().strip()
    if not content:
        THEMES_FILE.write_text("{}")
        return {}
    return json.loads(content)

def _save(data: dict):
    THEMES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def list_themes() -> dict:
    return _load()

def create_theme(name: str) -> bool:
    data = _load()
    slug = name.lower().replace(" ", "_")
    if slug in data:
        return False
    theme_dir = THEMES_DIR / slug
    (theme_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    data[slug] = {
        "name": name,
        "slug": slug,
        "created_at": datetime.now().isoformat(),
        "last_search": None,
        "paper_count": 0,
        "searches": [],
        "chat_history": [],
        "papers": []
    }
    _save(data)
    return True

def delete_theme(slug: str):
    data = _load()
    if slug in data:
        del data[slug]
        _save(data)
    theme_dir = THEMES_DIR / slug
    if theme_dir.exists():
        shutil.rmtree(theme_dir)

def rename_theme(slug: str, new_name: str):
    data = _load()
    if slug in data:
        data[slug]["name"] = new_name
        _save(data)

def update_theme_stats(slug: str, keywords: str, papers_added: int):
    data = _load()
    if slug in data:
        data[slug]["last_search"] = datetime.now().isoformat()
        data[slug]["searches"].append({
            "keywords": keywords,
            "papers_added": papers_added,
            "date": datetime.now().isoformat()
        })
        _save(data)

def add_paper(slug: str, paper: dict):
    """Ajoute un papier à la liste persistée du thème."""
    data = _load()
    if slug not in data:
        return
    existing_ids = {p["paper_id"] for p in data[slug].get("papers", [])}
    if paper["paper_id"] not in existing_ids:
        data[slug].setdefault("papers", []).append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "published": paper["published"],
            "abstract": paper["abstract"],
            "source": paper.get("source", "arxiv"),
            "external_url": paper.get("external_url", ""),
            "added_at": datetime.now().isoformat()
        })
        _save(data)

def get_papers(slug: str) -> list[dict]:
    """Retourne la liste des papiers indexés pour un thème."""
    data = _load()
    return data.get(slug, {}).get("papers", [])

def get_paper_ids(slug: str) -> set[str]:
    """Retourne les paper_id déjà indexés pour éviter les doublons."""
    return {p["paper_id"] for p in get_papers(slug)}

def get_paper_count(slug: str) -> int:
    """Compte les PDFs réellement présents sur disque."""
    pdf_dir = Path(f"./data/themes/{slug}/pdfs")
    if not pdf_dir.exists():
        return 0
    return len(list(pdf_dir.glob("*.pdf")))

def sync_papers_from_disk(slug: str):
    """
    Synchronise la liste papers[] avec les PDFs présents sur disque.
    Supprime les entrées sans PDF, ne touche pas aux entrées valides.
    """
    data = _load()
    if slug not in data:
        return

    pdf_dir = Path(f"./data/themes/{slug}/pdfs")
    pdf_ids = {p.stem for p in pdf_dir.glob("*.pdf")} if pdf_dir.exists() else set()

    current_papers = data[slug].get("papers", [])
    current_ids = {p["paper_id"] for p in current_papers}

    # Supprime les entrées dont le PDF n'existe plus
    synced = [p for p in current_papers if p["paper_id"] in pdf_ids]

    # Log les écarts
    removed = current_ids - pdf_ids
    if removed:
        logger.warning(f"sync_papers_from_disk : {len(removed)} entrées sans PDF supprimées : {removed}")

    data[slug]["papers"] = synced
    _save(data)

def save_chat_history(slug: str, history: list[dict]):
    data = _load()
    if slug in data:
        data[slug]["chat_history"] = history
        _save(data)

def load_chat_history(slug: str) -> list[dict]:
    data = _load()
    return data.get(slug, {}).get("chat_history", [])
