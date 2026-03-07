import json
from pathlib import Path
from datetime import datetime

THEMES_FILE = Path("./data/themes.json")
THEMES_DIR = Path("./data/themes")

def _load() -> dict:
    if not THEMES_FILE.exists():
        THEMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        THEMES_FILE.write_text("{}")
    return json.loads(THEMES_FILE.read_text())

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
        "searches": []
    }
    _save(data)
    return True

def delete_theme(slug: str):
    import shutil
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
        data[slug]["paper_count"] += papers_added
        data[slug]["searches"].append({
            "keywords": keywords,
            "papers_added": papers_added,
            "date": datetime.now().isoformat()
        })
        _save(data)