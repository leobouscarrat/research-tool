import streamlit as st
from datetime import datetime
from core.theme_manager import get_papers, get_paper_count, sync_papers_from_disk


def render(active_slug: str | None, themes: dict):
    st.header("📚 Papiers indexés")

    if not active_slug:
        st.warning("Crée un thème d'abord dans l'onglet Gestion.")
        return

    papers = get_papers(active_slug)
    real_count = get_paper_count(active_slug)

    if real_count != len(papers):
        st.warning(f"⚠️ {real_count} PDFs sur disque mais {len(papers)} entrées en base — synchronisation en cours...")
        sync_papers_from_disk(active_slug)
        papers = get_papers(active_slug)

    if not papers:
        st.info("Aucun papier indexé pour ce thème. Utilise l'onglet Bibliographie pour en ajouter.")
        return

    st.caption(f"**{len(papers)} papier(s)** dans le thème *{themes[active_slug]['name']}*")

    # Barre de recherche locale
    search_query = st.text_input("🔎 Filtrer par titre ou auteur", placeholder="ex: transformer")

    filtered_papers = papers
    if search_query:
        query_lower = search_query.lower()
        filtered_papers = [
            p for p in papers
            if query_lower in p["title"].lower()
            or query_lower in p.get("authors", "").lower()
        ]
        st.caption(f"{len(filtered_papers)} résultat(s) pour « {search_query} »")

    st.markdown("---")

    for paper in sorted(filtered_papers, key=lambda p: p["published"], reverse=True):
        added_at = datetime.fromisoformat(paper["added_at"]).strftime("%d/%m/%Y")
        with st.expander(f"📄 {paper['title']}"):
            col_meta, col_date = st.columns([4, 1])
            with col_meta:
                st.write(f"**Auteurs :** {paper['authors']}")
                st.write(f"**Publié :** {paper['published']}")
                source = paper.get("source", "arxiv")
                external_url = paper.get("external_url") or f"https://arxiv.org/abs/{paper['paper_id']}"
                source_label = "Semantic Scholar" if source == "semantic_scholar" else "ArXiv"
                st.write(f"**ID :** `{paper['paper_id']}`")
                st.link_button(f"🔗 Voir sur {source_label}", external_url)
            with col_date:
                st.caption(f"Ajouté le {added_at}")
            st.markdown(f"**Abstract :** {paper['abstract']}")
