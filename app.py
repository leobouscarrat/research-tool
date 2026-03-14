from datetime import datetime

import streamlit as st

from core.theme_manager import (
    create_theme,
    delete_theme,
    get_paper_count,
    list_themes,
    rename_theme,
    sync_papers_from_disk,
)
from tabs import bibliography, papers, qa

st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Research Assistant")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Thème actif")
themes = list_themes()
for slug in themes:
    sync_papers_from_disk(slug)
themes = list_themes()  # Recharge après sync

if not themes:
    st.sidebar.info("Aucun thème. Crée-en un dans l'onglet Gestion.")
    active_slug = None
else:
    theme_options = {v["name"]: k for k, v in themes.items()}
    selected_name = st.sidebar.selectbox("Choisir un thème", list(theme_options.keys()))
    active_slug = theme_options[selected_name]
    theme = themes[active_slug]

    st.sidebar.markdown("---")
    st.sidebar.metric("📄 Papiers indexés", get_paper_count(active_slug))
    if theme["last_search"]:
        last = datetime.fromisoformat(theme["last_search"]).strftime("%d/%m/%Y %H:%M")
        st.sidebar.caption(f"Dernière recherche : {last}")

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab_bibliography, tab_papers, tab_qa, tab_themes = st.tabs([
    "📥 Bibliographie", "📚 Papiers indexés", "💬 Questions", "⚙️ Gestion des thèmes"
])

with tab_bibliography:
    bibliography.render(active_slug, themes)

with tab_papers:
    papers.render(active_slug, themes)

# ── TAB : Q&A ─────────────────────────────────────────────────────────────────
with tab_qa:
    qa.render(active_slug, themes)

# ── TAB : Gestion des thèmes ──────────────────────────────────────────────────
with tab_themes:
    st.header("⚙️ Gestion des thèmes")

    with st.expander("➕ Créer un nouveau thème", expanded=not themes):
        new_theme_name = st.text_input("Nom du thème", placeholder="ex: Computer Vision")
        if st.button("Créer", type="primary"):
            if not new_theme_name:
                st.warning("Entre un nom.")
            elif create_theme(new_theme_name):
                st.success(f"✅ Thème '{new_theme_name}' créé !")
                st.rerun()
            else:
                st.error("Un thème avec ce nom existe déjà.")

    st.markdown("---")
    st.subheader("Thèmes existants")
    themes = list_themes()

    if not themes:
        st.info("Aucun thème pour l'instant.")
    else:
        for slug, theme in themes.items():
            with st.expander(f"📁 {theme['name']} — {theme['paper_count']} papiers"):
                col_info, col_rename, col_delete = st.columns([2, 1, 1])

                with col_info:
                    created = datetime.fromisoformat(theme["created_at"]).strftime("%d/%m/%Y")
                    st.caption(f"Créé le {created}")
                    if theme["last_search"]:
                        last = datetime.fromisoformat(theme["last_search"]).strftime("%d/%m/%Y %H:%M")
                        st.caption(f"Dernière recherche : {last}")
                    if theme["searches"]:
                        st.caption(f"Recherches effectuées : {len(theme['searches'])}")

                with col_rename:
                    new_name = st.text_input("Renommer", key=f"rename_{slug}", placeholder=theme["name"])
                    if st.button("✏️ Renommer", key=f"btn_rename_{slug}"):
                        if new_name:
                            rename_theme(slug, new_name)
                            st.rerun()

                with col_delete:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Supprimer", key=f"del_{slug}", type="secondary"):
                        delete_theme(slug)
                        st.rerun()
