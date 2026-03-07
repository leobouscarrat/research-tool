import streamlit as st
from core.theme_manager import list_themes, create_theme, delete_theme, rename_theme
from core.arxiv_fetcher import search_and_index
from core.rag_agent import ask
from datetime import datetime

st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Research Assistant")

# ── SIDEBAR : Sélection du thème ─────────────────────────────────────────────
st.sidebar.header("Thème actif")
themes = list_themes()

if not themes:
    st.sidebar.info("Aucun thème. Crée-en un dans l'onglet Gestion.")
    active_slug = None
else:
    theme_options = {v["name"]: k for k, v in themes.items()}
    selected_name = st.sidebar.selectbox("Choisir un thème", list(theme_options.keys()))
    active_slug = theme_options[selected_name]
    theme = themes[active_slug]

    st.sidebar.markdown("---")
    st.sidebar.metric("📄 Papiers indexés", theme["paper_count"])
    if theme["last_search"]:
        last = datetime.fromisoformat(theme["last_search"]).strftime("%d/%m/%Y %H:%M")
        st.sidebar.caption(f"Dernière recherche : {last}")

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab_bibliography, tab_qa, tab_themes = st.tabs(["📥 Bibliographie", "💬 Questions", "⚙️ Gestion des thèmes"])

# ── TAB : Recherche ArXiv ─────────────────────────────────────────────────────
with tab_bibliography:
    st.header("📥 Recherche ArXiv")
    if not active_slug:
        st.warning("Crée un thème d'abord dans l'onglet Gestion.")
    else:
        st.caption(f"Thème actif : **{themes[active_slug]['name']}**")

        if themes[active_slug]["searches"]:
            with st.expander("📋 Recherches précédentes"):
                for s in reversed(themes[active_slug]["searches"]):
                    date = datetime.fromisoformat(s["date"]).strftime("%d/%m/%Y")
                    st.write(f"- `{s['keywords']}` — {s['papers_added']} papiers ({date})")

        col_keywords, col_max_results = st.columns([3, 1])
        with col_keywords:
            keywords = st.text_input("Mots-clés", placeholder="ex: retrieval augmented generation LLM")
        with col_max_results:
            max_results = st.number_input("Nb papiers", min_value=1, max_value=50, value=10)

        if st.button("🔍 Rechercher et indexer", type="primary"):
            if not keywords:
                st.warning("Entre des mots-clés d'abord.")
            else:
                with st.spinner(f"Indexation de {max_results} papiers..."):
                    results = search_and_index(active_slug, keywords, max_results)
                st.success(f"✅ {len(results)} papiers indexés !")
                for paper in results:
                    with st.expander(f"📄 {paper['title']}"):
                        st.write(f"**Auteurs :** {paper['authors']}")
                        st.write(f"**Publié :** {paper['published']}")
                        st.write(f"**ArXiv ID :** `{paper['arxiv_id']}`")
                        st.write(paper['summary'])

# ── TAB : Q&A ─────────────────────────────────────────────────────────────────
with tab_qa:
    st.header("💬 Questions")
    if not active_slug:
        st.warning("Crée un thème d'abord dans l'onglet Gestion.")
    elif themes[active_slug]["paper_count"] == 0:
        st.warning("Indexe des papiers d'abord dans l'onglet Bibliographie.")
    else:
        st.caption(f"Thème actif : **{themes[active_slug]['name']}**")

        history_key = f"history_{active_slug}"
        if history_key not in st.session_state:
            st.session_state[history_key] = []

        if st.session_state[history_key]:
            if st.button("🗑️ Effacer l'historique"):
                st.session_state[history_key] = []
                st.rerun()

        for msg in st.session_state[history_key]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question := st.chat_input("Ex: Quelles sont les principales approches de RAG ?"):
            st.session_state[history_key].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Recherche dans les papiers..."):
                    answer = ask(question, active_slug, st.session_state[history_key])
                st.write(answer)
                st.session_state[history_key].append({"role": "assistant", "content": answer})

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