import streamlit as st
from core.rag_agent import ask
from core.theme_manager import save_chat_history, load_chat_history, get_paper_count


def render(active_slug: str | None, themes: dict):
    st.header("💬 Questions")

    if not active_slug:
        st.warning("Crée un thème d'abord dans l'onglet Gestion.")
        return

    if get_paper_count(active_slug) == 0:
        st.warning("Indexe des papiers d'abord dans l'onglet Bibliographie.")
        return

    st.caption(f"Thème actif : **{themes[active_slug]['name']}**")

    history_key = f"history_{active_slug}"
    if history_key not in st.session_state:
        st.session_state[history_key] = load_chat_history(active_slug)

    if st.session_state[history_key]:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state[history_key] = []
            save_chat_history(active_slug, [])
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
            save_chat_history(active_slug, st.session_state[history_key])

        st.rerun()