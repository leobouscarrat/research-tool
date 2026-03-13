import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from core.paper_fetcher import download_and_index
from core.research_agent import run_research_agent_turn
from core.theme_manager import get_paper_ids


def render(active_slug: str | None, themes: dict):
    st.header("📥 Recherche ArXiv")

    if not active_slug:
        st.warning("Crée un thème d'abord dans l'onglet Gestion.")
        return

    st.caption(f"Thème actif : **{themes[active_slug]['name']}**")

    if themes[active_slug]["searches"]:
        with st.expander("📋 Recherches précédentes"):
            for search in reversed(themes[active_slug]["searches"]):
                date = datetime.fromisoformat(search["date"]).strftime("%d/%m/%Y")
                st.write(f"- `{search['keywords']}` — {search['papers_added']} papiers ({date})")

    # ── Init session state ────────────────────────────────────────────────────
    search_chat_key = f"search_chat_{active_slug}"
    search_agent_messages_key = f"search_agent_msgs_{active_slug}"

    if search_chat_key not in st.session_state:
        st.session_state[search_chat_key] = []
    if search_agent_messages_key not in st.session_state:
        st.session_state[search_agent_messages_key] = []

    # ── Header + bouton reset ─────────────────────────────────────────────────
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.subheader("🤖 Chat de recherche")
    with col_reset:
        if st.button("🔄 Reset", help="Recommencer une nouvelle recherche"):
            st.session_state[search_chat_key] = []
            st.session_state[search_agent_messages_key] = []
            st.session_state.pop("suggested_papers", None)
            st.session_state.pop("paper_selections", None)
            st.rerun()

    # ── Sélection des sources ─────────────────────────────────────────────────────
    st.subheader("📡 Sources")
    col_arxiv, col_semantic, col_score = st.columns([1, 1, 2])
    with col_arxiv:
        use_arxiv = st.checkbox("ArXiv", value=True)
    with col_semantic:
        use_semantic = st.checkbox("Semantic Scholar", value=False)
    with col_score:
        min_score = st.slider(
            "Score minimum",
            min_value=1, max_value=10, value=5,
            help="Les papiers en dessous de ce score seront exclus"
        )

    active_sources = []
    if use_arxiv:
        active_sources.append("arxiv")
    if use_semantic:
        active_sources.append("semantic_scholar")

    if not active_sources:
        st.warning("Sélectionne au moins une source.")

    # ── Historique du chat ────────────────────────────────────────────────────
    for msg in st.session_state[search_chat_key]:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "steps":
                for s in msg["steps"]:
                    if s["type"] == "tool_call":
                        st.write(f"🔍 Recherche ArXiv : `{s['args'].get('query', '')}`")
                    elif s["type"] == "tool_result":
                        st.write(f"📄 {s['count']} papiers récupérés")
            else:
                st.write(msg["content"])

    # ── Input utilisateur ─────────────────────────────────────────────────────
    if not st.session_state.get("suggested_papers"):
    
        # Si le dernier message est une question de l'agent, on l'affiche en évidence
        last_messages = st.session_state[search_chat_key]
        agent_is_waiting = (
            last_messages
            and last_messages[-1]["role"] == "assistant"
            and last_messages[-1].get("type") != "steps"
        )

        if agent_is_waiting:
            st.info(f"💬 **L'agent attend ta réponse :** {last_messages[-1]['content']}")

        placeholder = "Réponds à l'agent..." if agent_is_waiting else "Décris ton sujet de recherche..."

        if user_input := st.chat_input(placeholder):
            st.session_state[search_chat_key].append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.write(user_input)

            st.session_state[search_agent_messages_key].append(
                HumanMessage(content=user_input)
            )

            steps_collected = []

            with st.chat_message("assistant"):
                steps_placeholder = st.empty()

                def steps_callback(step):
                    steps_collected.append(step)
                    with steps_placeholder.container():
                        for s in steps_collected:
                            if s["type"] == "phase":
                                st.write(s["label"])
                            elif s["type"] == "tool_result":
                                source_label = "Semantic Scholar" if s.get("source") == "semantic_scholar_search" else "ArXiv"
                                st.write(f"📄 {s['count']} papiers récupérés via {source_label}")
                            elif s["type"] == "scoring_progress":
                                current, total = s["current"], s["total"]
                                st.write(f"🧠 Notation : {current}/{total} papiers")
                                st.progress(current / total)

                with st.spinner(""):
                    is_first_message = len(st.session_state[search_agent_messages_key]) == 1

                    response_text, papers, updated_msgs = run_research_agent_turn(
                        st.session_state[search_agent_messages_key],
                        existing_paper_ids=get_paper_ids(active_slug),
                        min_score=min_score,
                        sources=active_sources,
                        steps_callback=steps_callback,
                        fresh_start=is_first_message
                    )
                st.session_state[search_agent_messages_key] = updated_msgs

                if response_text:
                    steps_placeholder.empty()
                    st.write(response_text)
                    st.session_state[search_chat_key].append({
                        "role": "assistant",
                        "content": response_text
                    })

                elif papers:
                    st.session_state[search_chat_key].append({
                        "role": "assistant",
                        "type": "steps",
                        "steps": steps_collected,
                        "content": ""
                    })
                    st.session_state["suggested_papers"] = papers
                    st.session_state["paper_selections"] = {
                        p.paper_id: True for p in papers
                    }
                    st.rerun()

    # ── Propositions de papiers ───────────────────────────────────────────────────
    if "suggested_papers" in st.session_state and st.session_state["suggested_papers"]:
        suggested = st.session_state["suggested_papers"]
        selections = st.session_state["paper_selections"]

        st.markdown("---")

        col_title, col_accept_all, col_reject_all = st.columns([4, 1, 1])
        with col_title:
            selected_count = sum(1 for v in selections.values() if v)
            st.subheader(f"📋 {len(suggested)} papiers proposés — {selected_count} sélectionnés")
        with col_accept_all:
            if st.button("✅ Tout accepter", use_container_width=True):
                for arxiv_id in selections:
                    st.session_state["paper_selections"][arxiv_id] = True
                st.rerun()
        with col_reject_all:
            if st.button("❌ Tout refuser", use_container_width=True):
                for arxiv_id in selections:
                    st.session_state["paper_selections"][arxiv_id] = False
                st.rerun()

        st.markdown("---")

        for i, paper in enumerate(suggested):
            paper_id = paper.paper_id
            score = paper.relevance_score
            is_selected = selections.get(paper_id, True)

            col_toggle, col_info, col_score = st.columns([1, 7, 1])

            with col_toggle:
                toggle_label = "✅" if is_selected else "⬜"
                if st.button(toggle_label, key=f"toggle_{paper_id}", use_container_width=True):
                    st.session_state["paper_selections"][paper_id] = not is_selected
                    st.rerun()

            with col_info:
                source_badge = "🟦 Semantic Scholar" if paper.source == "semantic_scholar" else "🟧 ArXiv"
                with st.expander(f"{paper.title}"):
                    st.caption(f"{source_badge}  ·  **Auteurs :** {paper.authors}  ·  **Publié :** {paper.published}  ·  `{paper_id}`")
                    st.write(paper.abstract)
                    st.info(f"💡 {paper.justification}")
                    st.link_button("🔗 Voir le papier", paper.external_url)

            with col_score:
                try:
                    score_int = int(score)
                except (ValueError, TypeError):
                    score_int = 0
                score_color = "🟢" if score_int >= 8 else "🟡" if score_int >= 6 else "🔴"
                st.markdown(
                    f"<div style='text-align:center; padding-top: 8px'>{score_color}<br><small>{score_int}/10</small></div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

        # Bouton indexation
        selected_papers = [
            p for p in suggested
            if st.session_state["paper_selections"].get(p.paper_id, False)
        ]
        col_count, col_button = st.columns([3, 2])
        with col_count:
            st.write(f"**{len(selected_papers)} papier(s) sélectionné(s)**")
        with col_button:
            if st.button("📥 Indexer la sélection", type="primary", use_container_width=True, disabled=not selected_papers):
                with st.spinner(f"Téléchargement et indexation de {len(selected_papers)} papiers..."):
                    count = download_and_index(active_slug, selected_papers)
                st.success(f"✅ {count} papiers indexés !")
                st.session_state.pop("suggested_papers", None)
                st.session_state.pop("paper_selections", None)
                st.session_state[search_chat_key] = []
                st.session_state[search_agent_messages_key] = []
                st.rerun()
