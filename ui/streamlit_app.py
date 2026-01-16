# ui/streamlit_app.py
"""
Streamlit-Frontend für die lokale RAG-Anwendung.
"""

import textwrap
from typing import List, Dict, Any

import streamlit as st

from config import APP_TITLE, TOPICS, ROLES, DEFAULT_TOPIC, DEFAULT_ROLE
from pipeline.rag import build_qa_chain, ask_question


def _init_session_state() -> None:
    """
    Initialisiert Session-State Variablen.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, Any]] = []

    if "active_topic" not in st.session_state:
        st.session_state["active_topic"] = DEFAULT_TOPIC  # None = alle Dokumente

    if "active_role" not in st.session_state:
        st.session_state["active_role"] = DEFAULT_ROLE

    if "qa_chain" not in st.session_state or st.session_state["qa_chain"] is None:
        st.session_state["qa_chain"] = build_qa_chain(
            topic=st.session_state["active_topic"],
            role=st.session_state["active_role"],
        )


def _dedup_source_documents(source_documents: List[Any]) -> List[Any]:
    """Entfernt doppelte Quellenstellen."""
    seen = set()
    unique_docs: List[Any] = []

    for d in source_documents:
        metadata = getattr(d, "metadata", {}) or {}
        source = metadata.get("source")
        page = metadata.get("page")
        content_prefix = (getattr(d, "page_content", "") or "")[:200]

        key = (source, page, content_prefix)
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


def _render_sources(source_documents: List[Any]) -> None:
    """Zeigt Quellenstellen dedupliziert in einem Expander."""
    if not source_documents:
        return

    source_documents = _dedup_source_documents(source_documents)

    with st.expander("Verwendete Dokumentstellen anzeigen"):
        for i, doc in enumerate(source_documents, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "Unbekannte Quelle")
            page = meta.get("page", "n/a")
            topic = meta.get("topic", "n/a")

            st.markdown(f"**Quelle {i}:** {source} (Thema: {topic}), Seite {page}")
            st.caption(textwrap.shorten(doc.page_content, width=350, placeholder=" ..."))


def _sidebar_controls() -> None:
    """Sidebar UI: Thema + Rolle auswählen; bei Änderung Chain neu bauen."""
    st.sidebar.header("Einstellungen")

    # --- Thema ---
    topic_labels = list(TOPICS.keys())
    # aktuelles label finden
    current_topic = st.session_state["active_topic"]
    current_topic_label = [k for k, v in TOPICS.items() if v == current_topic][0]

    selected_topic_label = st.sidebar.selectbox(
        "Thema",
        options=topic_labels,
        index=topic_labels.index(current_topic_label),
    )
    new_topic = TOPICS[selected_topic_label]  # None bei Default

    # --- Rolle ---
    role_labels = list(ROLES.keys())
    current_role = st.session_state["active_role"]
    current_role_label = [k for k, v in ROLES.items() if v == current_role][0]

    selected_role_label = st.sidebar.selectbox(
        "Rolle",
        options=role_labels,
        index=role_labels.index(current_role_label),
    )
    new_role = ROLES[selected_role_label]

    reset_chat = st.sidebar.checkbox("Chat beim Wechsel zurücksetzen", value=True)

    # Nur bei Änderung neu bauen
    if new_topic != st.session_state["active_topic"] or new_role != st.session_state["active_role"]:
        st.session_state["active_topic"] = new_topic
        st.session_state["active_role"] = new_role

        st.session_state["qa_chain"] = build_qa_chain(topic=new_topic, role=new_role)

        if reset_chat:
            st.session_state["messages"] = []

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Hinweis: 'Default (alle Dokumente)' nutzt keinen Topic-Filter und durchsucht alle PDFs."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    _init_session_state()
    _sidebar_controls()

    st.title(APP_TITLE)
    st.markdown(
        "Diese Anwendung beantwortet Fragen zu lokal gespeicherten PDF-Dokumenten "
        "mittels Retrieval-Augmented Generation (RAG) auf Basis von Ollama, "
        "ChromaDB und LangChain."
    )

    # Aktive Auswahl anzeigen
    active_topic = st.session_state["active_topic"]
    active_role = st.session_state["active_role"]

    active_topic_label = [k for k, v in TOPICS.items() if v == active_topic][0]
    active_role_label = [k for k, v in ROLES.items() if v == active_role][0]

    st.markdown(f"**Aktives Thema:** {active_topic_label}  \n**Aktive Rolle:** {active_role_label}")

    # Chatverlauf rendern
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # Chat input
    user_input = st.chat_input("Stelle eine Frage zu den geladenen Dokumenten …")

    if user_input:
        # User Message speichern & anzeigen
        st.session_state["messages"].append({"role": "user", "content": user_input, "sources": None})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Antwort generieren
        with st.chat_message("assistant"):
            with st.spinner("Lokale Antwort wird generiert …"):
                response = ask_question(st.session_state["qa_chain"], user_input)

            answer = response.get("result", "")
            sources = response.get("source_documents", [])

            st.markdown(answer)
            _render_sources(sources)

        # Assistant Message speichern
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
