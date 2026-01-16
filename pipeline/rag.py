# pipeline/rag.py
from typing import Dict, Any, Optional

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from config import RETRIEVER_K
from models.llm import get_llm
from models.embeddings import get_embedding_model
from database.vector_store import load_vector_store


ROLE_INSTRUCTIONS: Dict[str, str] = {
    "default": (
        "Antworte neutral, präzise und gut strukturiert. "
        "Nutze Fachbegriffe nur, wenn sie im Kontext vorkommen."
    ),
    "technician": (
        "Antworte technisch präzise und detailreich. "
        "Gib wenn möglich konkrete Schritte, Parameter oder Formeln an."
    ),
    "manager": (
        "Antworte kurz und verständlich, mit Fokus auf Nutzen, Risiken und Entscheidungen. "
        "Vermeide unnötige technische Details."
    ),
}


def build_qa_chain(topic: Optional[str] = None, role: str = "default") -> RetrievalQA:
    """
    Erzeugt eine themen- und rollenabhängige RetrievalQA-Chain.
    """
    llm = get_llm()
    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    search_kwargs: Dict[str, Any] = {
        "k": RETRIEVER_K,
        "fetch_k": 10,
    }

    # Nur filtern, wenn ein echtes Topic gewählt wurde
    if topic is not None:
        search_kwargs["filter"] = {"topic": topic}

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    role_text = ROLE_INSTRUCTIONS.get(role, ROLE_INSTRUCTIONS["default"])

    template = f"""
Du bist ein Assistent für interne Dokumente (Richtlinien, Vorschriften, technische Dokus).

Rolle des Nutzers / Antwortstil:
{role_text}

Nutze AUSSCHLIESSLICH die bereitgestellten Kontexte, um die Frage zu beantworten.
Wenn die Informationen nicht ausreichen, sage klar, dass du die Frage mit den
vorhandenen Dokumenten nicht sicher beantworten kannst.

Kontexte:
{{context}}

Frage:
{{question}}

Antworte klar und strukturiert.
"""
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def ask_question(chain: RetrievalQA, query: str) -> Dict[str, Any]:
    """
    Führt eine Anfrage gegen die RAG-Chain aus.
    """
    return chain.invoke({"query": query})
