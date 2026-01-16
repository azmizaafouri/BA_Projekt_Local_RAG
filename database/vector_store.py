# database/vector_store.py
"""
Verwaltung der Chroma-Vektor-Datenbank.
"""

from typing import List
from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR


def reset_vector_store() -> None:
    """
    Löscht den bestehenden ChromaDB-Ordner vollständig.
    """
    if CHROMA_PERSIST_DIR.exists() and CHROMA_PERSIST_DIR.is_dir():
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Bestehender Vektorstore unter {CHROMA_PERSIST_DIR} wurde gelöscht.")
    else:
        print(f"Kein bestehender Vektorstore unter {CHROMA_PERSIST_DIR} gefunden.")


def create_vector_store(docs: List[Document], embedding_model) -> Chroma:
    """
    Erzeugt eine neue ChromaDB aus Dokument-Chunks.
    """
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="engineering_docs",
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    return vector_store


def load_vector_store(embedding_model) -> Chroma:
    """
    Lädt eine bestehende ChromaDB von der Platte.
    """
    vector_store = Chroma(
        collection_name="engineering_docs",
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model,
    )
    return vector_store
