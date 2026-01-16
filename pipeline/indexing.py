# pipeline/indexing.py
"""
Indexierung: PDFs laden, chunking, Embeddings, Speichern in ChromaDB.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOPICS
from models.embeddings import get_embedding_model
from database.vector_store import create_vector_store, reset_vector_store


def load_pdfs(pdf_root: Path | None = None) -> List[Document]:
    """
    Lädt PDFs aus themenspezifischen Unterordnern unterhalb von PDF_DIR.

    Erwartete Struktur:
    data/manual/*.pdf
    data/konstruktion/*.pdf

    Setzt pro Seite:
    - source: Dateiname
    - topic: technischer Topic-Name (manual / konstruktion)
    """
    if pdf_root is None:
        pdf_root = PDF_DIR

    if not pdf_root.exists():
        raise RuntimeError(f"PDF-Verzeichnis {pdf_root} existiert nicht.")

    docs: List[Document] = []

    # Wir ignorieren den Default-Eintrag (None) und laden nur echte Topics
    real_topics = {label: tid for label, tid in TOPICS.items() if tid is not None}

    for display_name, topic_id in real_topics.items():
        topic_dir = pdf_root / topic_id
        if not topic_dir.exists():
            raise RuntimeError(
                f"Erwarteter Themenordner fehlt: {topic_dir} (Thema: {display_name})"
            )

        pdf_paths = list(topic_dir.glob("*.pdf"))
        if not pdf_paths:
            raise RuntimeError(
                f"Keine PDFs gefunden in {topic_dir} (Thema: {display_name})"
            )

        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()

            for d in pdf_docs:
                d.metadata.setdefault("source", pdf_path.name)
                d.metadata["topic"] = topic_id  # <<< wichtig für Filter

            docs.extend(pdf_docs)

    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Zerlegt Dokumente in Text-Chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_index():
    """
    Kompletter Indexierungs-Workflow:
    - bestehenden Vektorstore löschen
    - PDFs laden
    - Chunks erzeugen
    - Embeddings berechnen
    - ChromaDB neu aufbauen
    """
    print("Bereinige vorhandenen Vektorstore (falls vorhanden) ...")
    reset_vector_store()

    print(f"Lade PDFs aus {PDF_DIR} ...")
    docs = load_pdfs()
    print(f"Seiten insgesamt: {len(docs)}")

    chunks = split_documents(docs)
    print(f"Chunks insgesamt: {len(chunks)}")

    embeddings = get_embedding_model()
    print("ChromaDB wird neu erstellt ...")
    create_vector_store(chunks, embeddings)
    print("Indexierung abgeschlossen.")



if __name__ == "__main__":
    build_index()
