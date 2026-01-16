from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


PDF_DIR = BASE_DIR / "data"

# Persistenzverzeichnis für ChromaDB
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Modellnamen für Ollama
LLM_MODEL_NAME = "llama3.2"          
EMBEDDING_MODEL_NAME = "mxbai-embed-large"

# Chunking-Parameter
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

# Retrieval-Parameter
RETRIEVER_K = 5

# LLM-Temperatur (für konsistente Antworten laut Aufgabenstellung)
TEMPERATURE = 0.2

# Titel der Streamlit-App
APP_TITLE = "Lokale RAG-KI (Bachelorarbeit)"



TOPICS = {
    "Default (alle Dokumente)": None,     # wichtig: None = kein Filter
    "Benutzereinleitung": "Benutzereinleitung",
    "Fachliche_Beschreibung": "Fachliche_Beschreibung",
}

ROLES = {
    "Default": "default",
    "Techniker": "technician",
    "Manager": "manager",
}

DEFAULT_TOPIC = None
DEFAULT_ROLE = "default"
