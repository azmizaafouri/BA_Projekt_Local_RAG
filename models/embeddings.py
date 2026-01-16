# models/embeddings.py
"""
Embedding-Modell mxbai-embed-large über Ollama.
"""

from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL_NAME


def get_embedding_model() -> OllamaEmbeddings:
    """
    Initialisiert das Embedding-Modell.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    return embeddings
