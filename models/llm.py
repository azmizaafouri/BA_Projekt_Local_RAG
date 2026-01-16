# models/llm.py
"""
LLaMA-Modell über Ollama.
"""

from langchain_ollama import OllamaLLM
from config import LLM_MODEL_NAME, TEMPERATURE


def get_llm() -> OllamaLLM:
    """
    Initialisiert das LLM über Ollama.
    """
    llm = OllamaLLM(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
    )
    return llm
