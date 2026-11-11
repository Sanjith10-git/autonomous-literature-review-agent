import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings

class ChromaDefaultEmbeddings(Embeddings):
    """
    Zero-dependency LangChain Embeddings wrapper around Chroma's built-in 
    ONNX all-MiniLM-L6-v2 embedding model.
    Runs locally on CPU in memory without requiring Ollama or external API keys.
    """
    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, vec)) for vec in self._ef(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [list(map(float, vec)) for vec in self._ef([text])][0]


def get_embeddings(provider: str = "auto", base_url: str = "http://localhost:11434") -> Embeddings:
    """
    Returns the appropriate embeddings model.
    If Ollama is explicitly selected, attempts OllamaEmbeddings.
    Otherwise falls back smoothly to built-in ONNX embeddings for cloud zero-setup.
    """
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=base_url
            )
        except Exception:
            pass

    return ChromaDefaultEmbeddings()


def get_llm(
    provider: str = "groq",
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    model: Optional[str] = None,
    temperature: float = 0.1
):
    """
    Factory function returning the active LLM based on user selection or cloud environment.
    Supports:
      - Groq Cloud: Free, blazing-fast Llama 3 models (ideal for Streamlit Community Cloud)
      - Ollama: Local workstation execution (ideal for private local workflows)
    """
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        # Fallback to environment variable or standard Groq key if set
        key = api_key or os.environ.get("GROQ_API_KEY")
        target_model = model or "llama-3.3-70b-versatile"

        if not key:
            raise ValueError(
                "Groq API Key is required. Enter your free API key in the sidebar or configure GROQ_API_KEY in Streamlit Secrets."
            )

        return ChatGroq(
            groq_api_key=key,
            model_name=target_model,
            temperature=temperature
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        target_model = model or "llama3.2"

        return ChatOllama(
            model=target_model,
            base_url=base_url,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
