import os
import urllib.request
from typing import List, Optional
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around Chroma's built-in ONNX all-MiniLM-L6-v2 model."""

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, vec)) for vec in self._ef(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [list(map(float, vec)) for vec in self._ef([text])][0]


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Quickly checks whether a local Ollama daemon is reachable."""
    try:
        with urllib.request.urlopen(base_url, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_groq_api_key() -> str:
    """Safely resolves GROQ_API_KEY from environment variables or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    return ""


def get_embeddings() -> Embeddings:
    """
    Returns Ollama embeddings when a local Ollama daemon is running;
    otherwise uses Chroma's zero-dependency in-memory ONNX embeddings.
    """
    if is_ollama_running():
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model="nomic-embed-text")
        except Exception:
            pass

    return ChromaDefaultEmbeddings()


def get_llm(api_key: Optional[str] = None, temperature: float = 0):
    """
    Returns local ChatOllama when Ollama is running locally (unless an explicit
    cloud key override is needed), or ChatGroq on cloud deployments.
    """
    explicit_key = (api_key or "").strip()

    if not explicit_key and is_ollama_running():
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=temperature)

    key = explicit_key or get_groq_api_key()
    if key:
        from langchain_groq import ChatGroq
        for candidate_model in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"):
            try:
                return ChatGroq(
                    groq_api_key=key,
                    model_name=candidate_model,
                    temperature=temperature
                )
            except Exception:
                continue

    if is_ollama_running():
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=temperature)

    raise ConnectionError(
        "No active LLM backend found. Configure GROQ_API_KEY in Streamlit Secrets (or enter a Groq API key) "
        "for cloud execution, or start Ollama locally with `llama3.2`."
    )


def invoke_llm(prompt: str, api_key: Optional[str] = None, temperature: float = 0) -> str:
    """Invokes the active LLM and returns a plain string response."""
    model = get_llm(api_key=api_key, temperature=temperature)
    try:
        response = model.invoke(prompt)
    except Exception as err:
        # Fallback if selected Groq model is unavailable on this key tier
        err_text = str(err).lower()
        key = (api_key or "").strip() or get_groq_api_key()
        if key and ("model_not_found" in err_text or "not found" in err_text or "404" in err_text):
            from langchain_groq import ChatGroq
            fallback = ChatGroq(
                groq_api_key=key,
                model_name="qwen/qwen3.8-27b",
                temperature=temperature
            )
            response = fallback.invoke(prompt)
        else:
            raise

    return response.content if hasattr(response, "content") else str(response)
