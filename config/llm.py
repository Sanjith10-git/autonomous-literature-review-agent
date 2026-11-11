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


def get_embeddings(provider: str = "auto") -> Embeddings:
    """
    Returns Ollama embeddings only when explicitly requested AND Ollama is running.
    Falls back to Chroma's zero-dependency in-memory ONNX embeddings in all other cases.
    """
    # Only try Ollama embeddings when explicitly selected or auto-detected locally
    # AND no Groq key is available (meaning we're definitely in a local environment)
    if provider == "ollama" or (
        provider == "auto"
        and not get_groq_api_key()   # no cloud key → local-only environment
        and is_ollama_running()
    ):
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model="nomic-embed-text")
        except Exception:
            pass

    # Always safe fallback: in-process ONNX embeddings, no network required
    return ChromaDefaultEmbeddings()


def get_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "auto",
    temperature: float = 0
):
    """
    Resolves and returns the active ChatModel.

    Priority order (auto mode):
      1. Groq Cloud — if GROQ_API_KEY is available (env var or Streamlit secrets)
      2. Local Ollama — if reachable and no cloud key is configured
      3. Raise ConnectionError with actionable guidance
    """
    explicit_key = (api_key or "").strip()
    selected_provider = (provider or "auto").lower()

    # Explicit Ollama selection (never used on Streamlit Cloud)
    if selected_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.2", temperature=temperature)

    # Resolve cloud key (secrets.toml / env var / explicit argument)
    key = explicit_key or get_groq_api_key()

    # --- Groq Cloud (preferred when key is available) ---
    if key and selected_provider in ("auto", "groq"):
        from langchain_groq import ChatGroq
        candidates = [model] if model else []
        for c in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"):
            if c not in candidates:
                candidates.append(c)

        for candidate_model in candidates:
            try:
                return ChatGroq(
                    groq_api_key=key,
                    model_name=candidate_model,
                    temperature=temperature
                )
            except Exception:
                continue

    # --- Local Ollama fallback (auto mode, no cloud key) ---
    if selected_provider == "auto" and not key and is_ollama_running():
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.2", temperature=temperature)

    raise ConnectionError(
        "No active LLM backend found. "
        "On Streamlit Cloud: add GROQ_API_KEY to App Settings → Secrets. "
        "Locally: start Ollama with `ollama serve` (llama3.2 + nomic-embed-text)."
    )


def invoke_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "auto",
    temperature: float = 0
) -> str:
    """Invokes the active LLM and returns a plain string response."""
    llm = get_llm(
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature
    )
    try:
        response = llm.invoke(prompt)
    except Exception as err:
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
