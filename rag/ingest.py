import os
import uuid
import chromadb
from typing import Dict, Any, Optional, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.llm import get_embeddings

DATA_DIR = "data"

# Pure In-Memory Chroma Client (Immune to SQLite disk locks & cloud filesystem error code 14)
_CHROMA_CLIENT = chromadb.EphemeralClient()
_GLOBAL_VECTORSTORE: Optional[Chroma] = None


def get_vectorstore(
    embeddings_provider: str = "auto",
    ollama_base_url: str = "http://localhost:11434"
) -> Optional[Chroma]:
    """
    Returns the active in-memory vectorstore or None if uninitialized.
    """
    global _GLOBAL_VECTORSTORE
    return _GLOBAL_VECTORSTORE


def ingest_pdfs(
    data_dir: str = DATA_DIR,
    pdf_paths: Optional[List[str]] = None,
    embeddings_provider: str = "auto",
    ollama_base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Ingests and embeds PDF research papers into an isolated in-memory ChromaDB collection.
    Returns metadata summary dictionary for UI display.
    """
    global _CHROMA_CLIENT, _GLOBAL_VECTORSTORE

    target_paths = []
    if pdf_paths:
        target_paths = [p for p in pdf_paths if os.path.exists(p) and p.lower().endswith(".pdf")]
    else:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        for filename in sorted(os.listdir(data_dir)):
            if filename.lower().endswith(".pdf"):
                target_paths.append(os.path.join(data_dir, filename))

    documents = []
    processed_files = []

    for path in target_paths:
        filename = os.path.basename(path)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            processed_files.append({
                "name": filename,
                "pages": len(docs),
                "size_kb": round(os.path.getsize(path) / 1024, 1)
            })
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        return {
            "status": "empty",
            "files_count": 0,
            "pages_count": 0,
            "chunks_count": 0,
            "files": []
        }

    # Split documents into semantic chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Initialize embeddings (ONNX all-MiniLM-L6-v2 or Ollama)
    embeddings = get_embeddings(
        provider=embeddings_provider,
        base_url=ollama_base_url
    )

    # Create a fresh isolated collection in memory
    unique_col = f"research_{uuid.uuid4().hex[:8]}"
    _GLOBAL_VECTORSTORE = Chroma(
        client=_CHROMA_CLIENT,
        collection_name=unique_col,
        embedding_function=embeddings
    )

    _GLOBAL_VECTORSTORE.add_documents(chunks)

    return {
        "status": "success",
        "files_count": len(processed_files),
        "pages_count": len(documents),
        "chunks_count": len(chunks),
        "files": processed_files
    }


if __name__ == "__main__":
    result = ingest_pdfs()
    print("Ingest result:", result)
