import os
import shutil
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.llm import get_embeddings

DATA_DIR = "data"
DB_DIR = "rag/chroma_db"


def ingest_pdfs(
    data_dir: str = DATA_DIR,
    db_dir: str = DB_DIR,
    embeddings_provider: str = "auto",
    ollama_base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Ingests and embeds all PDF research papers found in data_dir into ChromaDB.
    Returns metadata summary dictionary for UI display.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    documents = []
    processed_files = []

    for filename in sorted(os.listdir(data_dir)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(data_dir, filename)
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

    # Initialize embeddings (auto chooses Ollama or built-in CPU ONNX for zero cloud crashes)
    embeddings = get_embeddings(
        provider=embeddings_provider,
        base_url=ollama_base_url
    )

    # Reset existing collection cleanly if re-indexing
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except Exception:
            pass

    vectorstore = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=db_dir
    )

    vectorstore.add_documents(chunks)

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
