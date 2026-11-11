import os
import uuid
import chromadb
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.llm import get_embeddings


DATA_DIR = "data"

_CHROMA_CLIENT = chromadb.EphemeralClient()
_VECTORSTORE: Optional[Chroma] = None


def get_vectorstore() -> Optional[Chroma]:
    """Returns the active in-memory Chroma vectorstore, if initialized."""
    return _VECTORSTORE


def ingest_pdfs(pdf_paths: Optional[List[str]] = None, provider: str = "auto"):
    """Ingests PDFs into an isolated in-memory Chroma collection."""
    global _VECTORSTORE

    target_paths = []
    if pdf_paths:
        target_paths = [p for p in pdf_paths if os.path.exists(p) and p.lower().endswith(".pdf")]
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        for filename in sorted(os.listdir(DATA_DIR)):
            if filename.lower().endswith(".pdf"):
                target_paths.append(os.path.join(DATA_DIR, filename))

    documents = []
    for path in target_paths:
        print(f"Loading: {os.path.basename(path)}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    if not documents:
        print("No PDF files found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    embeddings = get_embeddings(provider=provider)
    collection_name = f"research_documents_{uuid.uuid4().hex[:8]}"

    _VECTORSTORE = Chroma(
        client=_CHROMA_CLIENT,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    _VECTORSTORE.add_documents(chunks)

    print("Documents successfully stored in Chroma.")


if __name__ == "__main__":
    ingest_pdfs()
