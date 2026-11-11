import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "data"
DB_DIR = "rag/chroma_db"


def ingest_pdfs():

    documents = []

    for filename in os.listdir(DATA_DIR):

        if filename.lower().endswith(".pdf"):

            path = os.path.join(DATA_DIR, filename)

            print(f"Loading: {filename}")

            loader = PyPDFLoader(path)

            docs = loader.load()

            documents.extend(docs)

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

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    vectorstore.add_documents(chunks)

    print("Documents successfully stored in Chroma.")


if __name__ == "__main__":
    ingest_pdfs()