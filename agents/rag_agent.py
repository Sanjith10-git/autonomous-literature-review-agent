from typing import Optional
from config.llm import invoke_llm
from rag.ingest import get_vectorstore


def rag_agent(question: str, api_key: Optional[str] = None) -> str:
    """Retrieves relevant chunks from ingested PDFs and answers using the active LLM."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "No relevant information found in the uploaded documents."

    documents = vectorstore.similarity_search(question, k=4)
    if not documents:
        return "No relevant information found in the uploaded documents."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a research assistant.

Answer the question using the research information below.

Research information:
{context}

Question:
{question}

Instructions:
- Give a clear answer.
- Use the provided research information.
- Do not invent facts.
- If the information is unavailable, say so.
"""
    return invoke_llm(prompt, api_key=api_key)
