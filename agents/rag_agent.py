import os
from typing import Optional
from config.llm import invoke_llm
from rag.ingest import get_vectorstore


def rag_agent(
    question: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "auto",
    temperature: float = 0,
    vectorstore: Optional[any] = None
) -> str:
    """Retrieves relevant chunks from ingested PDFs and answers using the active LLM."""
    vs = vectorstore if vectorstore is not None else get_vectorstore()
    if vs is None:
        return "No uploaded documents were indexed for this session."

    documents = vs.similarity_search(question, k=4)
    if not documents:
        return "No relevant information found in the uploaded documents."

    context_blocks = []
    for i, document in enumerate(documents, start=1):
        src = os.path.basename(document.metadata.get("source", "Manuscript"))
        page = document.metadata.get("page")
        page_label = f" (Page {page + 1})" if isinstance(page, int) else ""
        context_blocks.append(f"[{src}{page_label}]\n{document.page_content}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a research assistant.

Answer the question using the research information below.

Research information:
{context}

Question:
{question}

Instructions:
- Give a clear, evidence-backed answer.
- Cite document names and page numbers from the context when available.
- Do not invent facts.
- If the information is unavailable in the excerpts, say so clearly.
"""
    return invoke_llm(
        prompt,
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature
    )
