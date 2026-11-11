import os
from typing import Optional
from config.llm import get_llm
from rag.ingest import get_vectorstore


def rag_agent(
    question: str,
    llm: Optional[any] = None,
    embeddings_provider: str = "auto",
    ollama_base_url: str = "http://localhost:11434"
) -> str:
    """
    Retrieves evidence from ingested PDF research papers stored in the active in-memory ChromaDB vectorstore.
    """
    vectorstore = get_vectorstore(
        embeddings_provider=embeddings_provider,
        ollama_base_url=ollama_base_url
    )

    if vectorstore is None:
        return "No local research papers were uploaded or indexed for this analysis. Relying on live web literature."

    try:
        documents = vectorstore.similarity_search(question, k=4)
        if not documents:
            return "No directly relevant passages found in the uploaded documents for this specific topic."

        context_blocks = []
        for i, doc in enumerate(documents, start=1):
            source = os.path.basename(doc.metadata.get("source", "Uploaded Paper"))
            page = doc.metadata.get("page", 0) + 1
            context_blocks.append(f"[Source {i}: {source} (Page {page})]\n{doc.page_content.strip()}")

        context = "\n\n".join(context_blocks)

        prompt = f"""You are an expert Document Analysis and RAG Agent.

Your goal is to extract evidence-backed insights from the provided academic paper excerpts.

Research Topic / Query:
{question}

Extracted Academic Evidence:
{context}

Guidelines:
- Provide clear, evidence-based synthesis.
- Cite specific document passages or source numbers where possible.
- Do not fabricate claims or statistics not present in the excerpts.
- If certain aspects of the question are unaddressed in the text, state that explicitly.
"""
        model = llm if llm is not None else get_llm(provider="groq" if os.environ.get("GROQ_API_KEY") else "ollama")
        response = model.invoke(prompt)

        return response.content if hasattr(response, "content") else str(response)

    except Exception as e:
        return f"Document retrieval notice: {str(e)}"
