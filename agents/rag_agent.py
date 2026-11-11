from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


llm = OllamaLLM(
    model="llama3.2"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="research_documents",
    persist_directory="rag/chroma_db",
    embedding_function=embeddings
)


def rag_agent(question):

    documents = vectorstore.similarity_search(
        question,
        k=4
    )

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

    answer = llm.invoke(prompt)

    return answer