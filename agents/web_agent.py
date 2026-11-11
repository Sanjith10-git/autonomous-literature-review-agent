import os
from typing import Optional, Dict, Any
from tools.web_search import web_search
from config.llm import get_llm


def web_agent(
    topic: str,
    llm: Optional[any] = None
) -> Dict[str, Any]:
    """
    Performs web discovery and literature reconnaissance on the topic.
    Returns a dictionary containing synthesized findings and structured sources.
    """
    search_query = f"{topic} research paper recent findings"
    results = web_search(search_query, max_results=5)

    if not results:
        # Retry with broader query
        results = web_search(topic, max_results=5)

    if not results:
        return {
            "summary": "No real-time web search results could be retrieved for this query.",
            "sources": []
        }

    context = "\n\n".join(
        f"Source [{i}]: {r['title']}\nURL: {r['url']}\nExcerpt: {r['snippet']}"
        for i, r in enumerate(results, start=1)
    )

    prompt = f"""You are an elite Autonomous Web Research Agent specializing in academic discovery.

Research Topic:
{topic}

Live Web Research Data:
{context}

Analyze these findings and deliver a structured synthesis containing:
1. Current State of the Art & Breakthroughs
2. Empirical Findings & Methodologies
3. Prominent Research Groups or Real-World Implementations
4. Critical Questions for Cross-Verification

Rules:
- Be precise, technical, and objective.
- Explicitly cite source numbers (e.g., [Source 1], [Source 2]) when referencing facts.
- Do not invent claims or metrics not verified by the search context.
"""
    model = llm if llm is not None else get_llm(provider="groq" if os.environ.get("GROQ_API_KEY") else "ollama")
    response = model.invoke(prompt)
    summary_text = response.content if hasattr(response, "content") else str(response)

    return {
        "summary": summary_text,
        "sources": results
    }
