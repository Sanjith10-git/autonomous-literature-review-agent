from typing import Optional
from config.llm import invoke_llm
from tools.web_search import web_search


def web_agent(topic: str, api_key: Optional[str] = None) -> str:
    """Searches the web for the research topic and summarizes key sourced findings."""
    results = web_search(
        topic,
        max_results=5
    )

    if not results:
        return "No web search results found."

    context = "\n\n".join(
        f"Title: {result['title']}\n"
        f"URL: {result['url']}\n"
        f"Summary: {result['snippet']}"
        for result in results
    )

    prompt = f"""
You are a web research agent.

Research topic:
{topic}

Web search results:
{context}

Analyze the search results and provide:

1. Important findings
2. Key facts
3. Relevant sources
4. Information that should be verified

Rules:
- Do not invent information.
- Use only the provided search results.
- Include relevant source URLs.
"""
    return invoke_llm(prompt, api_key=api_key)
