from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model="llama3.2"
)


def verification_agent(
    topic,
    rag_information,
    web_information
):

    prompt = f"""
You are a research verification agent.

Research topic:
{topic}

Information from uploaded documents:
{rag_information}

Information from web research:
{web_information}

Compare the two sources.

Provide:

1. Claims supported by both sources
2. Information found only in the documents
3. Information found only on the web
4. Contradictions
5. Unsupported claims
6. Important information that should be used in the final report

Rules:
- Do not invent facts.
- Do not invent statistics.
- Clearly identify uncertainty.
"""

    answer = llm.invoke(prompt)

    return answer