from config.llm import llm


def researcher_agent(topic):

    prompt = f"""
You are a professional AI research agent.

Research topic:
{topic}

Your job is to create a research plan.

Provide:

1. Definition of the topic
2. Key concepts
3. Important subtopics
4. Questions that need to be answered
5. Important evidence that should be collected
6. Potential challenges or limitations

Do not invent statistics or sources.

Return a structured research plan.
"""

    response = llm.invoke(prompt)

    return response.content