from agents.researcher import researcher_agent


topic = "Generative AI applications in healthcare"

print("Starting Research Agent...\n")

result = researcher_agent(topic)

print("=" * 60)
print("RESEARCH AGENT")
print("=" * 60)

print(result)