from agents.web_agent import web_agent


print("=" * 60)
print("WEB RESEARCH AGENT")
print("=" * 60)

topic = input("\nEnter research topic: ")

print("\nSearching the web...\n")

result = web_agent(topic)

print("=" * 60)
print("WEB RESEARCH RESULTS")
print("=" * 60)

print(result)