from agents.rag_agent import rag_agent


print("=" * 60)
print("RAG AGENT TEST")
print("=" * 60)

question = input("\nEnter your research question: ")

print("\nSearching research documents...\n")

answer = rag_agent(question)

print("=" * 60)
print("RAG ANSWER")
print("=" * 60)

print(answer)