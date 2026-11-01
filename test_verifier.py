from agents.rag_agent import rag_agent
from agents.web_agent import web_agent
from agents.verifier import verification_agent


topic = input("Enter research topic: ")


print("\nRunning RAG Agent...")
rag_result = rag_agent(topic)


print("\nRunning Web Agent...")
web_result = web_agent(topic)


print("\nRunning Verification Agent...")

verification = verification_agent(
    topic,
    rag_result,
    web_result
)


print("\n")
print("=" * 70)
print("VERIFICATION RESULT")
print("=" * 70)

print(verification)