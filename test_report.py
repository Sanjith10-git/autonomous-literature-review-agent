from agents.rag_agent import rag_agent
from agents.web_agent import web_agent
from agents.verifier import verification_agent
from agents.report_agent import report_agent


topic = input("Enter research topic: ")


print("\n1. Running RAG Agent...")
rag_result = rag_agent(topic)


print("\n2. Running Web Agent...")
web_result = web_agent(topic)


print("\n3. Running Verification Agent...")

verification = verification_agent(
    topic,
    rag_result,
    web_result
)


print("\n4. Creating final report...")

report = report_agent(
    topic,
    rag_result,
    web_result,
    verification
)


print("\n")
print("=" * 70)
print("FINAL RESEARCH REPORT")
print("=" * 70)

print(report)


with open(
    "outputs/research_report.md",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print("\nReport saved to:")
print("outputs/research_report.md")