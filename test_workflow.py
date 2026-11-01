from workflow import research_workflow


topic = input("Enter research topic: ")


print("\nStarting Multi-Agent Research Assistant...\n")


result = research_workflow.invoke({
    "topic": topic,
    "rag_result": "",
    "web_result": "",
    "verification": "",
    "report": ""
})


print("\n")
print("=" * 70)
print("FINAL RESEARCH REPORT")
print("=" * 70)

print(result["report"])


with open(
    "outputs/research_report.md",
    "w",
    encoding="utf-8"
) as file:

    file.write(result["report"])


print("\nReport saved successfully.")