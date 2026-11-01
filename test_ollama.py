from langchain_ollama import ChatOllama

print("Connecting to Ollama...")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("Sending question to Llama 3.2...")

response = llm.invoke(
    "Explain artificial intelligence in 3 simple sentences."
)

print("\n==============================")
print("OLLAMA TEST")
print("==============================\n")

print(response.content)