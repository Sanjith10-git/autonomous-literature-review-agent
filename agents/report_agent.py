from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model="llama3.2"
)


def report_agent(
    topic,
    rag_information,
    web_information,
    verification
):

    prompt = f"""
You are a professional research report writer.

Research topic:
{topic}

Information from uploaded documents:
{rag_information}

Web research:
{web_information}

Verification:
{verification}

Create a detailed research report.

Use this structure:

# Research Report

## 1. Introduction

## 2. Background

## 3. Key Findings

## 4. Evidence from Uploaded Documents

## 5. Web Research Findings

## 6. Verification

## 7. Challenges and Limitations

## 8. Conclusion

## 9. Sources

Rules:
- Do not invent information.
- Do not invent statistics.
- Clearly distinguish verified and uncertain information.
- Include relevant source URLs from the web research.
- Write in a professional academic style.
"""

    answer = llm.invoke(prompt)

    return answer