from typing import Optional
from config.llm import invoke_llm


def report_agent(
    topic: str,
    rag_information: str,
    web_information: str,
    verification: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "auto",
    temperature: float = 0,
    review_focus: str = "Comprehensive Literature Review",
    writing_style: str = "Academic Prose"
) -> str:
    """Compiles a structured academic literature review report tailored to the researcher's focus and style."""
    prompt = f"""
You are a professional academic research report writer.

Research topic:
{topic}

Requested Review Focus:
{review_focus}

Requested Writing Style:
{writing_style}

Information from uploaded documents:
{rag_information}

Web research:
{web_information}

Verification:
{verification}

Create a detailed research report aligned with the requested review focus and writing style.

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
- Tailor depth and formatting to '{review_focus}' using '{writing_style}'.
"""
    return invoke_llm(
        prompt,
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature
    )
