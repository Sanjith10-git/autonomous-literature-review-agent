import os
from typing import Optional, List, Dict
from config.llm import get_llm


def report_agent(
    topic: str,
    rag_information: str,
    web_information: str,
    verification: str,
    sources: Optional[List[Dict[str, str]]] = None,
    llm: Optional[any] = None
) -> str:
    """
    Synthesizes an executive-level, publication-quality academic research report.
    """
    sources_text = ""
    if sources:
        sources_text = "\n".join(
            f"- **[{s.get('title', 'Source')}]({s.get('url', '#')})**: {s.get('snippet', '')[:160]}..."
            for s in sources if s.get("url")
        )

    prompt = f"""You are an elite Academic Research Director and Scientific Report Writer.

Synthesize all findings into a comprehensive, highly rigorous Literature Review & State-of-the-Art Report.

Topic: {topic}

---
INPUT EVIDENCE STREAM:
- Internal Document Findings:
{rag_information[:2500]}

- External Web Discovery:
{web_information[:2500]}

- Cross-Verification & Fact-Checking:
{verification[:2500]}

- Verified External References:
{sources_text}

---
Report Formatting Structure:

# Comprehensive Literature Review: {topic}

## 1. Executive Summary & Abstract
(Concise synthesis of core findings, relevance, and current landscape)

## 2. Background & Theoretical Foundations
(Contextual definition of core principles, underlying algorithms, or historical trajectory)

## 3. State-of-the-Art Analysis & Methodologies
(Breakdown of prevailing architectures, experimental setups, and empirical evidence)

## 4. Cross-Source Evidence Synthesis & Fact Verification
(Critical evaluation of verified claims vs. emerging uncorroborated theories)

## 5. Limitations, Open Challenges & Failure Modes
(Technical bottlenecks, data constraints, or scalability challenges)

## 6. Strategic Conclusions & Future Research Directions
(Forward-looking implications and high-impact research opportunities)

## 7. Bibliography & Verified Citations
(List relevant citations with Markdown links where available)

Authoring Guidelines:
- Adopt a formal, academic tone suitable for peer review.
- Do not repeat placeholders. Write substantive, technical paragraphs.
- Highlight concrete takeaways rather than generic generalizations.
"""
    model = llm if llm is not None else get_llm(provider="groq" if os.environ.get("GROQ_API_KEY") else "ollama")
    response = model.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)
