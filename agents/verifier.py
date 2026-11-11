import os
from typing import Optional
from config.llm import get_llm


def verification_agent(
    topic: str,
    rag_information: str,
    web_information: str,
    llm: Optional[any] = None
) -> str:
    """
    Acts as an automated academic reviewer.
    Cross-verifies claims between local document evidence and external web data.
    """
    prompt = f"""You are a Senior Academic Peer Reviewer and Verification Agent.

Your objective is to rigorously cross-verify claims between internal research papers and external web findings to detect consensus, identify contradictions, and eliminate hallucinations.

Research Topic:
{topic}

---
INTERNAL DOCUMENT EVIDENCE (PDF Papers):
{rag_information}

---
EXTERNAL WEB DISCOVERY (Recent Web Sources):
{web_information}

---
Execute a structured cross-source comparative analysis:

### 1. Corroborated Findings (High Confidence)
- Detail facts, methodologies, or findings affirmed by BOTH sources.

### 2. Discrepancies & Divergent Claims
- Identify any conflicting metrics, conflicting timelines, or conflicting interpretations between documents and the web.

### 3. Document-Exclusive Knowledge
- Valuable empirical findings or formulas present only in the uploaded papers.

### 4. Recent Web-Exclusive Developments
- Emerging breakthroughs, recent benchmarks, or industry shifts found only on the web.

### 5. Verification Verdict & Evidence Integrity
- Provide an overall assessment of reliability and point out any unverified hypotheses.

Rules:
- Remain strictly analytical, scholarly, and unbiased.
- Never invent citations or synthetic performance numbers.
"""
    model = llm if llm is not None else get_llm(provider="groq" if os.environ.get("GROQ_API_KEY") else "ollama")
    response = model.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)
