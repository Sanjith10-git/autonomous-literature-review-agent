from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from agents.rag_agent import rag_agent
from agents.web_agent import web_agent
from agents.verifier import verification_agent
from agents.report_agent import report_agent
from config.llm import get_llm


class ResearchState(TypedDict):
    topic: str
    mode: str                    # "hybrid" | "web_only" | "docs_only"
    provider: str                # "groq" | "ollama"
    api_key: Optional[str]
    base_url: str
    model: Optional[str]
    rag_result: str
    web_result: str
    verification: str
    report: str
    sources: List[Dict[str, str]]


def get_state_llm(state: ResearchState):
    provider = state.get("provider", "groq")
    api_key = state.get("api_key")
    base_url = state.get("base_url", "http://localhost:11434")
    model = state.get("model")
    try:
        return get_llm(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    except Exception as e:
        print(f"Workflow LLM factory warning: {e}")
        return None


def run_rag(state: ResearchState) -> Dict[str, Any]:
    mode = state.get("mode", "hybrid")
    if mode == "web_only":
        return {
            "rag_result": "Skipped per research mode selection: Autonomous Web Discovery Mode."
        }

    llm = get_state_llm(state)
    base_url = state.get("base_url", "http://localhost:11434")
    provider = state.get("provider", "groq")
    embeddings_provider = "ollama" if provider == "ollama" else "chroma_default"

    result = rag_agent(
        question=state["topic"],
        llm=llm,
        embeddings_provider=embeddings_provider,
        ollama_base_url=base_url
    )
    return {"rag_result": result}


def run_web(state: ResearchState) -> Dict[str, Any]:
    mode = state.get("mode", "hybrid")
    if mode == "docs_only":
        return {
            "web_result": "Skipped per research mode selection: Document Deep-Dive Mode.",
            "sources": []
        }

    llm = get_state_llm(state)
    web_data = web_agent(
        topic=state["topic"],
        llm=llm
    )
    return {
        "web_result": web_data.get("summary", ""),
        "sources": web_data.get("sources", [])
    }


def run_verification(state: ResearchState) -> Dict[str, Any]:
    llm = get_state_llm(state)
    result = verification_agent(
        topic=state["topic"],
        rag_information=state.get("rag_result", ""),
        web_information=state.get("web_result", ""),
        llm=llm
    )
    return {"verification": result}


def run_report(state: ResearchState) -> Dict[str, Any]:
    llm = get_state_llm(state)
    result = report_agent(
        topic=state["topic"],
        rag_information=state.get("rag_result", ""),
        web_information=state.get("web_result", ""),
        verification=state.get("verification", ""),
        sources=state.get("sources", []),
        llm=llm
    )
    return {"report": result}


# Build LangGraph state machine
graph = StateGraph(ResearchState)

graph.add_node("rag", run_rag)
graph.add_node("web", run_web)
graph.add_node("verification", run_verification)
graph.add_node("report", run_report)

graph.add_edge(START, "rag")
graph.add_edge("rag", "web")
graph.add_edge("web", "verification")
graph.add_edge("verification", "report")
graph.add_edge("report", END)

research_workflow = graph.compile()
