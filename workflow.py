from typing import TypedDict, Optional, Any
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END

from agents.rag_agent import rag_agent
from agents.web_agent import web_agent
from agents.verifier import verification_agent
from agents.report_agent import report_agent


class ResearchState(TypedDict):
    topic: str
    rag_result: str
    web_result: str
    verification: str
    report: str
    api_key: NotRequired[Optional[str]]
    provider: NotRequired[str]
    model: NotRequired[Optional[str]]
    temperature: NotRequired[float]
    max_results: NotRequired[int]
    include_web: NotRequired[bool]
    source_mode: NotRequired[str]
    review_focus: NotRequired[str]
    writing_style: NotRequired[str]
    vectorstore: NotRequired[Optional[Any]]


def run_rag(state: ResearchState):
    if state.get("source_mode") == "web_only":
        return {"rag_result": "Document retrieval skipped (Web Discovery Only mode selected)."}

    result = rag_agent(
        state["topic"],
        api_key=state.get("api_key"),
        model=state.get("model"),
        provider=state.get("provider", "auto"),
        temperature=state.get("temperature", 0.0),
        vectorstore=state.get("vectorstore")
    )
    return {"rag_result": result}


def run_web(state: ResearchState):
    if state.get("source_mode") == "docs_only" or not state.get("include_web", True):
        return {"web_result": "Web search skipped (Uploaded Manuscripts Only mode selected)."}

    result = web_agent(
        state["topic"],
        api_key=state.get("api_key"),
        model=state.get("model"),
        provider=state.get("provider", "auto"),
        temperature=state.get("temperature", 0.0),
        max_results=state.get("max_results", 5)
    )
    return {"web_result": result}


def run_verification(state: ResearchState):
    result = verification_agent(
        state["topic"],
        state["rag_result"],
        state["web_result"],
        api_key=state.get("api_key"),
        model=state.get("model"),
        provider=state.get("provider", "auto"),
        temperature=state.get("temperature", 0.0)
    )
    return {"verification": result}


def run_report(state: ResearchState):
    result = report_agent(
        state["topic"],
        state["rag_result"],
        state["web_result"],
        state["verification"],
        api_key=state.get("api_key"),
        model=state.get("model"),
        provider=state.get("provider", "auto"),
        temperature=state.get("temperature", 0.0),
        review_focus=state.get("review_focus", "Comprehensive Literature Review"),
        writing_style=state.get("writing_style", "Academic Prose")
    )
    return {"report": result}


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
