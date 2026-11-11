from typing import TypedDict, Optional
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


def run_rag(state: ResearchState):
    result = rag_agent(state["topic"], api_key=state.get("api_key"))
    return {"rag_result": result}


def run_web(state: ResearchState):
    result = web_agent(state["topic"], api_key=state.get("api_key"))
    return {"web_result": result}


def run_verification(state: ResearchState):
    result = verification_agent(
        state["topic"],
        state["rag_result"],
        state["web_result"],
        api_key=state.get("api_key")
    )
    return {"verification": result}


def run_report(state: ResearchState):
    result = report_agent(
        state["topic"],
        state["rag_result"],
        state["web_result"],
        state["verification"],
        api_key=state.get("api_key")
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
