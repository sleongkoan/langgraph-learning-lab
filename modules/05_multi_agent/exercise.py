"""
Module 5: multi-agent orchestration (supervisor + workers)

Goal: a supervisor node classifies the request and routes to a specialized
worker agent -- and the worker is itself a fully compiled graph, dropped in
as a single node. This is how reusable agentic components actually compose:
you build a worker once (e.g. the module 3 tool-agent), then wire it into
any number of parent graphs.

Run with:
    .venv\\Scripts\\python.exe modules\\05_multi_agent\\exercise.py
"""
from typing import Annotated, TypedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Worker 1: math agent (same shape as module 3's tool-agent, given to you --
# this part is review, not new material)
# ---------------------------------------------------------------------------

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


MATH_TOOLS = [add, multiply]
MATH_TOOLS_BY_NAME = {t.name: t for t in MATH_TOOLS}


def math_agent_node(state: AgentState) -> dict:
    model = get_chat_model().bind_tools(MATH_TOOLS)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def math_tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    for call in getattr(last, "tool_calls", []):
        fn = MATH_TOOLS_BY_NAME[call["name"]]
        result = fn.invoke(call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}


def math_should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", []) else "end"


def build_math_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", math_agent_node)
    graph.add_node("tools", math_tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", math_should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


MATH_AGENT = build_math_agent()


# ---------------------------------------------------------------------------
# Worker 2: general chat agent (given, review)
# ---------------------------------------------------------------------------

def general_agent_node(state: AgentState) -> dict:
    model = get_chat_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_general_agent():
    graph = StateGraph(AgentState)
    graph.add_node("respond", general_agent_node)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile()


GENERAL_AGENT = build_general_agent()


# ---------------------------------------------------------------------------
# Supervisor (this is the new material -- fill these in)
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


def supervisor_node(state: SupervisorState) -> dict:
    """
    TODO:
    1. Get the latest user question: state["messages"][-1].content
    2. Create a chat model with get_chat_model() and invoke it with a short
       message list of your own construction (NOT state["messages"] --
       this classification call should NOT become part of the visible
       conversation):
         [SystemMessage(content="You are a router. Reply with exactly one "
                                 "word: 'math' if the user's question "
                                 "requires a calculation, otherwise "
                                 "'general'. Output only that one word."),
          HumanMessage(content=<the question>)]
    3. Take response.content, strip/lowercase it. If it's not exactly "math",
       default it to "general" (small models don't always follow
       instructions perfectly -- never trust raw output as a safe dict key
       without a fallback).
    4. Return {"next": <that word>} -- note: do NOT return "messages" here.
       The routing decision is bookkeeping, not something the user should
       see appended to the conversation.
    """
    raise NotImplementedError


def route_to_worker(state: SupervisorState) -> str:
    """TODO: return state["next"] -- this is what add_conditional_edges will look up."""
    raise NotImplementedError


def build_supervisor_graph():
    """
    TODO:
    1. StateGraph(SupervisorState).
    2. add_node("supervisor", supervisor_node)
    3. add_node("math", MATH_AGENT)      -- a compiled graph used directly as a node
    4. add_node("general", GENERAL_AGENT)
    5. START -> "supervisor"
    6. add_conditional_edges("supervisor", route_to_worker,
           {"math": "math", "general": "general"})
    7. "math" -> END, "general" -> END
    8. Compile and return.
    """
    raise NotImplementedError


def ask(question: str):
    app = build_supervisor_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)], "next": ""})
    print(f"routed to: {result['next']}")
    print(f"answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("--- math question ---")
    ask("What is 12 times 7?")
    print("\n--- general question ---")
    ask("Tell me a fun fact about octopuses.")
