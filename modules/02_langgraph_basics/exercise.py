"""
Module 2: LangGraph basics

Goal: understand StateGraph -- state schema, nodes, edges, conditional routing,
and reducers. This is the actual orchestration layer this course is about.

Run with:
    .venv\\Scripts\\python.exe modules\\02_langgraph_basics\\exercise.py
"""
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model


# ---------------------------------------------------------------------------
# Part 1: a minimal linear graph
# ---------------------------------------------------------------------------

class CounterState(TypedDict):
    count: int
    history: list[int]


def increment(state: CounterState):
    """
    BUG TO FIND: this returns a bare int instead of a dict of updates.
    LangGraph nodes must always return a dict (or None), never a raw value.
    Run part1 as-is first and read the error LangGraph gives you, then fix it
    to return {"count": ..., "history": [...]} with the new count appended.
    """
    new_count = state["count"] + 1
    history = state["history"] + [new_count]
    return {"count": new_count, "history": history}


def double(state: CounterState) -> dict:
    """TODO: return {'count': state['count'] * 2, 'history': [...]} with the new value appended."""
    new_count = state["count"] * 2
    history = state["history"] + [new_count]
    return {"count": new_count, "history": history}


def part1_linear_graph():
    """
    TODO:
    1. Build a StateGraph(CounterState).
    2. Add "increment" and "double" as nodes (graph.add_node(name, fn)).
    3. Wire edges: START -> "increment" -> "double" -> END.
    4. Compile it into `app` (graph.compile()).
    5. Invoke with {"count": 0, "history": []} and print the result.
    """
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_node("double", double)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double", END)
    app = graph.compile()
    result = app.invoke({"count": 0, "history": []})
    print(result)


# ---------------------------------------------------------------------------
# Part 2: conditional routing
# ---------------------------------------------------------------------------

class RouteState(TypedDict):
    number: int
    label: str


def classify(state: RouteState) -> dict:
    return {}  # no update needed, just a pass-through node before routing


def handle_even(state: RouteState) -> dict:
    return {"label": f"{state['number']} is even"}


def handle_odd(state: RouteState) -> dict:
    return {"label": f"{state['number']} is odd"}


def route_even_or_odd(state: RouteState) -> str:
    """
    TODO: return the string "even" or "odd" depending on state['number'].
    Whatever string you return gets looked up in the path_map passed to
    add_conditional_edges to decide which node runs next.
    """
    return "even" if state["number"] % 2 == 0 else "odd"


def part2_conditional_graph():
    """
    TODO:
    1. Build a StateGraph(RouteState) with nodes "classify", "handle_even", "handle_odd".
    2. START -> "classify".
    3. graph.add_conditional_edges("classify", route_even_or_odd,
           {"even": "handle_even", "odd": "handle_odd"})
    4. Both "handle_even" and "handle_odd" -> END.
    5. Compile, invoke once with {"number": 7, "label": ""} and once with
       {"number": 4, "label": ""}, print both results.
    """
    graph = StateGraph(RouteState)
    graph.add_node("classify", classify)
    graph.add_node("handle_even", handle_even)
    graph.add_node("handle_odd", handle_odd)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_even_or_odd,
                                {"even": "handle_even", "odd": "handle_odd"})
    graph.add_edge("handle_even", END)
    graph.add_edge("handle_odd", END)
    app = graph.compile()
    result1 = app.invoke({"number": 7, "label": ""})
    result2 = app.invoke({"number": 4, "label": ""})
    print(result1)
    print(result2)


# ---------------------------------------------------------------------------
# Part 3: an LLM node with a reducer-based message list
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: ChatState) -> dict:
    """
    TODO:
    1. Create a chat model with get_chat_model().
    2. Invoke it with state["messages"].
    3. Return {"messages": [<the response>]} -- the new message only, NOT the
       full list.

    Why returning just the new message is correct here: ChatState.messages is
    typed Annotated[list, add_messages], which tells LangGraph to *append*
    whatever you return onto the existing list, instead of overwriting it the
    way part 1's plain `count: int` field did. Different fields in the same
    state can have completely different merge behavior.
    """
    model = get_chat_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def part3_llm_node():
    """
    TODO:
    1. Build a StateGraph(ChatState) with one node "call_model".
    2. START -> "call_model" -> END.
    3. Compile, invoke with:
       {"messages": [HumanMessage(content="What is 2+2? Answer in one word.")]}
    4. Print the full final messages list (should have 2 messages: your
       HumanMessage and the model's AIMessage).
    """
    graph = StateGraph(ChatState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    app = graph.compile()
    final_state = app.invoke(
        {"messages": [HumanMessage(content="What is 2+2? Answer in one word.")]})
    print(final_state["messages"])


if __name__ == "__main__":
    print("=== part1 ===")
    part1_linear_graph()
    print("=== part2 ===")
    part2_conditional_graph()
    print("=== part3 ===")
    part3_llm_node()
