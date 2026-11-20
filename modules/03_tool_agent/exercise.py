"""
Module 3: a tool-using agent (the ReAct loop)

Goal: build the actual agent loop -- a model node that can request tool calls,
a tool-execution node, and a cycle that routes back to the model until it's
done. This is the pattern almost every real "agent" is built from underneath.

Run with:
    .venv\\Scripts\\python.exe modules\\03_tool_agent\\exercise.py
"""
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated, TypedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


_SECRET_NUMBERS = {"alpha": 17, "beta": 42, "gamma": 5}


@tool
def secret_number(key: str) -> int:
    """Look up the secret number for a key. This cannot be guessed or computed -- it must be looked up."""
    return _SECRET_NUMBERS[key]


TOOLS = [add, multiply, secret_number]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def call_agent(state: AgentState) -> dict:
    """
    TODO:
    1. Create a chat model with get_chat_model() and bind TOOLS to it.
    2. Invoke it with state["messages"].
    3. Return {"messages": [<the response>]}.

    Same shape as module 2 part 3's call_model, just with tools bound.
    """
    model = get_chat_model()
    model = model.bind_tools(TOOLS)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """
    TODO:
    The last message in state["messages"] is the AIMessage the agent node
    just produced. It may contain one or more requested tool calls in
    `.tool_calls`, each a dict like:
        {"name": "add", "args": {"a": 3, "b": 4}, "id": "call_xyz"}

    1. Grab the last message: state["messages"][-1]
    2. For each entry in its .tool_calls:
         - look up the real tool function in TOOLS_BY_NAME by name
         - run it with .invoke(call["args"])
         - wrap the result in a ToolMessage(content=str(result),
           tool_call_id=call["id"]) -- the tool_call_id is what lets the
           model match this result back to the request it made.
    3. Return {"messages": [<list of ToolMessages>]}
    """
    last_message = state["messages"][-1]
    tool_messages = []
    for call in getattr(last_message, "tool_calls", []):
        tool_fn = TOOLS_BY_NAME[call["name"]]
        try:
            result = tool_fn.invoke(call["args"])
            content = str(result)
        except Exception as e:
            content = f"Error calling {call['name']}: {e}"
        tool_messages.append(ToolMessage(
            content=content, tool_call_id=call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """
    TODO: look at the last message's .tool_calls. Return "continue" if it's
    non-empty (the agent wants to use a tool), otherwise return "end".
    """
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", []) else "end"


def build_agent_graph():
    """
    TODO:
    1. StateGraph(AgentState), add nodes "agent" (call_agent) and "tools" (call_tools).
    2. START -> "agent".
    3. add_conditional_edges("agent", should_continue,
           {"continue": "tools", "end": END})
    4. Wire "tools" back to "agent" -- this is the loop. Try running WITHOUT
       this edge first (comment it out) and see what the final answer looks
       like; then add it back and compare.
    5. Compile and return the app.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue,
                                {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
    app = graph.compile()
    return app


def run_agent(question: str):
    app = build_agent_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    for m in result["messages"]:
        print(f"[{m.__class__.__name__}] {m.content}")


if __name__ == "__main__":
    print("--- arithmetic (model can shortcut this) ---")
    run_agent(
        "What is (3 + 4) * 2? Use the add and multiply tools to compute it "
        "step by step, don't just answer from memory."
    )
    print("\n--- forces a real loop: model cannot know the secret number ---")
    run_agent(
        "Look up the secret number for 'alpha', then multiply it by 3. "
        "You cannot know the secret number without calling secret_number first."
    )
