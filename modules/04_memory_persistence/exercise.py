"""
Module 4: memory / persistence (checkpointing)

Goal: make a graph remember conversation state across separate .invoke() calls,
the way a real chat-based agent needs to across multiple user turns, without
the caller resending the whole history every time.

Run with:
    .venv\\Scripts\\python.exe modules\\04_memory_persistence\\exercise.py
"""
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import Annotated, TypedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: ChatState) -> dict:
    model = get_chat_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """
    TODO:
    1. StateGraph(ChatState), one node "call_model".
    2. START -> "call_model" -> END.
    3. Compile WITH a checkpointer: graph.compile(checkpointer=MemorySaver()).
       Without that argument, nothing persists between separate .invoke()
       calls -- each call starts from exactly the input dict you give it,
       nothing more.
    """
    graph = StateGraph(ChatState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    app = graph.compile(checkpointer=MemorySaver())
    return app


def part1_threads(app):
    """
    TODO:
    A `thread_id` identifies one persisted conversation. You pass it via a
    config dict: config = {"configurable": {"thread_id": "..."}}

    1. config_a = {"configurable": {"thread_id": "thread-a"}}
       Invoke app with {"messages": [HumanMessage(content="My name is Sam.")]}
       and config=config_a.
    2. Invoke AGAIN on config_a, this time with ONLY
       {"messages": [HumanMessage(content="What's my name?")]} -- do not
       resend the first message yourself. Print the final answer's content.
       (It should know, because the checkpointer kept thread-a's history.)
    3. config_b = {"configurable": {"thread_id": "thread-b"}}
       Invoke with just {"messages": [HumanMessage(content="What's my name?")]}
       on config_b, which never saw the introduction. Print the answer and
       notice it has no idea -- threads are isolated conversations.
    4. Return config_a (part 2 reuses it).
    """
    config_a = {"configurable": {"thread_id": "thread-a"}}
    app.invoke(
        {"messages": [HumanMessage(content="My name is Sam.")]}, config=config_a)
    response_a = app.invoke(
        {"messages": [HumanMessage(content="What's my name?")]}, config=config_a)
    print(response_a["messages"][0].content)
    config_b = {"configurable": {"thread_id": "thread-b"}}
    response_b = app.invoke(
        {"messages": [HumanMessage(content="What's my name?")]}, config=config_b)
    print(response_b["messages"][0].content)
    return config_a


def part2_inspect_state(app, config):
    """
    TODO:
    1. Call app.get_state(config) and print state.values["messages"] -- the
       full persisted history for that thread.
    2. Call list(app.get_state_history(config)) and print len(...) -- one
       snapshot per step the graph has ever taken on that thread.

    Why this matters operationally: if an agent misbehaves in production,
    get_state_history is how you reconstruct exactly what it saw and did,
    step by step.
    """
    state = app.get_state(config)
    print(state.values["messages"])
    history = list(app.get_state_history(config))
    print(len(history))


if __name__ == "__main__":
    app = build_graph()
    print("=== part1 ===")
    config_a = part1_threads(app)
    print("=== part2 ===")
    part2_inspect_state(app, config_a)
