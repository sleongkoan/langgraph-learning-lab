"""
Module 7: testing and evaluating agents with pytest

Goal: test LangGraph code without making real LLM calls for most tests --
real calls are slow, can cost money/quota, and are non-deterministic, which
makes them a bad fit for a suite you want to run on every commit.

Three techniques this module covers:
  1. Pull deterministic logic (tool execution, routing) out into plain
     functions and unit-test them directly -- no model involved at all.
  2. For the parts that DO need a model in the loop, swap in a stub chat
     model (StubChatModel below) so the test is fast and repeatable, then
     assert on the *shape* of the result rather than exact wording.
  3. (Last section) a true "eval" -- a test that DOES call your configured
     model and checks loose properties of a real answer. Useful, but not
     something you want on every commit, so it's skipped unless you opt in.

The agent under test below is given, fully working -- your job in this
module is to write the tests, not the agent.

Run with:
    .venv\\Scripts\\python.exe -m pytest modules\\07_testing_evals\\test_exercise.py -v

Run the real-model eval too (costs a call to whatever MODEL_PROVIDER you
have configured):
    set RUN_LLM_EVALS=1
    .venv\\Scripts\\python.exe -m pytest modules\\07_testing_evals\\test_exercise.py -v
"""
from typing import Annotated, TypedDict
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# The agent under test (given -- same ReAct shape as module 3)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


TOOLS = [add]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def call_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    for call in getattr(last_message, "tool_calls", []):
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", []) else "end"


def call_agent(state: AgentState) -> dict:
    model = get_chat_model().bind_tools(TOOLS)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# A stand-in for a real chat model -- returns canned responses in order
# instead of calling Ollama/OpenAI/etc. over the network.
# ---------------------------------------------------------------------------

class StubChatModel:
    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Part 1: unit-test the deterministic plumbing -- no model involved
# ---------------------------------------------------------------------------

def test_should_continue_true_when_tool_calls_present():
    """
    TODO: build an AgentState whose last message is an AIMessage with a
    non-empty tool_calls list, e.g.:
        AIMessage(content="", tool_calls=[
            {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}
        ])
    Assert should_continue(state) == "continue".
    """
    raise NotImplementedError


def test_should_continue_false_when_no_tool_calls():
    """
    TODO: same idea, but the last message has no tool calls (a plain
    AIMessage(content="some answer")). Assert should_continue(state) == "end".
    """
    raise NotImplementedError


def test_call_tools_executes_and_wraps_result():
    """
    TODO:
    1. Build a state whose last message is:
       AIMessage(content="", tool_calls=[
           {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1"}
       ])
    2. Call call_tools(state).
    3. Assert the result has exactly one message, it's a ToolMessage, its
       .content == "5", and its .tool_call_id == "call_1".

    This is the heart of testing agent plumbing: you never need the LLM to
    verify that *your* code correctly executes a tool call and wires the
    result back -- you just need to construct the AIMessage it would have
    produced.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: test the full graph with a stub model (fast, deterministic, free)
# ---------------------------------------------------------------------------

def test_agent_graph_with_stub_model(monkeypatch):
    """
    TODO:
    1. monkeypatch.setattr(sys.modules[__name__], "get_chat_model", lambda:
       StubChatModel([...])), with two canned responses:
         - AIMessage(content="", tool_calls=[
               {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1"}
           ])
         - AIMessage(content="The answer is 5.")
       The first simulates the model deciding to call `add`; the second
       simulates it answering once it sees the tool result.
    2. Build the graph with build_agent_graph() and invoke it with
       {"messages": [HumanMessage(content="What is 2 + 3?")]}.
    3. Assert the final message's content == "The answer is 5." and that
       the conversation has 4 messages total (human, AI tool-call request,
       tool result, final AI answer).

    Why this matters: this test runs in milliseconds, costs nothing, and
    gives the exact same result every time -- regardless of which real
    MODEL_PROVIDER a developer has configured locally.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: an eval -- calls your REAL configured model, checked loosely.
# Skipped by default: set RUN_LLM_EVALS=1 to opt in. Evals are slower, may
# cost money/quota depending on your provider, and aren't fully
# deterministic -- that's normal for evals, unlike the unit tests above.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_LLM_EVALS") != "1",
    reason="set RUN_LLM_EVALS=1 to run tests that call a real model",
)
def test_agent_eval_real_model():
    """
    TODO:
    1. Build the real graph with build_agent_graph() (no stubbing this time).
    2. Invoke it with a question like "What is 12 + 30?".
    3. Instead of asserting exact text, assert a loose property of the
       answer -- e.g. that "42" appears somewhere in the final message's
       content. Exact wording from a real model varies; evals check
       properties, not transcripts.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
