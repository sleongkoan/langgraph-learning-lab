"""
Module 6: serving a graph behind a REST API

Goal: wrap the chat graph from module 4 behind a small FastAPI service.
A real agent almost never lives in a single invoke() script -- something
calls it over HTTP, needs a stable request/response shape, and needs to
keep per-conversation state across separate requests the same way module 4's
checkpointer kept it across separate invoke() calls.

Run with:
    .venv\\Scripts\\python.exe modules\\06_api_service\\exercise.py

Then, in another terminal:
    curl -X POST http://127.0.0.1:8000/chat ^
         -H "Content-Type: application/json" ^
         -d "{\"thread_id\": \"demo\", \"message\": \"My name is Sam.\"}"

    curl -X POST http://127.0.0.1:8000/chat ^
         -H "Content-Type: application/json" ^
         -d "{\"thread_id\": \"demo\", \"message\": \"What is my name?\"}"

    curl http://127.0.0.1:8000/threads/demo/history
"""
from typing import Annotated, TypedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# The graph (given -- same shape as module 4's checkpointed chat graph)
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: ChatState) -> dict:
    model = get_chat_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile(checkpointer=MemorySaver())


GRAPH = build_graph()
app = FastAPI(title="LangGraph chat service")


# ---------------------------------------------------------------------------
# Request/response schemas (this is the new material)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    TODO: add two fields:
      thread_id: str  -- identifies which persisted conversation this
                          request belongs to (see module 4).
      message: str    -- the user's message text.
    """
    pass


class ChatResponse(BaseModel):
    """TODO: add one field: reply: str."""
    pass


# ---------------------------------------------------------------------------
# Endpoints (this is the new material)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    TODO:
    1. Build a config dict from request.thread_id, the same shape module 4
       used: {"configurable": {"thread_id": request.thread_id}}.
    2. Invoke GRAPH with {"messages": [HumanMessage(content=request.message)]}
       and that config.
    3. Return ChatResponse(reply=<the final message's content>).

    Because GRAPH was compiled with a checkpointer and you reuse the same
    thread_id across requests, the model sees the full prior history for
    that thread without the caller resending it -- exactly like module 4,
    just reachable over HTTP now instead of from a Python script.
    """
    raise NotImplementedError


@app.get("/threads/{thread_id}/history")
def history(thread_id: str):
    """
    TODO:
    1. Build the same config shape as in chat().
    2. Call GRAPH.get_state(config) and pull out .values["messages"].
    3. Return a list of {"role": <role>, "content": <text>} dicts -- don't
       return the raw LangChain message objects, they're not JSON
       serializable as-is. Use message.type ("human"/"ai"/"tool") for role.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
