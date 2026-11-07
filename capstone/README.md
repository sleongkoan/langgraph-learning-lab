# Capstone: build your own agent service

No TODOs, no starter code -- this is the "now build something" step. Design
and build a small service that combines what modules 1-7 covered. Treat the
modules as a toolbox, not a template to copy.

## Requirements

Your capstone must:

- Be a LangGraph `StateGraph` with at least one **conditional edge** (a
  routing decision, like module 2's even/odd split or module 5's supervisor).
- Include at least one **tool-calling agent** (module 3's loop: a model node
  that can request tool calls, a tool-execution node, routing back until done).
- **Persist conversation state** across requests using a checkpointer
  (module 4), keyed by a thread/session id.
- Be **reachable over HTTP** (module 6's FastAPI pattern) with at minimum a
  "send a message" endpoint and a "get this thread's history" endpoint.
- Have a **pytest suite** (module 7) that tests your deterministic graph
  logic (routing, tool execution) without hitting a real model, plus at
  least one stub-model test of the full graph.
- Work against **any configured provider** -- build it on `get_chat_model()`
  from `common/llm.py`, the same as every other module, so it isn't tied to
  Ollama specifically.

## Suggested shape (pick your own, this is just a starting idea)

A small "agent catalog" service: more than one specialized worker behind a
supervisor (e.g. a math agent, a tool that looks something up, a general
chat fallback), each conversation pinned to a thread, all of it served over
HTTP and tested without needing a live model for most of the suite.

Other reasonable ideas: a customer-support-style triage bot, a small
research assistant that chains two tools, a code-review-comment generator --
anything that forces you to actually wire a router, a tool loop, persistence,
and an HTTP layer together rather than just invoking a graph from `__main__`.

## Suggested structure

```
capstone/
    app.py          # FastAPI app + graph construction
    test_app.py     # pytest suite
```

Nothing enforces that layout -- organize it however makes sense once you're
building it for real.
