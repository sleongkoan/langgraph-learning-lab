# LangGraph Learning Lab

A hands-on, project-based path through [LangChain](https://python.langchain.com/)
and [LangGraph](https://langchain-ai.github.io/langgraph/) fundamentals. Each
module is a small, runnable program with TODOs to fill in, not just notes --
the goal is to come out the other side able to actually build agentic flows,
not just describe them.

This repo was built as an **AI-assisted learning project**: the exercises,
docs, and scaffolding were put together with an AI coding assistant, and it's
designed to keep being used that way -- see
[AI_TUTOR.md](AI_TUTOR.md) for a ready-made prompt that turns any AI
assistant (Claude Code, GitHub Copilot, Gemini, ChatGPT, Cursor, ...) into a
tutor that walks you through it module by module instead of just handing you
the answers.

## Modules

| # | Module | What you'll learn |
|---|--------|--------------------|
| 1 | [langchain_basics](modules/01_langchain_basics) | Chat models, messages, prompt templates, basic tool binding |
| 2 | [langgraph_basics](modules/02_langgraph_basics) | `StateGraph`: state schemas, nodes, edges, conditional routing, reducers |
| 3 | [tool_agent](modules/03_tool_agent) | The ReAct loop: a model node that calls tools and loops until done |
| 4 | [memory_persistence](modules/04_memory_persistence) | Checkpointing: multi-turn, multi-thread conversation state |
| 5 | [multi_agent](modules/05_multi_agent) | Supervisor + worker graphs: composing agents out of smaller agents |
| 6 | [api_service](modules/06_api_service) | Serving a graph behind a REST API (FastAPI) |
| 7 | [testing_evals](modules/07_testing_evals) | pytest-based testing/evaluation of agents |
| - | [capstone](capstone) | A small end-to-end project combining everything above -- no starter code, just a spec |

Each of modules 1-7 has one file to work through (`exercise.py`, or
`test_exercise.py` for module 7), with TODOs for you to fill in. Run it
(module 7: run it with pytest), read the error or the TODO docstring, write
the code. There's no answer key on purpose -- see [AI_TUTOR.md](AI_TUTOR.md)
for how to get unstuck without just reading a solution file.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1      # PowerShell
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Run any module directly, e.g.:

```
.venv\Scripts\python.exe modules\01_langchain_basics\exercise.py
```

## How to use this repo

1. Do the [Setup](#setup) below, then pick a [model provider](#choosing-a-model-provider)
   (Ollama needs no account/API key, so it's the easiest way to start).
2. Work through the [modules](#modules) **in order, 1 through 7** -- each
   one builds on concepts from the last (module 3's tool loop reappears
   inside module 5's worker, module 4's checkpointing reappears behind
   module 6's API, and so on).
3. For each module:
   - Open its `exercise.py` (module 7: `test_exercise.py`) and read the
     file's top docstring, then the docstring on the first TODO.
   - Write just enough code to satisfy that one TODO, then run the file
     (module 7: run it with `pytest`) and read what happens -- an error, a
     printed result, or a failing assertion all tell you something.
   - Move to the next TODO in the same file and repeat.
4. There's no answer key by design. If you get stuck, re-read the
   docstring and the linked LangChain/LangGraph docs first; if you want an
   AI assistant to help without it just writing the module for you, see
   [AI_TUTOR.md](AI_TUTOR.md) for a copy-paste prompt that works with any
   assistant -- and for turning off your editor's inline AI autocomplete so
   it doesn't finish the TODOs for you.
5. Once modules 1-7 are done, attempt the [capstone](capstone) -- it has a
   spec, not starter code, since the point is combining everything
   yourself.

## Choosing a model provider

Every module gets its chat model from `common/llm.py`'s `get_chat_model()`,
so switching providers never means touching exercise code -- just environment
variables. Copy `.env.example` to `.env` and set:

```
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b
```

| Provider | `MODEL_PROVIDER` | Extra install | API key env var |
|---|---|---|---|
| Ollama (local, free, default) | `ollama` | `langchain-ollama` (included) | none |
| OpenAI | `openai` | `pip install langchain-openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `pip install langchain-anthropic` (included) | `ANTHROPIC_API_KEY` |
| Google Gemini | `google_genai` | `pip install langchain-google-genai` | `GOOGLE_API_KEY` |
| Groq | `groq` | `pip install langchain-groq` | `GROQ_API_KEY` |

Any provider id supported by LangChain's
[`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
works here -- the table above is just the common ones.

### Running locally with Ollama (default, no API key)

1. Install [Ollama](https://ollama.com) (`winget install Ollama.Ollama` on Windows).
2. Pull the model: `ollama pull qwen2.5:7b`.
3. Ollama runs as a background service after install, listening on
   `localhost:11434` -- nothing else to configure.

## Repo layout

```
common/llm.py        # get_chat_model() -- the only provider-specific code
modules/NN_name/
    exercise.py       # TODOs -- no answer key, see AI_TUTOR.md
.env.example          # copy to .env, pick a provider
.cursorignore, .aiexclude, .codeiumignore, .continueignore
                      # best-effort: keep AI autocomplete off exercise files
```
