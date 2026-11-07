"""
Shared chat-model factory used by every module's exercise and solution.

Swap model providers without touching any exercise code by setting two
environment variables (see .env.example):

    MODEL_PROVIDER   default "ollama" -- any provider id supported by
                     langchain's init_chat_model(), e.g. "openai",
                     "anthropic", "google_genai", "groq", "mistralai", ...
    MODEL_NAME       default "qwen2.5:7b"

Provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, ...)
are read directly from the environment by each provider's integration --
put them in a local .env file (git-ignored) and they're picked up
automatically via load_dotenv() below.

The default provider, Ollama, needs no API key: install Ollama, run
`ollama pull qwen2.5:7b`, and every module works offline for free. Switching
to a hosted provider also needs that provider's langchain integration
package installed (e.g. `pip install langchain-openai` for "openai") --
see README.md for the full list.
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"


def get_chat_model(**kwargs):
    """Build the chat model every module uses, based on MODEL_PROVIDER/MODEL_NAME."""
    provider = os.environ.get("MODEL_PROVIDER", DEFAULT_PROVIDER)
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    return init_chat_model(model, model_provider=provider, **kwargs)
