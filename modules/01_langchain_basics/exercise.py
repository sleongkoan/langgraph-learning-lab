"""
Module 1: LangChain basics

Goal: get comfortable with ChatModel, Messages, PromptTemplate, and basic
tool-calling -- the vocabulary everything later (LangGraph, agents) is built on.

Run with:
    .venv\\Scripts\\python.exe modules\\01_langchain_basics\\exercise.py

Fill in each TODO. Nothing here needs LangGraph yet.

The model itself is provider-agnostic: get_chat_model() (from common/llm.py)
returns whichever chat model your .env configures (Ollama by default, or
OpenAI/Anthropic/Gemini/etc. if you set MODEL_PROVIDER/MODEL_NAME) -- the
rest of this module's code never changes based on that choice.
"""
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_chat_model


def part1_basic_call():
    """
    TODO:
    1. Create a chat model with get_chat_model().
    2. Call it with a single HumanMessage asking it to explain what LangGraph is
       in one sentence.
    3. Print the response's .content
    """
    raise NotImplementedError


def part2_system_and_history():
    """
    TODO:
    1. Build a short conversation: a SystemMessage setting the assistant's persona
       (e.g. "You are a terse backend engineer who answers in one sentence."),
       then two HumanMessage/AIMessage turns of your choosing, then a final
       HumanMessage question.
    2. Invoke the model with the full message list and print the answer.

    This mirrors how real chat state is just a growing list of messages --
    there's no hidden session on the server side.
    """
    raise NotImplementedError


def part3_prompt_template():
    """
    TODO:
    1. Build a ChatPromptTemplate with a system message and a human message
       that both contain a variable, e.g. {topic} and {audience}.
    2. Format it for topic="GraphQL" and audience="a junior engineer", producing
       actual messages (look at .invoke() on the template).
    3. Pass the formatted messages into the model and print the answer.

    Why this matters: production code never hardcodes prompt strings inline --
    templates let you version/reuse prompts and keep them out of business logic.
    """
    raise NotImplementedError


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    # TODO: implement this (one line)
    raise NotImplementedError


def part4_tool_calling():
    """
    TODO:
    1. Bind the get_word_length tool to your model (model.bind_tools([...])).
    2. Invoke it with a HumanMessage like "How many letters are in 'LangGraph'?"
    3. Print response.tool_calls -- do NOT execute the tool yet, just inspect
       what the model decided to call and with what arguments.

    This is the core mechanic agents are built on: the LLM doesn't run code,
    it *requests* a function call with structured arguments, and your code
    decides whether/how to execute it.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== part1 ===")
    part1_basic_call()
    print("=== part2 ===")
    part2_system_and_history()
    print("=== part3 ===")
    part3_prompt_template()
    print("=== part4 ===")
    part4_tool_calling()
