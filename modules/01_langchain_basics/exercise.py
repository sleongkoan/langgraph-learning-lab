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
    model = get_chat_model()
    response = model.invoke(
        [HumanMessage(content="Explain what LangGraph is in one sentence.")])
    print(response.content)


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
    model = get_chat_model()
    first_messages = [
        SystemMessage(
            content="You are a terse backend engineer who answers in one sentence."),
        HumanMessage(content="What's LangGraph?"),]

    first_response = model.invoke(first_messages)

    second_messages = [
        AIMessage(content=first_response.content),
        HumanMessage(content="How does it work?"),
    ]

    second_response = model.invoke(first_messages + second_messages)

    last_messages = [
        AIMessage(content=second_response.content),
        HumanMessage(content="Can you give an example?"),
    ]
    final_response = model.invoke(
        first_messages + second_messages + last_messages)
    print(final_response.content)


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
    model = get_chat_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert on {topic}."),
        ("human", "Explain {topic} to {audience}."),
    ])
    messages = prompt.invoke(
        {"topic": "GraphQL", "audience": "a junior engineer"})
    response = model.invoke(messages)
    print(response.content)


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


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
    model = get_chat_model()
    model = model.bind_tools([get_word_length])
    response = model.invoke(
        [HumanMessage(content="How many letters are in 'LangGraph'?")])
    print(response.tool_calls)


if __name__ == "__main__":
    print("=== part1 ===")
    part1_basic_call()
    print("=== part2 ===")
    part2_system_and_history()
    print("=== part3 ===")
    part3_prompt_template()
    print("=== part4 ===")
    part4_tool_calling()
