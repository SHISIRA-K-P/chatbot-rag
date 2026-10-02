import logging
from typing import Any

from ollama import chat

from app import rag
from app.tools import PAPER_RETRIEVAL_TOOL, execute_tool


logger = logging.getLogger(__name__)

NO_TOOL_ANSWER = (
    "I can answer questions using the uploaded paper. "
    "Please ask about its content."
)

AGENT_INSTRUCTIONS = """
You are the routing agent for a chatbot about the uploaded paper, "Attention Is All You Need".

For any factual question about the paper, its methods, experiments, or claims, call
the retrieve_paper_context tool before answering. Do not answer paper questions
from memory. Do not answer unrelated factual questions from outside knowledge.
For greetings or questions unrelated to the paper, do not call a tool.
""".strip()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def run_agent(question: str, top_k: int = rag.TOP_K) -> tuple[str, list[dict[str, Any]], str]:
    logger.info("User question: %s", question)

    response = chat(
        model=rag.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": AGENT_INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        tools=[PAPER_RETRIEVAL_TOOL],
        options={"temperature": 0},
    )

    message = _field(response, "message", {})
    tool_calls = _field(message, "tool_calls", None) or []

    if not tool_calls:
        logger.info("Agent decision: do not retrieve; question is outside paper scope or conversational")
        logger.info("Final answer: %s", NO_TOOL_ANSWER)
        return NO_TOOL_ANSWER, [], "no_tool"

    function = _field(tool_calls[0], "function", {})
    tool_name = _field(function, "name", "")
    arguments = _field(function, "arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    logger.info("Agent decision: retrieve paper context")
    logger.info("Tool called: %s", tool_name)
    documents = execute_tool(tool_name, arguments, default_top_k=top_k)
    logger.info(
        "Retrieved documents: %s",
        [
            {
                "page": document.get("page"),
                "score": document.get("score"),
                "excerpt": document.get("text", "")[:300],
            }
            for document in documents
        ],
    )

    if documents:
        answer = rag.generate_answer(question, documents)
    else:
        answer = "I could not find the answer in the document."

    logger.info("Final answer: %s", answer)
    return answer, documents, "retrieve_paper_context"