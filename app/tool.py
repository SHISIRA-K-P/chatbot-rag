from typing import Any

from app import rag


PAPER_RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_paper_context",
        "description": (
            "Search the uploaded Attention Is All You Need paper in Pinecone. "
            "Use this for questions about the paper or its contents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or terms to search for in the paper.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of paper passages to retrieve.",
                },
            },
            "required": ["query"],
        },
    },
}


def retrieve_paper_context(
    query: str,
    top_k: int = rag.TOP_K,
) -> list[dict[str, Any]]:
    return rag.retrieve_documents(query, top_k=top_k)


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    default_top_k: int = rag.TOP_K,
) -> list[dict[str, Any]]:
    if name != "retrieve_paper_context":
        raise ValueError(f"Unknown agent tool: {name}")

    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("The retrieval tool requires a non-empty query.")

    requested_top_k = arguments.get("top_k", default_top_k)
    if not isinstance(requested_top_k, int):
        requested_top_k = default_top_k
    top_k = min(max(requested_top_k, 1), 5)

    return retrieve_paper_context(query, top_k=top_k)
