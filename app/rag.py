import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ollama import chat
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "intfloat/e5-base-v2"
EMBEDDING_DIMENSION = 768
OLLAMA_MODEL = "qwen2.5:3b"
INDEX_NAME = "attention-is-all-you-need"
TOP_K = 3

load_dotenv(PROJECT_DIR / ".env")

embedding_model: SentenceTransformer | None = None
pc: Pinecone | None = None
index: Any = None


def initialize_rag() -> None:
    """Load the query embedding model and connect to the populated index."""
    global embedding_model, index, pc

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to the project .env file."
        )

    pc = Pinecone(api_key=api_key)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)

    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' does not exist. "
            "Run notebooks/01_pdf_processing.ipynb to create and populate it."
        )

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    vector_count = (
        stats.get("total_vector_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "total_vector_count", 0)
    )
    if vector_count == 0:
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' is empty. "
            "Run notebooks/01_pdf_processing.ipynb to index the document."
        )

    logger.info("Connected to Pinecone index '%s' (%s vectors)", INDEX_NAME, vector_count)


def retrieve_documents(
    question: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Embed a question and retrieve its most relevant passages from Pinecone."""
    if embedding_model is None or index is None:
        raise RuntimeError("RAG is not initialized. Call initialize_rag() first.")
    if not question.strip():
        raise ValueError("The retrieval question must not be empty.")
    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be between 1 and 5.")

    query_embedding = embedding_model.encode(
        f"query: {question}",
        normalize_embeddings=True,
    )
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    documents = []
    for match in results["matches"]:
        metadata = match.get("metadata") or {}
        documents.append(
            {
                "text": metadata.get("text", ""),
                "page": metadata.get("page", "not available"),
                "score": match.get("score", 0),
            }
        )
    return documents


def generate_answer(
    question: str,
    documents: list[dict[str, Any]],
) -> str:
    """Ask the local Ollama model to answer using retrieved passages only."""
    context = "\n\n".join(document["text"] for document in documents)
    prompt = f"""You are a helpful question-answering assistant.

Answer the question using ONLY the information provided in the context.
Do not use outside knowledge or make up information.
If the answer is not available in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask_rag(
    question: str,
    top_k: int = TOP_K,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve passages and generate an answer in one call."""
    documents = retrieve_documents(question, top_k=top_k)
    answer = generate_answer(question, documents)
    return answer, documents
