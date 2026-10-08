import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import rag
from app.agent import run_agent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


app = FastAPI(
    title="RAG FastAPI Chatbot",
    version="1.0"
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=rag.TOP_K, ge=1, le=5)


# --------------------------------------------------
# STARTUP
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    logging.info("Initializing RAG...")
    rag.initialize_rag()
    logging.info("RAG initialized successfully.")


# --------------------------------------------------
# STATUS
# --------------------------------------------------

@app.get("/api/status")
def status():
    if rag.index is None:
        raise RuntimeError("RAG is not initialized.")

    stats = rag.index.describe_index_stats()
    vector_count = (
        stats.get("total_vector_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "total_vector_count", 0)
    )

    return {
        "status": "ok",
        "indexed": vector_count > 0,
        "chunks": vector_count,
        "pinecone_vectors": vector_count,
        "embedding_model": rag.EMBEDDING_MODEL,
        "llm_model": rag.OLLAMA_MODEL
    }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    answer, documents, decision = run_agent(
        request.question,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "answer": answer,
        "agent_decision": decision,
        "sources": [
            {
                "page": document["page"],
                "score": document["score"],
                "excerpt": document["text"][:500],
            }
            for document in documents
        ]
    }


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "RAG FastAPI backend is running",
        "frontend": "Run `streamlit run streamlit_app.py` from the project folder.",
        "docs": "/docs",
    }