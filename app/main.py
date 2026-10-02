import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


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

    print("Initializing RAG...")

    rag.initialize_rag()

    print("RAG initialized successfully.")


# --------------------------------------------------
# STATUS
# --------------------------------------------------

@app.get("/api/status")
def status():

    vector_count = 0

    try:
        stats = rag.index.describe_index_stats()

        vector_count = stats.get(
            "total_vector_count",
            0
        )

    except Exception as e:

        return {
            "status": "error",
            "indexed": False,
            "chunks": 0,
            "message": str(e)
        }

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
    return FileResponse(FRONTEND_DIR / "index.html")