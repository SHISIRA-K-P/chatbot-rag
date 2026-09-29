from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import rag


PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

app = FastAPI(title="Papertrail RAG Chatbot")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=rag.TOP_K, ge=1, le=5)


@app.on_event("startup")
def start_rag():
    rag.initialize_rag()


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    indexed = rag.collection is not None and rag.collection.count() > 0
    return {
        "indexed": indexed,
        "chunks": rag.collection.count() if indexed else 0,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    if rag.collection is None or rag.collection.count() == 0:
        raise HTTPException(status_code=503, detail="The document index is not ready.")

    try:
        answer, documents = rag.ask_rag(request.question, top_k=request.top_k)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not generate an answer: {error}") from error

    sources = [
        {"page": document["page"], "excerpt": document["text"][:320]}
        for document in documents
    ]
    return {"answer": answer, "sources": sources}