from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import rag


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
            "message": str(e)
        }

    return {
        "status": "ok",
        "pinecone_vectors": vector_count,
        "embedding_model": rag.EMBEDDING_MODEL,
        "llm_model": rag.OLLAMA_MODEL
    }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    answer, documents = rag.ask_rag(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "page": document["page"],
                "score": document["score"]
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