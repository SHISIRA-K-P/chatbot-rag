# Papertrail RAG Chatbot

A beginner-friendly local PDF chatbot. FastAPI serves the web page and answers questions using the same RAG steps as `chat2.ipynb`: split PDF text into chunks, find the closest chunks with embeddings and ChromaDB, then ask Ollama to write an answer from that context.

## Setup

Use Python 3.10 or newer. Open a terminal in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull tinyllama
```

The PDF is `data/Attention Is All You Need.pdf`. On the first server start, the app reads it, creates the chunks and embeddings, and saves them in `chroma_db/`. The first run also downloads the embedding model, so it can take a few minutes. Later starts reuse the saved index.

Start the answer API and web page:

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, type a question, and submit it. The frontend sends the question to `POST /api/chat`; the API retrieves matching passages, asks Ollama for an answer, and returns the answer with page excerpts. API docs are at `http://127.0.0.1:8000/docs`.

## API

- `GET /api/health` checks that the service is running.
- `GET /api/status` reports whether the notebook created an index.
- `POST /api/chat` accepts `{"question": "What is attention?", "top_k": 4}` and returns `answer` and `sources`.

If the app cannot connect to Ollama, check that Ollama is running and that you pulled the model with `ollama pull tinyllama`.

## Files

- `app/main.py`: FastAPI routes used by the frontend.
- `app/rag.py`: PDF loading, chunking, vector retrieval, and Ollama answer generation.
- `frontend/`: chat page, styles, and JavaScript API calls.
- `data/Attention Is All You Need.pdf`: source document.
- `chroma_db/`: saved vector index, created automatically.