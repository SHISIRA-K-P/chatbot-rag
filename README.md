# Papertrail RAG Agent

Papertrail answers questions about `Attention Is All You Need.pdf` with a local Qwen model and a Pinecone-backed retrieval pipeline. The agent decides whether to call the paper retrieval tool. The existing FastAPI endpoint and Papertrail UI remain in place.

## Agent Flow

1. FastAPI passes the user's question to `app/agent.py`.
2. The Qwen agent uses Ollama tool calling to decide whether the question is about the uploaded paper.
3. For paper questions, it calls `retrieve_paper_context` in `app/tools.py`.
4. The tool delegates to the existing `rag.retrieve_documents()` function. That function embeds the query with `intfloat/e5-base-v2` and searches Pinecone.
5. The existing `rag.generate_answer()` function asks `qwen2.5:3b` through Ollama to answer only from the retrieved passages. The API returns page numbers and excerpts for Papertrail to display.
6. For questions outside the paper's scope, the agent does not call retrieval and returns a fixed scope response instead of guessing.

Agent, tool, retrieved-passage, and final-answer details are written to the server log.

## Requirements

- Python 3.10 or newer
- A Pinecone API key
- Ollama installed and running locally
- The `qwen2.5:3b` Ollama model

Create a `.env` file in the project root with:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key
```

Install Python packages and download the Ollama model:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

The Pinecone and dotenv dependencies can also be installed directly with:

```powershell
pip install "pinecone>=5.0,<11.0" "python-dotenv>=1.0,<2.0"
```

The first server start loads the PDF and E5 embedding model. If the configured Pinecone index is empty, it chunks the PDF, embeds the chunks, and uploads them. Later starts reuse the existing Pinecone vectors.

## Run

Start Ollama if it is not already running, then from the project root run:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. FastAPI serves the Papertrail page and its assets, as well as the API. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## API

- `GET /api/status` returns the Pinecone vector count and model names.
- `POST /api/chat` accepts `{"question": "What is attention?", "top_k": 4}`.
- Chat responses include `answer`, `agent_decision`, and `sources` containing page, score, and excerpt fields.

## Code Layout

- `app/main.py`: FastAPI app, UI serving, status, and chat endpoints.
- `app/agent.py`: Ollama tool-calling agent and decision logging.
- `app/tools.py`: agent's Pinecone retrieval tool.
- `app/rag.py`: existing PDF, E5 embedding, Pinecone search, and Ollama answer functions.
- `frontend/`: unchanged Papertrail HTML, CSS, and JavaScript.
- `data/Attention Is All You Need.pdf`: source paper.