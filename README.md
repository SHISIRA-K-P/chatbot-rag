# Papertrail RAG Agent

Papertrail answers questions about `Attention Is All You Need.pdf` with a local Qwen model and a Pinecone-backed retrieval pipeline. A Streamlit frontend communicates with the FastAPI agent API.

## Agent Flow

1. Streamlit sends the user's question to FastAPI's `POST /api/chat` endpoint.
2. `app/agent.py` asks Qwen through Ollama whether the paper retrieval tool is needed.
3. For paper questions, the agent calls `retrieve_paper_context` in `app/tool.py`.
4. The tool delegates to the existing `rag.retrieve_documents()` function, which embeds the query with `intfloat/e5-base-v2` and searches Pinecone.
5. `rag.generate_answer()` asks `qwen2.5:3b` through Ollama to answer only from the retrieved passages. FastAPI returns the answer and source page excerpts for Streamlit to display.
6. For questions outside the paper's scope, the agent skips retrieval and returns a scope response rather than guessing.

Agent decisions, tool calls, retrieved passages, and final answers are written to the FastAPI log.

## Requirements

- Python 3.10 or newer
- A Pinecone API key
- Ollama installed and running locally with `qwen2.5:3b`

Create a `.env` file in the project root:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key
```

Install the project dependencies and download the Ollama model:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

To install Streamlit by itself in an existing environment:

```powershell
pip install "streamlit>=1.35,<2.0"
```

## One-time document indexing

Document processing is separate from application startup. Run `notebooks/01_pdf_processing.ipynb` manually to extract the PDF text, create chunks, generate E5 passage embeddings, and upload them to the Pinecone index. Run it again only when the source PDF changes. Its upload step replaces vectors in the target index's default namespace so that old chunks do not remain after a document update.

The notebook and runtime use `intfloat/e5-base-v2` (768 dimensions), `passage:` for document embeddings, and `query:` for searches. Pinecone uses the `attention-is-all-you-need` serverless index in AWS `us-east-1` with cosine similarity.

Application startup only loads the query embedding model, connects to the existing Pinecone index, and verifies that it contains vectors. It does not read the PDF, create chunks, generate document embeddings, or upload vectors. Startup gives a clear error if you have not run the indexing notebook or the index is empty.

To run the notebook locally from the project root, activate the virtual environment, install the dependencies if needed, then launch Jupyter:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook notebooks/01_pdf_processing.ipynb
```

## Run

Start Ollama if it is not already running. In the first terminal, from the project root, start FastAPI:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

In a second terminal, also from the project root, start Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal, usually `http://localhost:8501`. The FastAPI API and interactive documentation remain available at `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

If FastAPI runs on a different address, set `RAG_API_URL` before starting Streamlit. For example:

```powershell
$env:RAG_API_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

## API

- `GET /api/status` returns the Pinecone vector count and model names.
- `POST /api/chat` accepts `{"question": "What is attention?", "top_k": 4}`.
- Chat responses include `answer`, `agent_decision`, and `sources` with page, score, and excerpt fields.

## Code Layout

- `streamlit_app.py`: Streamlit chat UI, status, and source excerpt display.
- `app/main.py`: FastAPI status and chat endpoints.
- `app/agent.py`: Ollama tool-calling agent and decision logging.
- `app/tool.py`: agent's Pinecone retrieval tool.
- `app/rag.py`: runtime E5 query embedding, Pinecone retrieval, and Ollama answer functions.
- `notebooks/01_pdf_processing.ipynb`: one-time PDF extraction, chunking, embedding, and Pinecone upload.
- `frontend/`: previous static UI assets; not used by the Streamlit app.
- `data/Attention Is All You Need.pdf`: source paper.






Questions

1. What is the Transformer architecture?

2. What is self-attention?

3. Explain multi-head attention.

4. What are Query, Key and Value?

5. What is positional encoding?

6. Why is positional encoding required?

7. What is scaled dot-product attention?

8. Why is the dot product divided by the square root of dk?

9. How many encoder layers are used in the original Transformer?

10. What is masked self-attention?

11. What is the role of the feed-forward network?

12. Why does the Transformer use residual connections?

13. What are the advantages of the Transformer over RNNs?

14. How does the decoder generate the output?

15. What is the main contribution of the paper?