# Models

The embedding model is downloaded by `sentence-transformers` on first ingestion and cached by the library. The default is `sentence-transformers/all-MiniLM-L6-v2`.

The answer model runs locally through Ollama. Install Ollama, then run:

```powershell
ollama pull tinyllama
```

Model weights are intentionally not committed to this repository.

# venv\Scripts\activate
# uvicorn app.main:app --reload