from pathlib import Path

import chromadb
from ollama import chat
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "data" / "Attention Is All You Need.pdf"
CHROMA_PATH = PROJECT_DIR / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "tinyllama"
COLLECTION_NAME = "attention_is_all_you_need"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

embedding_model = None
collection = None


def load_pdf():
    reader = PdfReader(PDF_PATH)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            pages.append((page_number, page_text))

    return pages


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def initialize_rag():
    global embedding_model, collection

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        documents = []
        page_numbers = []

        for page_number, page_text in load_pdf():
            page_chunks = create_chunks(page_text)
            documents.extend(page_chunks)
            page_numbers.extend([page_number] * len(page_chunks))

        if not documents:
            raise ValueError(f"No text was extracted from {PDF_PATH}")

        embeddings = embedding_model.encode(
            documents,
            normalize_embeddings=True,
        )
        collection.add(
            ids=[f"chunk_{index}" for index in range(len(documents))],
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=[{"page": page} for page in page_numbers],
        )


def retrieve_documents(question, top_k=TOP_K):
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [
        {"text": document, "page": metadata.get("page", "not available")}
        for document, metadata in zip(documents, metadatas)
    ]


def generate_answer(question, documents):
    context = "\n\n".join(document["text"] for document in documents)
    prompt = f"""
You are a helpful question-answering assistant.

Answer the question using ONLY the information provided in the context.
Do not make up information.
If the answer is not available in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask_rag(question, top_k=TOP_K):
    documents = retrieve_documents(question, top_k=top_k)
    answer = generate_answer(question, documents)
    return answer, documents