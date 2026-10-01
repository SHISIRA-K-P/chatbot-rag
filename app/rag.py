from pathlib import Path
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from ollama import chat
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_DIR / "data" / "Attention Is All You Need.pdf"

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-base-v2"

# LLM
OLLAMA_MODEL = "qwen2.5:3b"

# Pinecone
INDEX_NAME = "attention-is-all-you-need"

# E5-base-v2 embedding dimension
EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in the .env file."
    )


# --------------------------------------------------
# PINECONE CLIENT
# --------------------------------------------------

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------

embedding_model = None
index = None


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

def load_pdf():

    reader = PdfReader(PDF_PATH)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:
            pages.append(
                (page_number, page_text)
            )

    return pages


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(
            " ".join(words[start:end])
        )

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# INITIALIZE RAG
# --------------------------------------------------

def initialize_rag():

    global embedding_model
    global index

    # Load embedding model
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        f"Loaded embedding model: {EMBEDDING_MODEL}"
    )

    # Get existing Pinecone indexes
    existing_indexes = pc.list_indexes().names()

    # Create index if it doesn't exist
    if INDEX_NAME not in existing_indexes:

        print("Creating Pinecone index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for index to become ready
        while not pc.describe_index(
            INDEX_NAME
        ).status["ready"]:

            time.sleep(1)

    # Connect to Pinecone index
    index = pc.Index(INDEX_NAME)

    # Check existing vectors
    stats = index.describe_index_stats()

    vector_count = stats.get(
        "total_vector_count",
        0
    )

    print(
        f"Existing vectors in Pinecone: {vector_count}"
    )

    # --------------------------------------------------
    # INSERT DOCUMENTS ONLY IF INDEX IS EMPTY
    # --------------------------------------------------

    if vector_count == 0:

        print("Loading PDF...")

        documents = []
        page_numbers = []

        for page_number, page_text in load_pdf():

            page_chunks = create_chunks(
                page_text
            )

            documents.extend(
                page_chunks
            )

            page_numbers.extend(
                [page_number] *
                len(page_chunks)
            )

        if not documents:

            raise ValueError(
                f"No text was extracted from {PDF_PATH}"
            )

        print(
            f"Created {len(documents)} chunks."
        )

        # --------------------------------------------------
        # E5 DOCUMENT EMBEDDINGS
        # --------------------------------------------------

        # E5 models work best when documents use
        # the "passage:" prefix.

        passages = [
            f"passage: {document}"
            for document in documents
        ]

        embeddings = embedding_model.encode(
            passages,
            normalize_embeddings=True
        )

        # --------------------------------------------------
        # CREATE PINECONE VECTORS
        # --------------------------------------------------

        vectors = []

        for index_number, (
            document,
            embedding,
            page
        ) in enumerate(
            zip(
                documents,
                embeddings,
                page_numbers
            )
        ):

            vectors.append(
                {
                    "id": f"chunk_{index_number}",

                    "values": embedding.tolist(),

                    "metadata": {
                        "text": document,
                        "page": page
                    }
                }
            )

        # --------------------------------------------------
        # UPLOAD TO PINECONE
        # --------------------------------------------------

        index.upsert(
            vectors=vectors
        )

        print(
            f"Uploaded {len(vectors)} vectors to Pinecone."
        )

    else:

        print(
            "Pinecone index already contains vectors."
        )


# --------------------------------------------------
# RETRIEVE DOCUMENTS
# --------------------------------------------------

def retrieve_documents(
    question,
    top_k=TOP_K
):

    # E5 query prefix
    query = f"query: {question}"

    question_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = index.query(
        vector=question_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    documents = []

    for match in results["matches"]:

        metadata = match.get(
            "metadata",
            {}
        )

        documents.append(
            {
                "text": metadata.get(
                    "text",
                    ""
                ),

                "page": metadata.get(
                    "page",
                    "not available"
                ),

                "score": match.get(
                    "score",
                    0
                )
            }
        )

    return documents


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(
    question,
    documents
):

    context = "\n\n".join(
        document["text"]
        for document in documents
    )

    prompt = f"""
    You are a helpful question-answering assistant.

    Answer the question using ONLY the information
    provided in the context.

    Do not use outside knowledge.

    Do not make up information.

    If the answer is not available in the context,
    say:

    "I could not find the answer in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = chat(
        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ]["content"].strip()


# --------------------------------------------------
# ASK RAG
# --------------------------------------------------

def ask_rag(
    question,
    top_k=TOP_K
):

    documents = retrieve_documents(
        question,
        top_k=top_k
    )

    answer = generate_answer(
        question,
        documents
    )

    return answer, documents