import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.retriever import load_index
from src.pipeline import build_query_engine, ask

import shutil
from fastapi import File, UploadFile
from typing import List

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Global State ───────────────────────────────────────────
query_engine = None   # Loaded once at startup, reused for all requests

# ── Startup & Shutdown ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once when the server starts."""
    global query_engine
    print("\n🚀 Starting RAG server...")
    print("⏳ Loading index and query engine...")
    index        = load_index()
    query_engine = build_query_engine(index)
    print("✅ Server ready!\n")
    yield
    print("🛑 Shutting down RAG server.")

# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="RAG API",
    description="Query your documents using Retrieval Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request & Response Models ──────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3      # Optional: override number of chunks retrieved

class SourceDoc(BaseModel):
    file_name: str
    page: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]
    time_taken_seconds: float

# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "RAG API is running!",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }


@app.get("/health")
def health():
    """Check if the server and pipeline are ready."""
    return {
        "status": "ok" if query_engine else "not ready",
        "pipeline_loaded": query_engine is not None,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question and get an answer from your documents."""

    if not query_engine:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()

    try:
        response = query_engine.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Extract source documents
    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append(SourceDoc(
                file_name=node.metadata.get("file_name", "unknown"),
                page=str(node.metadata.get("page_label", "?")),
            ))

    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        question=request.question,
        answer=str(response),
        sources=sources,
        time_taken_seconds=elapsed,
    )


@app.get("/stats")
def stats():
    """Return basic info about the loaded index."""
    import chromadb
    client     = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("rag_collection")
    return {
        "total_chunks": collection.count(),
        "collection_name": "rag_collection",
        "vector_db": "ChromaDB",
        "pipeline_ready": query_engine is not None,
    }
    
    @app.post("/upload")
    async def upload_documents(files: List[UploadFile] = File(...)):
        """Upload new documents, re-embed and update the index."""
        global query_engine

        saved = []
        for file in files:
            dest = os.path.join("data", file.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved.append(file.filename)
            print(f"📥 Saved: {file.filename}")

        # Re-ingest and re-embed all documents
        from src.ingest import load_documents, chunk_documents
        from src.embedder import embed_and_store

        docs  = load_documents()
        nodes = chunk_documents(docs)
        index = embed_and_store(nodes)
        query_engine = build_query_engine(index)

        return {
            "message": f"Uploaded & indexed {len(saved)} file(s): {', '.join(saved)}",
            "total_chunks": index.vector_store._collection.count()
        }