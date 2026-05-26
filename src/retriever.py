import os
import chromadb
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Configuration ──────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "rag_collection"
TOP_K           = 3   # Number of chunks to retrieve per query

# ── Load Index from ChromaDB ───────────────────────────────
def load_index() -> VectorStoreIndex:
    """Load the existing embedded index from ChromaDB."""

    # Must use same embedding model as Step 3
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        raise ValueError("ChromaDB is empty! Run `python -m src.embedder` first.")

    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"✅ Loaded index — {collection.count()} chunk(s) in ChromaDB")
    return index


# ── Retrieve Relevant Chunks ───────────────────────────────
def retrieve_chunks(index: VectorStoreIndex, query: str) -> list:
    """Return the top-K most relevant chunks for a query."""

    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes     = retriever.retrieve(query)

    print(f"\n🔍 Top {len(nodes)} chunk(s) retrieved for: '{query}'")
    print("─" * 50)
    for i, node in enumerate(nodes):
        source = node.metadata.get("file_name", "unknown")
        score  = round(node.score, 4) if node.score else "N/A"
        print(f"[{i+1}] Source: {source} | Score: {score}")
        print(f"     {node.text[:150]}...")
    print("─" * 50)

    return nodes