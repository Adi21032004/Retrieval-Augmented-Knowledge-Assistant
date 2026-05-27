import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from src.ingest import load_documents, chunk_documents

load_dotenv()

# ── Configuration ──────────────────────────────────────────
CHROMA_DIR        = "chroma_db"      # Where ChromaDB saves data locally
COLLECTION_NAME   = "rag_collection" # Name of the collection inside ChromaDB
EMBEDDING_MODEL   = "text-embedding-3-small"  # Best balance of cost vs quality

# ── Setup Embedding Model ──────────────────────────────────
def get_embedding_model():
    """Initialize free local HuggingFace embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2"  # Fast, high quality, free
    )
    Settings.embed_model = embed_model
    print("✅ Embedding model ready: all-MiniLM-L6-v2 (local, free)")
    return embed_model


# ── Setup ChromaDB ─────────────────────────────────────────
def get_chroma_collection():
    """Initialize ChromaDB client and collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
    )
    print(f"✅ ChromaDB ready  : '{CHROMA_DIR}/'")
    print(f"   Collection      : '{COLLECTION_NAME}'")
    print(f"   Existing chunks : {collection.count()}")
    return chroma_client, collection


# ── Embed & Store ──────────────────────────────────────────
def embed_and_store(nodes: list) -> VectorStoreIndex:
    """Embed chunks and store them in ChromaDB."""
    
    embed_model                = get_embedding_model()
    chroma_client, collection  = get_chroma_collection()
    
    # Connect LlamaIndex to ChromaDB
    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"\n⏳ Embedding {len(nodes)} chunk(s)... (this may take a moment)")
    
    # This embeds every chunk and stores it in ChromaDB
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    
    print(f"\n✅ Done! {collection.count()} chunk(s) stored in ChromaDB.")
    return index


# ── Load Existing Index ────────────────────────────────────
def load_index() -> VectorStoreIndex:
    """Load an already-embedded index from ChromaDB (no re-embedding)."""
    
    embed_model               = get_embedding_model()
    chroma_client, collection = get_chroma_collection()
    
    if collection.count() == 0:
        raise ValueError("ChromaDB is empty! Run embed_and_store() first.")
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print(f"✅ Loaded existing index with {collection.count()} chunk(s).")
    return index


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Load & chunk docs from Step 2
    docs  = load_documents()
    nodes = chunk_documents(docs)
    
    # Embed and store
    index = embed_and_store(nodes)
    
    print("\n🎉 All chunks embedded and stored successfully!")
    print(f"   Your vector DB is saved at: ./{CHROMA_DIR}/")