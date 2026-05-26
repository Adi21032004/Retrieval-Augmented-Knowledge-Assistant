import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

# ── Configuration ──────────────────────────────────────────
DATA_DIR = "data"          # Folder where your docs live
CHUNK_SIZE = 512           # Tokens per chunk
CHUNK_OVERLAP = 50         # Overlap between chunks (keeps context)

# ── Load Documents ─────────────────────────────────────────
def load_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """Load all documents from the data directory."""
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    files = list(Path(data_dir).iterdir())
    if not files:
        raise ValueError(f"No files found in '{data_dir}'. Add some documents!")
    
    print(f"📂 Found {len(files)} file(s) in '{data_dir}':")
    for f in files:
        print(f"   - {f.name}")
    
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,           # Also reads subfolders
        filename_as_id=True       # Uses filename as doc ID
    )
    
    documents = reader.load_data()
    print(f"\n✅ Loaded {len(documents)} document(s) successfully.")
    return documents


# ── Chunk Documents ────────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list:
    """Split documents into smaller chunks for embedding."""
    
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",   # Respects paragraph breaks
    )
    
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    
    print(f"\n📦 Created {len(nodes)} chunk(s) from {len(documents)} document(s).")
    print(f"   Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")
    
    return nodes


# ── Preview Chunks ─────────────────────────────────────────
def preview_chunks(nodes: list, n: int = 3):
    """Print a preview of the first N chunks."""
    print(f"\n🔍 Preview of first {n} chunk(s):\n" + "─" * 50)
    for i, node in enumerate(nodes[:n]):
        print(f"\n[Chunk {i+1}]")
        print(f"ID     : {node.node_id[:8]}...")
        print(f"Source : {node.metadata.get('file_name', 'unknown')}")
        print(f"Text   : {node.text[:200]}...")
    print("─" * 50)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    docs  = load_documents()
    nodes = chunk_documents(docs)
    preview_chunks(nodes)