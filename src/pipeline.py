import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

from src.retriever import load_index

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Configuration ──────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K        = 3

# ── Custom RAG Prompt ──────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    "You are a helpful assistant that answers questions strictly based on "
    "the provided context documents.\n\n"
    "Context:\n"
    "──────────────────────────────────────\n"
    "{context_str}\n"
    "──────────────────────────────────────\n\n"
    "Question: {query_str}\n\n"
    "Instructions:\n"
    "- Answer only using the context above.\n"
    "- If the answer isn't in the context, say: "
    "'I couldn't find this in the provided documents.'\n"
    "- Be concise and cite which document the info came from.\n\n"
    "Answer:"
)

# ── Build Query Engine ─────────────────────────────────────
def build_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """Build the RAG query engine with Claude as the LLM."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")

    llm = Groq(
        model=GROQ_MODEL,
        api_key=api_key,
    )
    Settings.llm = llm

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=RAG_PROMPT,
        streaming=False,
    )

    print(f"✅ Query engine ready — using {GROQ_MODEL}")
    return query_engine


# ── Ask a Question ─────────────────────────────────────────
def ask(query_engine: RetrieverQueryEngine, question: str) -> str:
    """Ask a question and get an answer from your documents."""

    print(f"\n❓ Question: {question}")
    print("⏳ Thinking...\n")

    response = query_engine.query(question)

    print("💬 Answer:")
    print("─" * 50)
    print(str(response))
    print("─" * 50)

    # Show source documents used
    if hasattr(response, "source_nodes") and response.source_nodes:
        print("\n📚 Sources used:")
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "unknown")
            page  = node.metadata.get("page_label", "?")
            print(f"   - {fname} (page {page})")

    return str(response)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    index        = load_index()
    query_engine = build_query_engine(index)

    print("\n" + "═" * 50)
    print("  RAG Chat — ask questions about your documents")
    print("  Type 'exit' or 'quit' to stop")
    print("═" * 50 + "\n")

    while True:
        try:
            question = input("❓ You: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye!")
                break

            ask(query_engine, question)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break