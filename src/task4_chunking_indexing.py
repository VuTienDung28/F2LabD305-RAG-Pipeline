"""Task 4: chunk standardized Markdown and index it in persistent ChromaDB."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Recursive splitting is a good fit for the mixed Markdown corpus: it preserves
# paragraphs/headings when possible and still enforces a hard character limit.
# 800 characters retain enough context for policies; 100 characters (12.5%)
# prevent facts around a boundary from being lost. These values follow LAB_GUIDE.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# OpenAI's small embedding model keeps indexing lightweight on machines without
# a GPU. Requesting 1024 dimensions preserves the collection shape expected by
# the rest of the lab while Task 4 and Task 5 share exactly the same embedder.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 64

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


def load_documents() -> list[dict[str, Any]]:
    """Load every non-empty Markdown file under ``data/standardized``."""
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Standardized data directory not found: {STANDARDIZED_DIR}")

    documents: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_source = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = relative_source.split("/", 1)[0]
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_source,
                    "filename": md_file.name,
                    "doc_type": doc_type,
                    # Keep ``type`` for compatibility with later starter tasks.
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents recursively while preserving source metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for document in documents:
        content = document.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        splits = splitter.split_text(content.strip())
        for chunk_index, chunk_text in enumerate(splits):
            text = chunk_text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **document.get("metadata", {}),
                        "chunk_index": chunk_index,
                        "chunk_characters": len(text),
                    },
                }
            )
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return one shared OpenAI client for document and query embeddings."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create OpenAI embeddings for documents or queries in API-safe batches."""
    if not texts:
        return []

    client = get_embedding_model()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
        )
        response_items = sorted(response.data, key=lambda item: item.index)
        batch_embeddings = [item.embedding for item in response_items]

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "OpenAI returned an unexpected number of embeddings: "
                f"expected {len(batch)}, got {len(batch_embeddings)}."
            )
        if any(len(vector) != EMBEDDING_DIM for vector in batch_embeddings):
            actual_dimensions = sorted({len(vector) for vector in batch_embeddings})
            raise ValueError(
                f"Unexpected embedding dimensions {actual_dimensions}; "
                f"expected {EMBEDDING_DIM} from {EMBEDDING_MODEL}."
            )

        embeddings.extend(batch_embeddings)

    return embeddings


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return chunks enriched with a normalized dense embedding."""
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def get_chroma_client():
    """Open the persistent local Chroma client."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Return the indexed collection used by Task 5."""
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    raw_id = f"{metadata['source']}::{metadata['chunk_index']}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def index_to_vectorstore(chunks: list[dict[str, Any]]):
    """Replace the Chroma collection with the current corpus and persist it."""
    if not chunks:
        raise ValueError("No chunks were provided for indexing.")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("All chunks must be embedded before indexing.")

    client = get_chroma_client()
    # Recreate only this collection so re-indexing cannot leave stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )
    collection.upsert(
        ids=[_chunk_id(chunk) for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline() -> dict[str, int]:
    """Run load -> chunk -> embed -> persistent Chroma indexing."""
    print("=" * 60)
    print("Task 4: Chunking & ChromaDB Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Vector store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 60)

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found under {STANDARDIZED_DIR}")
    print(f"[OK] Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"[OK] Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(embedded_chunks)} chunks")

    collection = index_to_vectorstore(embedded_chunks)
    print(f"[OK] Indexed {collection.count()} chunks into {COLLECTION_NAME}")
    return {"documents": len(documents), "chunks": collection.count()}


if __name__ == "__main__":
    run_pipeline()
