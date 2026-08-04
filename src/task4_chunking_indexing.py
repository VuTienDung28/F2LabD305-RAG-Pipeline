"""Task 4 — Chunk standardized Markdown and index it in ChromaDB."""

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "rmit_vietnam_library"


def _metadata_from_markdown(path: Path, content: str) -> dict:
    lines = content.splitlines()
    title = lines[0].removeprefix("# ").strip() if lines else path.stem
    fields = {}
    for line in lines[:12]:
        if line.startswith("**"):
            key, separator, value = line.replace("**", "").partition(":")
            if separator:
                fields[key.casefold()] = value.strip()
    return {
        "source": path.name,
        "document_path": path.relative_to(STANDARDIZED_DIR).as_posix(),
        "source_url": fields.get("source", ""),
        "title": title or path.stem,
        "type": fields.get("type", path.parent.name),
    }


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({"content": content, "metadata": _metadata_from_markdown(path, content)})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for document in documents:
        split_documents = splitter.create_documents(
            [document["content"]],
            metadatas=[document.get("metadata", {})],
        )
        for index, split in enumerate(split_documents):
            chunks.append({
                "content": split.page_content,
                "metadata": {
                    **split.metadata,
                    "chunk_index": index,
                    "chunk_start": split.metadata["start_index"],
                },
            })
    return chunks


@lru_cache(maxsize=1)
def get_embedding_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings")
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=60)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    client = get_embedding_client()
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start:start + EMBEDDING_BATCH_SIZE],
            dimensions=EMBEDDING_DIM,
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]


@lru_cache(maxsize=1)
def get_chroma_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]) -> int:
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if not chunks:
        return 0
    collection.upsert(
        ids=[f"{chunk['metadata']['source']}::{chunk['metadata']['chunk_index']}" for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return len(chunks)


def run_pipeline() -> int:
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {STANDARDIZED_DIR}")
    chunks = embed_chunks(chunk_documents(documents))
    count = index_to_vectorstore(chunks)
    print(f"Indexed {count} chunks with {EMBEDDING_MODEL}")
    return count


if __name__ == "__main__":
    run_pipeline()
