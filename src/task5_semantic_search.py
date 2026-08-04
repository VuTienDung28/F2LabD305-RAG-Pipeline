"""Task 5: dense semantic retrieval from the Task 4 Chroma index.

The default path is fully local cosine retrieval. Optional HyDE generates a
hypothetical answer with an LLM, embeds that answer, and uses it as the search
vector to reduce the vocabulary gap between a short question and documents.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any

from .task4_chunking_indexing import embed_texts, get_collection


HYDE_MODEL = os.getenv("HYDE_MODEL", "gpt-4o-mini")
HYDE_SYSTEM_PROMPT = (
    "Write a concise hypothetical university policy or student-service passage "
    "that directly answers the question. Use the same terminology that an "
    "official university document would use. Return only the passage."
)


def generate_hypothetical_document(
    query: str,
    generator: Callable[[str], str] | None = None,
) -> str:
    """Generate the hypothetical document used by HyDE.

    ``generator`` supports local/test LLMs. Without one, the function uses the
    OpenAI client and therefore requires ``OPENAI_API_KEY``.
    """
    if generator is not None:
        hypothetical_document = generator(query)
    else:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "HyDE requires OPENAI_API_KEY or a custom hyde_generator. "
                "Call semantic_search(..., use_hyde=False) for local dense search."
            )
        response = OpenAI().chat.completions.create(
            model=HYDE_MODEL,
            temperature=0.2,
            max_tokens=250,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        hypothetical_document = response.choices[0].message.content or ""

    hypothetical_document = hypothetical_document.strip()
    if not hypothetical_document:
        raise ValueError("The HyDE generator returned an empty document.")
    return hypothetical_document


def semantic_search(
    query: str,
    top_k: int = 10,
    *,
    use_hyde: bool = False,
    hyde_generator: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most semantically similar chunks using cosine similarity.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of chunks to return.
        use_hyde: Embed an LLM-generated hypothetical document instead of the
            short query (Hypothetical Document Embeddings).
        hyde_generator: Optional callable for a local/custom HyDE generator.

    Returns:
        Dictionaries containing ``content``, cosine ``score`` in ``[0, 1]``,
        and source ``metadata``, sorted by descending score.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    search_text = query.strip()
    if use_hyde:
        search_text = generate_hypothetical_document(search_text, hyde_generator)

    query_embedding = embed_texts([search_text])[0]
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]
    distances = (raw_results.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # Chroma cosine distance is 1 - cosine similarity. Clamp to [0, 1] so
        # later pipeline thresholds have a stable and interpretable scale.
        score = min(1.0, max(0.0, 1.0 - float(distance)))
        results.append(
            {
                "content": document,
                "score": round(score, 6),
                "metadata": metadata or {},
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over ChromaDB")
    parser.add_argument("query", nargs="?", default="How can students use the library?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hyde", action="store_true")
    args = parser.parse_args()

    for result in semantic_search(args.query, top_k=args.top_k, use_hyde=args.hyde):
        source = result["metadata"].get("source", "unknown")
        print(f"[{result['score']:.4f}] {source}")
        print(f"  {result['content'][:180].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    _main()
