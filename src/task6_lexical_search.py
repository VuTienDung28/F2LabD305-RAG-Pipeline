"""Task 6: lexical retrieval with a local BM25 index.

The corpus uses exactly the same chunks as Task 4.  This makes document
identity and metadata consistent when Task 9 fuses dense and sparse results.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import chunk_documents, load_documents


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)

# Kept as a public variable for the lab interface.  It is populated lazily so
# merely importing this module does not read files or build an index.
CORPUS: list[dict[str, Any]] = []


def tokenize(text: str) -> list[str]:
    """Return case-insensitive word tokens for English and Vietnamese text."""
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.casefold())


def build_bm25_index(corpus: list[dict[str, Any]]):
    """Build and return a BM25Okapi index for ``corpus``."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Cannot build a BM25 index from an empty corpus.")

    tokenized_corpus = [tokenize(str(document.get("content", ""))) for document in corpus]
    if not any(tokenized_corpus):
        raise ValueError("The corpus does not contain any searchable text.")
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


@lru_cache(maxsize=1)
def _get_bm25_index():
    """Load Task 4 chunks and cache one BM25 index per Python process."""
    global CORPUS

    documents = load_documents()
    CORPUS = chunk_documents(documents)
    if not CORPUS:
        raise RuntimeError("No Markdown chunks are available under data/standardized.")
    return build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search standardized chunks by exact terms using BM25.

    Only positive-score matches are returned.  Results use the common lab
    schema and are sorted from the highest BM25 score to the lowest.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25 = _get_bm25_index()
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(CORPUS)),
        key=lambda index: (-float(scores[index]), index),
    )

    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        document = CORPUS[index]
        results.append(
            {
                "content": document["content"],
                "score": score,
                "metadata": dict(document.get("metadata", {})),
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = lexical_search("library study room booking", top_k=5)
    for rank, result in enumerate(results, start=1):
        source = result["metadata"].get("source", "unknown")
        print(f"{rank}. [{result['score']:.3f}] {source}")
        print(f"   {result['content'][:120]}...")
