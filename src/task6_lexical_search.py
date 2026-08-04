"""Task 6 — BM25 lexical retrieval over the canonical Task 4 chunks."""

import re
from functools import lru_cache

from .task4_chunking_indexing import chunk_documents, load_documents

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
CORPUS: list[dict] = []


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def build_bm25_index(corpus: list[dict]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(document["content"]) for document in corpus])


@lru_cache(maxsize=1)
def _get_index():
    corpus = chunk_documents(load_documents())
    CORPUS.clear()
    CORPUS.extend(corpus)
    return build_bm25_index(CORPUS) if CORPUS else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    index = _get_index()
    if index is None:
        return []
    scores = index.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    return [
        {
            "content": CORPUS[index]["content"],
            "score": float(score),
            "metadata": CORPUS[index]["metadata"],
        }
        for index, score in ranked[:top_k]
        if score > 0
    ]


if __name__ == "__main__":
    for result in lexical_search("library study room", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
