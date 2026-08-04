"""Task 9 — Dense/hybrid retrieval with PageIndex fallback."""

import warnings

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    mode: str = "hybrid",
    use_query_expansion: bool = False,
) -> list[dict]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be 'dense' or 'hybrid'")
    dense_results = semantic_search(
        query,
        top_k=top_k * 2,
        use_query_expansion=use_query_expansion,
    )
    if mode == "dense":
        merged = [dict(item) for item in dense_results]
    else:
        sparse_results = lexical_search(query, top_k=top_k * 2)
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"
    final_results = (
        rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        if use_reranking and merged
        else merged[:top_k]
    )
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as error:
            warnings.warn(f"PageIndex fallback unavailable: {error}", RuntimeWarning)
    return final_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("How do I book a library study room?", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
