"""Task 9 — Dense/hybrid retrieval with PageIndex fallback."""

import os
from time import perf_counter
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
    diagnostics: dict | None = None,
) -> list[dict]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be 'dense' or 'hybrid'")
    started = perf_counter()
    timings = {}

    stage_started = perf_counter()
    dense_results = semantic_search(
        query,
        top_k=top_k * 2,
        use_query_expansion=use_query_expansion,
    )
    timings["dense"] = round((perf_counter() - stage_started) * 1000, 1)

    sparse_results = []
    stage_started = perf_counter()
    if mode == "dense":
        merged = [dict(item) for item in dense_results]
    else:
        sparse_results = lexical_search(query, top_k=top_k * 2)
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    timings["lexical_and_fusion"] = round((perf_counter() - stage_started) * 1000, 1)

    for item in merged:
        item["source"] = "hybrid"
    stage_started = perf_counter()
    final_results = (
        rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        if use_reranking and merged
        else merged[:top_k]
    )
    timings["reranking"] = round((perf_counter() - stage_started) * 1000, 1)

    best_dense_score = float(dense_results[0]["score"]) if dense_results else 0.0
    fallback = {"attempted": False, "used": False, "status": "not_needed"}
    if best_dense_score < score_threshold:
        fallback["attempted"] = True
        fallback["status"] = "no_results"
        stage_started = perf_counter()
        try:
            pageindex_results = pageindex_search(query, top_k=top_k)
            if pageindex_results:
                final_results = pageindex_results
                fallback.update({"used": True, "status": "success"})
        except Exception as error:
            fallback["status"] = "unavailable"
            warnings.warn(f"PageIndex fallback unavailable: {error}", RuntimeWarning)
        timings["pageindex"] = round((perf_counter() - stage_started) * 1000, 1)

    final_results = final_results[:top_k]
    timings["total"] = round((perf_counter() - started) * 1000, 1)
    if diagnostics is not None:
        diagnostics.update({
            "mode": mode,
            "top_k": top_k,
            "query_expansion": use_query_expansion,
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "reranking": RERANK_METHOD if use_reranking else "disabled",
            "result_score_type": "pageindex_rank" if fallback["used"] else (
                "jina_relevance" if os.getenv("JINA_API_KEY") and use_reranking else "rrf_rank"
            ),
            "counts": {
                "dense": len(dense_results),
                "lexical": len(sparse_results),
                "fused": len(merged),
                "final": len(final_results),
            },
            "best_dense_score": best_dense_score,
            "score_threshold": score_threshold,
            "fallback": fallback,
            "timings_ms": timings,
        })
    return final_results


if __name__ == "__main__":
    for result in retrieve("How do I book a library study room?", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
