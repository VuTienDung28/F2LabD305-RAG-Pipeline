"""Task 7: deterministic Reciprocal Rank Fusion (RRF) reranking.

RRF combines rankings without comparing incompatible score scales.  Dense
cosine scores and BM25 scores can therefore contribute equally based on rank:

    RRF(document) = sum(1 / (k + rank))

The fused score is only a ranking signal.  Task 9 must use the original dense
cosine score, not the RRF score, when deciding whether to invoke a fallback.
"""

from __future__ import annotations

from typing import Any


DEFAULT_RRF_K = 60


def _document_key(item: dict[str, Any]) -> str:
    """Create a stable identity shared by dense and lexical result objects."""
    metadata = item.get("metadata") or {}
    source = metadata.get("source") or metadata.get("filename")
    chunk_index = metadata.get("chunk_index")

    if source is not None and chunk_index is not None:
        return f"{source}::{chunk_index}"
    if item.get("id") is not None:
        return str(item["id"])

    # Content is the last-resort identifier used by the starter task examples.
    return " ".join(str(item.get("content", "")).split()).casefold()


def rerank_rrf(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int = 5,
    k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse one or more ranked lists with Reciprocal Rank Fusion.

    Duplicate documents inside one input list count only once.  The returned
    item preserves its original content and metadata, while ``score`` and
    ``rrf_score`` contain the fused ranking score.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("RRF smoothing constant k must be non-negative.")

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict) or not item.get("content"):
                continue

            key = _document_key(item)
            if not key or key in seen_in_list:
                continue
            seen_in_list.add(key)

            if key not in first_seen:
                first_seen[key] = seen_counter
                seen_counter += 1
                items[key] = dict(item)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            # Prefer the representation with the better original retrieval
            # score when two retrievers return the same chunk.
            current_score = float(items[key].get("score", float("-inf")))
            candidate_score = float(item.get("score", float("-inf")))
            if candidate_score > current_score:
                items[key] = dict(item)

    ordered_keys = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key]),
    )

    results: list[dict[str, Any]] = []
    for key in ordered_keys[:top_k]:
        result = dict(items[key])
        result["original_score"] = result.get("score")
        result["score"] = scores[key]
        result["rrf_score"] = scores[key]
        results.append(result)
    return results


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict[str, Any]]:
    """Apply the selected reranker to a candidate list.

    This Role 3 implementation deliberately chooses the no-API RRF option from
    the assignment.  With one list, RRF preserves the retriever's ordering;
    Task 9 performs the meaningful fusion by passing dense and BM25 lists to
    :func:`rerank_rrf` directly.
    """
    del query  # RRF uses rank positions rather than query text directly.

    if method.casefold() != "rrf":
        raise ValueError("This implementation supports method='rrf'.")
    return rerank_rrf([candidates], top_k=top_k)


if __name__ == "__main__":
    dense = [
        {"content": "Study room booking guide", "score": 0.82, "metadata": {"source": "a", "chunk_index": 0}},
        {"content": "Library events", "score": 0.65, "metadata": {"source": "b", "chunk_index": 0}},
    ]
    lexical = [
        {"content": "Study room booking guide", "score": 4.2, "metadata": {"source": "a", "chunk_index": 0}},
        {"content": "Alumni card policy", "score": 2.1, "metadata": {"source": "c", "chunk_index": 0}},
    ]
    for result in rerank_rrf([dense, lexical], top_k=3):
        print(f"[{result['score']:.6f}] {result['content']}")
