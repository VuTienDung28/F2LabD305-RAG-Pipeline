"""Task 7 — Reciprocal Rank Fusion and optional Jina reranking."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def _identity(item: dict) -> str:
    metadata = item.get("metadata", {})
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    return f"{source}::{chunk_index}" if source is not None and chunk_index is not None else item["content"]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [candidate["content"] for candidate in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("results"), list):
        raise RuntimeError("Jina reranker returned an invalid response")
    return [
        {
            **candidates[item["index"]],
            "score": float(item["relevance_score"]),
        }
        for item in payload["results"]
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    import numpy as np

    if not candidates:
        return []

    def cosine(left, right):
        left_array, right_array = np.asarray(left), np.asarray(right)
        denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
        return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0

    selected, remaining = [], list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        best_index, best_score = None, float("-inf")
        for index in remaining:
            relevance = cosine(query_embedding, candidates[index]["embedding"])
            diversity = max(
                (cosine(candidates[index]["embedding"], candidates[chosen]["embedding"]) for chosen in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return [candidates[index] for index in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    scores, items = {}, {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            identity = _identity(item)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            items.setdefault(identity, item)
    results = []
    for identity, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        results.append({**items[identity], "score": float(score)})
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
    if method == "mmr":
        raise ValueError("Use rerank_mmr() with a query embedding")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    print(rerank("library booking", candidates, top_k=2))
