"""Task 5 — Dense semantic retrieval with optional OpenRouter query expansion."""

import os

from dotenv import load_dotenv

from .task4_chunking_indexing import embed_texts, get_collection

load_dotenv()


def expand_query(query: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for query expansion")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=30)
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Rewrite the bilingual user question as one concise RMIT library search query. Return only the query.",
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenRouter returned an empty expanded query")
    return content.strip()


def semantic_search(query: str, top_k: int = 10, use_query_expansion: bool = False) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    search_query = expand_query(query) if use_query_expansion else query
    collection = get_collection()
    if collection.count() == 0:
        return []
    vector = embed_texts([search_query])[0]
    raw = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    results = [
        {
            "content": content,
            "score": float(max(0.0, 1.0 - distance)),
            "metadata": metadata or {},
        }
        for content, metadata, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("How do I book a library study room?", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
