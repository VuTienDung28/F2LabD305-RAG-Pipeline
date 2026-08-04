"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5


def _with_source(results: list[dict], source: str, top_k: int) -> list[dict]:
    '''Copy and normalize results before handing them to Task 10.'''
    normalized = []
    for item in results[:top_k]:
        result = item.copy()
        result.setdefault('metadata', {})
        result['source'] = source
        normalized.append(result)
    return normalized


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: True dùng Dense + Sparse qua RRF; False dùng dense-only
            làm baseline A/B.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    query = query.strip()
    candidate_k = top_k * 2

    # P0 runs the independent retrievers sequentially for easier debugging.
    # They can be parallelized after Task 5 and Task 6 are stable/thread-safe.
    dense_failed = False
    try:
        dense_results = semantic_search(query, top_k=candidate_k) or []
    except Exception as exc:
        print(f'Dense retrieval failed; continuing with sparse: {exc}')
        dense_results = []
        dense_failed = True

    try:
        sparse_results = lexical_search(query, top_k=candidate_k) or []
    except Exception as exc:
        print(f'Sparse retrieval failed; continuing with dense: {exc}')
        sparse_results = []

    # Fallback uses the original dense cosine score, never the RRF score.
    best_dense_score = dense_results[0].get('score', 0.0) if dense_results else 0.0
    if not dense_failed and best_dense_score < score_threshold:
        try:
            fallback_results = pageindex_search(query, top_k=top_k) or []
        except Exception as exc:
            print(f'  PageIndex fallback failed: {exc}')
            fallback_results = []

        if fallback_results:
            return _with_source(fallback_results, 'pageindex', top_k)

    if use_reranking:
        if not dense_results and not sparse_results:
            return []
        try:
            final_results = rerank_rrf(
                [dense_results, sparse_results],
                top_k=candidate_k,
            )
        except Exception as exc:
            print(f'RRF failed; using available retriever results: {exc}')
            final_results = dense_results or sparse_results
    else:
        # Dense-only A/B baseline. If dense and PageIndex are both unavailable,
        # keep sparse results as graceful degradation.
        final_results = dense_results or sparse_results

    return _with_source(final_results, 'hybrid', top_k)


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
