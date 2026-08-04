def test_chunk_metadata_is_preserved():
    from src.task4_chunking_indexing import chunk_documents

    chunks = chunk_documents([{
        "content": "Library access " * 100,
        "metadata": {
            "source": "access.md",
            "source_url": "https://example.test",
            "title": "Access",
            "type": "news",
        },
    }])

    assert chunks
    assert chunks[0]["metadata"]["source_url"] == "https://example.test"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_openai_embeddings_use_1024_dimensions(monkeypatch):
    import src.task4_chunking_indexing as indexing

    request = {}

    class Embeddings:
        def create(self, **kwargs):
            request.update(kwargs)
            item = type("Item", (), {"embedding": [0.0] * 1024})
            return type("Response", (), {"data": [item()]})

    client = type("Client", (), {"embeddings": Embeddings()})()
    monkeypatch.setattr(indexing, "get_embedding_client", lambda: client)

    vectors = indexing.embed_texts(["library access"])

    assert request["dimensions"] == 1024
    assert len(vectors[0]) == 1024


def test_openai_embeddings_are_batched(monkeypatch):
    import src.task4_chunking_indexing as indexing

    requests = []

    class Embeddings:
        def create(self, **kwargs):
            requests.append(kwargs["input"])
            items = [type("Item", (), {"embedding": [float(i)]})() for i, _ in enumerate(kwargs["input"])]
            return type("Response", (), {"data": items})()

    client = type("Client", (), {"embeddings": Embeddings()})()
    monkeypatch.setattr(indexing, "get_embedding_client", lambda: client)
    monkeypatch.setattr(indexing, "EMBEDDING_BATCH_SIZE", 2)

    vectors = indexing.embed_texts(["one", "two", "three"])

    assert requests == [["one", "two"], ["three"]]
    assert len(vectors) == 3


def test_rrf_rewards_documents_found_by_both_rankers():
    from src.task7_reranking import rerank_rrf

    shared = {"content": "shared", "score": 0.5, "metadata": {}}
    result = rerank_rrf([
        [shared, {"content": "dense", "score": 0.4, "metadata": {}}],
        [shared, {"content": "sparse", "score": 2.0, "metadata": {}}],
    ], top_k=2)

    assert result[0]["content"] == "shared"


def test_rerank_empty_candidates():
    from src.task7_reranking import rerank

    assert rerank("library", [], top_k=3) == []


def test_retrieval_diagnostics_capture_stages(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    monkeypatch.setattr(pipeline, "semantic_search", lambda *args, **kwargs: [
        {"content": "dense", "score": 0.9, "metadata": {}}
    ])
    monkeypatch.setattr(pipeline, "lexical_search", lambda *args, **kwargs: [
        {"content": "sparse", "score": 2.0, "metadata": {}}
    ])
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda *args, **kwargs: [
        {"content": "fused", "score": 0.8, "metadata": {}}
    ])
    monkeypatch.setattr(pipeline, "rerank", lambda *args, **kwargs: [
        {"content": "ranked", "score": 0.7, "metadata": {}}
    ])
    diagnostics = {}

    pipeline.retrieve("library", top_k=1, diagnostics=diagnostics)

    assert diagnostics["counts"] == {"dense": 1, "lexical": 1, "fused": 1, "final": 1}
    assert diagnostics["best_dense_score"] == 0.9
    assert diagnostics["fallback"]["attempted"] is False
    assert diagnostics["timings_ms"]["total"] >= 0


def test_generation_diagnostics_include_models(monkeypatch):
    import src.task10_generation as generation

    retrieval_diagnostics = {"mode": "hybrid", "timings_ms": {"total": 1.0}}

    def fake_retrieve(*args, **kwargs):
        kwargs["diagnostics"].update(retrieval_diagnostics)
        return []

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    result = generation.generate_with_citation("Unknown policy")

    assert result["diagnostics"]["embedding_model"] == "text-embedding-3-small"
    assert result["diagnostics"]["generation_model"]
    assert result["diagnostics"]["timings_ms"]["total"] >= 0


def test_generation_refuses_without_evidence(monkeypatch):
    import src.task10_generation as generation

    monkeypatch.setattr(generation, "retrieve", lambda *args, **kwargs: [])
    result = generation.generate_with_citation("Unknown policy")

    assert result["sources"] == []
    assert "verify" in result["answer"].lower() or "xác minh" in result["answer"].lower()


def test_context_contains_source_url():
    from src.task10_generation import format_context

    context = format_context([{
        "content": "Access",
        "score": 0.9,
        "metadata": {
            "source": "access.md",
            "source_url": "https://example.test",
            "type": "news",
        },
    }])

    assert "https://example.test" in context


def test_streamlit_frontend_assets_are_separated():
    from pathlib import Path

    app_source = Path("app.py").read_text(encoding="utf-8")
    stylesheet = Path("assets/styles.css").read_text(encoding="utf-8")

    assert "<style>" not in app_source
    assert "styles.css" in app_source
    assert "def render_header()" in app_source
    assert "def render_welcome()" in app_source
    assert '[data-testid="stAppViewContainer"]' in stylesheet


def test_streamlit_uses_dark_theme():
    from pathlib import Path

    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")

    assert 'base = "dark"' in config
    assert 'backgroundColor = "#09090b"' in config
    assert 'textColor = "#fafafa"' in config


def test_golden_dataset_has_bilingual_15_cases():
    from group_project.evaluation.eval_pipeline import load_golden_dataset

    data = load_golden_dataset()
    assert len(data) >= 15
    assert {item["language"] for item in data} == {"vi", "en"}
    assert all(item["source_urls"] for item in data)


def test_export_results_writes_metrics(tmp_path, monkeypatch):
    import group_project.evaluation.eval_pipeline as evaluation

    monkeypatch.setattr(evaluation, "RESULTS_PATH", tmp_path / "results.md")
    comparison = {
        "dense_only": {"summary": {"faithfulness": 0.5}, "rows": []},
        "hybrid": {"summary": {"faithfulness": 0.8}, "rows": []},
    }
    path = evaluation.export_results(comparison)
    text = path.read_text(encoding="utf-8")

    assert "dense_only" in text
    assert "hybrid" in text
    assert "Faithfulness" in text
    assert "A/B Analysis" in text
    assert "Recommendations" in text
