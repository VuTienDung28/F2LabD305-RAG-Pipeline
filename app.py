"""Streamlit chatbot for the RMIT Vietnam Library RAG pipeline."""

from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.task10_generation import generate_with_citation

st.set_page_config(page_title="RMIT Library Assistant", page_icon="R", layout="wide")
st.html(Path(__file__).parent / "assets" / "styles.css")

SCORE_LABELS = {
    "rrf_rank": "RRF rank score",
    "jina_relevance": "Jina relevance",
    "pageindex_rank": "PageIndex rank score",
}


def render_header() -> None:
    st.html(
        """
        <div class="app-eyebrow">RMIT Vietnam Library</div>
        <div class="app-title">Ask the library, with evidence.</div>
        <div class="app-subtitle">A bilingual RAG assistant that retrieves official sources, explains its pipeline and cites every grounded answer.</div>
        <div class="status-row">
          <span class="status-pill">Vietnamese + English</span>
          <span class="status-pill">Official RMIT sources</span>
          <span class="status-pill">Hybrid retrieval</span>
          <span class="status-pill">Inspectable pipeline</span>
        </div>
        """
    )


def render_welcome() -> None:
    st.html(
        """
        <div class="welcome-card">
          <strong>Start with a library question</strong><br>
          <span class="welcome-description">Choose a demo question from the sidebar or ask about borrowing, study rooms, databases and research support.</span>
        </div>
        """
    )


def render_sources(sources: list[dict], score_type: str = "rrf_rank") -> None:
    if not sources:
        return
    score_label = SCORE_LABELS.get(score_type, "Retrieval score")
    with st.expander(f"Nguồn tham khảo / Sources · {len(sources)}", expanded=False):
        st.caption(f"{score_label} dùng để xếp hạng tương đối, không phải xác suất hay confidence.")
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source", "Unknown source")
            url = metadata.get("source_url", "")
            with st.container(border=True):
                heading, score = st.columns([5, 1])
                heading.markdown(f"**[Source {index}] · [{title}]({url})**" if url else f"**[Source {index}] · {title}**")
                score.markdown(f"`{float(source.get('score', 0)):.4f}`")
                st.caption(f"{metadata.get('type', 'unknown')} · {metadata.get('source', '')}")
                excerpt = source.get("content", "").strip()
                st.markdown(f"> {excerpt[:320]}{'…' if len(excerpt) > 320 else ''}")


def render_diagnostics(diagnostics: dict, sources: list[dict]) -> None:
    if not diagnostics:
        return
    counts = diagnostics.get("counts", {})
    timings = diagnostics.get("timings_ms", {})
    fallback = diagnostics.get("fallback", {})
    score_type = diagnostics.get("result_score_type", "rrf_rank")
    with st.expander("Chi tiết pipeline / Pipeline details", expanded=False):
        total, best_score, result_count, retrieval_mode = st.columns(4)
        total.metric("Total latency", f"{float(timings.get('total', 0)) / 1000:.2f}s")
        best_score.metric("Best dense similarity", f"{float(diagnostics.get('best_dense_score', 0)):.4f}")
        result_count.metric("Final contexts", counts.get("final", len(sources)))
        retrieval_mode.metric("Retrieval mode", str(diagnostics.get("mode", "unknown")).title())

        overview, latency, citations = st.tabs(["Overview", "Stage latency", "Citation map"])
        with overview:
            st.markdown("**OpenAI Embedding → Chroma + BM25 → RRF → Rerank → PageIndex → OpenRouter**")
            st.caption(
                f"Embedding `{diagnostics.get('embedding_model', 'unknown')}` · "
                f"Generation `{diagnostics.get('generation_model', 'unknown')}` · "
                f"Final score `{SCORE_LABELS.get(score_type, score_type)}`"
            )
            st.table({
                "Stage": ["Dense", "BM25", "RRF fused", "Final context"],
                "Results": [
                    counts.get("dense", 0),
                    counts.get("lexical", 0),
                    counts.get("fused", 0),
                    counts.get("final", len(sources)),
                ],
            })
            st.caption(
                f"PageIndex `{fallback.get('status', 'not_needed')}` · "
                f"dense threshold `{float(diagnostics.get('score_threshold', 0)):.2f}` · "
                f"query expansion `{diagnostics.get('query_expansion', False)}` · "
                f"memory `{diagnostics.get('history_messages', 0)}/6`"
            )
        with latency:
            stage_names = {
                "dense": "Dense retrieval",
                "lexical_and_fusion": "BM25 + RRF",
                "reranking": "Reranking",
                "pageindex": "PageIndex fallback",
                "generation": "LLM generation",
                "total": "Total",
            }
            st.table({
                "Stage": [stage_names.get(stage, stage) for stage in timings],
                "Latency (ms)": [timings[stage] for stage in timings],
            })
        with citations:
            if not sources:
                st.caption("No sources were selected for this answer.")
            for index, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                title = metadata.get("title") or metadata.get("source", "Unknown source")
                url = metadata.get("source_url", "")
                st.markdown(f"`[Source {index}]` → [{title}]({url})" if url else f"`[Source {index}]` → {title}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown("### RMIT Library Assistant")
    st.caption("Grounded answers from official RMIT Vietnam sources")
    st.divider()
    st.markdown("**Retrieval settings**")
    top_k = st.slider("Number of contexts", 3, 10, 5, help="Maximum passages sent to the generation model.")
    use_query_expansion = st.toggle(
        "Query expansion",
        value=False,
        help="Rewrite short or ambiguous questions before retrieval.",
    )
    st.caption("Hybrid · Dense + BM25 · RRF · PageIndex fallback")
    st.divider()
    st.markdown("**Demo questions**")
    suggestions = [
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên được mượn bao nhiêu sách?",
        "How can I access library databases?",
        "What research support does the library provide?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_query = suggestion
    st.divider()
    if st.button("Clear conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

render_header()

if not st.session_state.messages:
    render_welcome()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        diagnostics = message.get("diagnostics", {})
        render_sources(message.get("sources", []), diagnostics.get("result_score_type", "rrf_rank"))
        render_diagnostics(diagnostics, message.get("sources", []))

user_input = st.chat_input("Hỏi về thư viện RMIT / Ask the RMIT Library")
query = user_input or st.session_state.pending_query
if query:
    st.session_state.pending_query = None
    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching official RMIT sources…"):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    history=history,
                    use_query_expansion=use_query_expansion,
                )
                answer, sources = response["answer"], response.get("sources", [])
                diagnostics = response.get("diagnostics", {})
            except Exception:
                answer, sources, diagnostics = "The pipeline is temporarily unavailable. Please try again.", [], {}
        st.markdown(answer)
        render_sources(sources, diagnostics.get("result_score_type", "rrf_rank"))
        render_diagnostics(diagnostics, sources)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "diagnostics": diagnostics,
    })
