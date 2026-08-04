"""Streamlit chatbot for the RMIT Vietnam Library RAG pipeline."""

from collections import OrderedDict
from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.source_evidence import highlight_document, load_document
from src.task10_generation import generate_with_citation

st.set_page_config(page_title="Trợ lý Thư viện RMIT", page_icon="R", layout="wide")
st.html(Path(__file__).parent / "assets" / "styles.css")

SCORE_LABELS = {
    "rrf_rank": "Điểm xếp hạng RRF",
    "jina_relevance": "Độ liên quan Jina",
    "pageindex_rank": "Điểm xếp hạng PageIndex",
}


def render_header() -> None:
    st.html(
        """
        <div class="app-eyebrow">Thư viện RMIT Việt Nam</div>
        <div class="app-title">Hỏi thư viện, nhận câu trả lời có căn cứ.</div>
        <div class="app-subtitle">Trợ lý RAG truy xuất nguồn chính thức, giải thích quy trình và trích dẫn bằng chứng cho từng câu trả lời.</div>
        <div class="status-row">
          <span class="status-pill">Tiếng Việt + Tiếng Anh</span>
          <span class="status-pill">Nguồn chính thức từ RMIT</span>
          <span class="status-pill">Truy xuất kết hợp</span>
          <span class="status-pill">Quy trình minh bạch</span>
        </div>
        """
    )


def render_welcome() -> None:
    st.html(
        """
        <div class="welcome-card">
          <strong>Hãy bắt đầu bằng một câu hỏi về thư viện</strong><br>
          <span class="welcome-description">Chọn câu hỏi mẫu ở thanh bên hoặc hỏi về mượn sách, phòng học nhóm, cơ sở dữ liệu và hỗ trợ nghiên cứu.</span>
        </div>
        """
    )


def render_sources(sources: list[dict], score_type: str = "rrf_rank") -> None:
    if not sources:
        return
    groups = OrderedDict()
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        key = metadata.get("document_path") or f"unavailable:{index}"
        group = groups.setdefault(key, {"sources": [], "labels": []})
        group["sources"].append(source)
        group["labels"].append(index)

    score_label = SCORE_LABELS.get(score_type, "Điểm truy xuất")
    with st.expander(f"Nguồn tham khảo · {len(sources)} đoạn bằng chứng", expanded=False):
        st.caption(f"{score_label} dùng để xếp hạng tương đối, không phải xác suất hay confidence.")
        st.caption(
            "Các đoạn được tô sáng là bằng chứng mà hệ thống RAG cung cấp cho mô hình. "
            "Đây không phải bằng chứng rằng mô hình đã sử dụng từng từ trong câu trả lời."
        )
        for document_path, group in groups.items():
            source = group["sources"][0]
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source", "Nguồn không xác định")
            url = metadata.get("source_url", "")
            labels = ", ".join(f"[Source {index}]" for index in group["labels"])
            score = max(float(item.get("score", 0)) for item in group["sources"])
            with st.container(border=True):
                heading, score_column = st.columns([5, 1])
                heading.markdown(f"**{labels} · [{title}]({url})**" if url else f"**{labels} · {title}**")
                score_column.markdown(f"`{score:.4f}`")
                st.caption(f"{metadata.get('type', 'unknown')} · {metadata.get('source', '')}")
                document = load_document(document_path) if not document_path.startswith("unavailable:") else None
                if document is None:
                    st.caption("Không có tài liệu chuẩn hóa đầy đủ cho kết quả PageIndex hoặc dữ liệu cũ.")
                    continue
                rendered, highlighted = highlight_document(document, group["sources"])
                with st.expander(f"Toàn bộ tài liệu · {highlighted} đoạn được tô sáng", expanded=False):
                    if highlighted != len(group["sources"]):
                        st.caption("Một số metadata bằng chứng đã cũ. Hãy lập chỉ mục lại để khôi phục toàn bộ highlight.")
                    st.markdown(rendered, unsafe_allow_html=True)


def render_diagnostics(diagnostics: dict, sources: list[dict]) -> None:
    if not diagnostics:
        return
    counts = diagnostics.get("counts", {})
    timings = diagnostics.get("timings_ms", {})
    fallback = diagnostics.get("fallback", {})
    score_type = diagnostics.get("result_score_type", "rrf_rank")
    with st.expander("Chi tiết quy trình RAG", expanded=False):
        total, best_score, result_count, retrieval_mode = st.columns(4)
        total.metric("Tổng thời gian", f"{float(timings.get('total', 0)) / 1000:.2f}s")
        best_score.metric("Độ tương đồng dense tốt nhất", f"{float(diagnostics.get('best_dense_score', 0)):.4f}")
        result_count.metric("Ngữ cảnh cuối", counts.get("final", len(sources)))
        retrieval_mode.metric("Chế độ truy xuất", str(diagnostics.get("mode", "không rõ")).title())

        overview, latency, citations = st.tabs(["Tổng quan", "Thời gian từng bước", "Ánh xạ trích dẫn"])
        with overview:
            st.markdown("**OpenAI Embedding → Chroma + BM25 → RRF → Xếp hạng lại → PageIndex → OpenRouter**")
            st.caption(
                f"Mô hình embedding `{diagnostics.get('embedding_model', 'không rõ')}` · "
                f"Mô hình sinh câu trả lời `{diagnostics.get('generation_model', 'không rõ')}` · "
                f"Điểm cuối `{SCORE_LABELS.get(score_type, score_type)}`"
            )
            st.table({
                "Giai đoạn": ["Dense", "BM25", "Hợp nhất RRF", "Ngữ cảnh cuối"],
                "Số kết quả": [
                    counts.get("dense", 0),
                    counts.get("lexical", 0),
                    counts.get("fused", 0),
                    counts.get("final", len(sources)),
                ],
            })
            fallback_labels = {
                "not_needed": "không cần",
                "no_results": "không có kết quả",
                "success": "thành công",
                "unavailable": "không khả dụng",
            }
            fallback_status = fallback.get("status", "not_needed")
            st.caption(
                f"PageIndex `{fallback_labels.get(fallback_status, fallback_status)}` · "
                f"ngưỡng dense `{float(diagnostics.get('score_threshold', 0)):.2f}` · "
                f"mở rộng truy vấn `{diagnostics.get('query_expansion', False)}` · "
                f"bộ nhớ hội thoại `{diagnostics.get('history_messages', 0)}/6`"
            )
        with latency:
            stage_names = {
                "dense": "Truy xuất dense",
                "lexical_and_fusion": "BM25 + RRF",
                "reranking": "Xếp hạng lại",
                "pageindex": "Dự phòng PageIndex",
                "generation": "LLM sinh câu trả lời",
                "total": "Tổng cộng",
            }
            st.table({
                "Giai đoạn": [stage_names.get(stage, stage) for stage in timings],
                "Thời gian (ms)": [timings[stage] for stage in timings],
            })
        with citations:
            if not sources:
                st.caption("Không có nguồn nào được chọn cho câu trả lời này.")
            for index, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                title = metadata.get("title") or metadata.get("source", "Nguồn không xác định")
                url = metadata.get("source_url", "")
                st.markdown(f"`[Source {index}]` → [{title}]({url})" if url else f"`[Source {index}]` → {title}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown("### Trợ lý Thư viện RMIT")
    st.caption("Câu trả lời có căn cứ từ nguồn chính thức của RMIT Việt Nam")
    st.divider()
    st.markdown("**Cài đặt truy xuất**")
    search_mode_label = st.radio(
        "Phương pháp tìm kiếm",
        ["Kết hợp (Dense + BM25)", "Chỉ Dense"],
        help="Kết hợp dùng cả tìm kiếm ngữ nghĩa và từ khóa; Dense chỉ dùng độ tương đồng ngữ nghĩa.",
    )
    retrieval_mode = "hybrid" if search_mode_label.startswith("Kết hợp") else "dense"
    top_k = st.slider("Số lượng ngữ cảnh", 3, 10, 5, help="Số đoạn tối đa được gửi đến mô hình sinh câu trả lời.")
    use_query_expansion = st.toggle(
        "Mở rộng truy vấn",
        value=False,
        help="Viết lại câu hỏi ngắn hoặc mơ hồ trước khi truy xuất.",
    )
    if use_query_expansion:
        st.info("Đã bật: LLM sẽ viết lại câu hỏi rõ nghĩa hơn trước khi tìm kiếm.")
    pipeline_steps = ["Câu hỏi"]
    if use_query_expansion:
        pipeline_steps.append("LLM viết lại câu hỏi")
    pipeline_steps.extend(
        ["Dense + BM25", "RRF", "Xếp hạng lại"]
        if retrieval_mode == "hybrid"
        else ["Dense", "Xếp hạng lại"]
    )
    st.caption(" → ".join(pipeline_steps))
    st.divider()
    st.markdown("**Câu hỏi mẫu**")
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
    if st.button("Xóa cuộc trò chuyện", use_container_width=True, disabled=not st.session_state.messages):
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

user_input = st.chat_input("Nhập câu hỏi về Thư viện RMIT")
query = user_input or st.session_state.pending_query
if query:
    st.session_state.pending_query = None
    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm nguồn chính thức từ RMIT…"):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    history=history,
                    use_query_expansion=use_query_expansion,
                    retrieval_mode=retrieval_mode,
                )
                answer, sources = response["answer"], response.get("sources", [])
                diagnostics = response.get("diagnostics", {})
            except Exception:
                answer, sources, diagnostics = "Hệ thống hiện tạm thời không khả dụng. Vui lòng thử lại.", [], {}
        st.markdown(answer)
        render_sources(sources, diagnostics.get("result_score_type", "rrf_rank"))
        render_diagnostics(diagnostics, sources)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "diagnostics": diagnostics,
    })
