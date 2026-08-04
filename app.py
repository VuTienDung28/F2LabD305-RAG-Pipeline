"""Streamlit chatbot for the RMIT Vietnam Library RAG pipeline."""

from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.task10_generation import generate_with_citation

st.set_page_config(page_title="RMIT Vietnam Library RAG", page_icon="📚", layout="wide")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo / Sources ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            label = f"[{title}]({url})" if url else title
            st.markdown(
                f"**{index}. {label}** · `{metadata.get('type', 'unknown')}` · score `{float(source.get('score', 0)):.4f}`"
            )
            st.markdown(f"> {source.get('content', '')[:300].strip()}…")


with st.sidebar:
    st.title("RMIT Vietnam Library")
    st.caption("Bilingual grounded assistant / Trợ lý song ngữ có trích dẫn")
    top_k = st.slider("Retrieval top_k", 3, 10, 5)
    use_query_expansion = st.toggle("Query expansion (bonus)", value=False)
    st.caption("Semantic + BM25 → RRF → Jina → PageIndex fallback → OpenRouter")
    st.subheader("Gợi ý / Suggestions")
    suggestions = [
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên được mượn bao nhiêu sách?",
        "How can I access databases off campus?",
        "What research support does the library provide?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_query = suggestion

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.title("RMIT Vietnam Library RAG Chatbot")
st.caption("Ask in Vietnamese or English. Answers are grounded in official public RMIT sources.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

user_input = st.chat_input("Hỏi về thư viện RMIT / Ask about the RMIT Library")
query = user_input or st.session_state.pending_query
if query:
    st.session_state.pending_query = None
    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving official sources…"):
            try:
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    history=history,
                    use_query_expansion=use_query_expansion,
                )
                answer, sources = response["answer"], response.get("sources", [])
            except Exception as error:
                answer, sources = f"Pipeline error: `{error}`", []
        st.markdown(answer)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
