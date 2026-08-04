# RMIT Vietnam Library RAG Chatbot

Chatbot song ngữ Việt–Anh trả lời câu hỏi về dịch vụ và tài nguyên Thư viện RMIT Việt Nam, sử dụng dữ liệu công khai chính thức và hiển thị nguồn cho câu trả lời.

## Kiến trúc

```text
RMIT PDF + web pages
        ↓
landing JSON/PDF → standardized Markdown → chunks (800/100)
        ↓
OpenAI Embeddings + ChromaDB ─┐
BM25 ───────────────┼→ RRF → Jina rerank → PageIndex fallback
                    ↓
          OpenRouter generation
                    ↓
        Streamlit chat + sources
```

- Dense search dùng OpenAI `text-embedding-3-small` với output 1024 chiều và cosine similarity trong ChromaDB.
- Lexical search dùng BM25, không phải phép đếm từ khóa đơn giản. BM25 cải thiện TF-IDF bằng cách bão hòa term frequency và chuẩn hóa độ dài tài liệu, nên một từ lặp nhiều lần không làm điểm tăng tuyến tính và chunk dài không được ưu tiên vô lý.
- Hybrid retrieval hợp nhất dense và BM25 bằng Reciprocal Rank Fusion (RRF), sau đó rerank bằng Jina nếu có key.
- Khi cosine score tốt nhất thấp hơn `0.48`, pipeline thử PageIndex trên các PDF đã tải lên.
- Generation chỉ dùng context truy xuất, trả lời theo ngôn ngữ câu hỏi và gắn citation dạng `[Source N: title]`.

## Cài đặt

Dùng Python 3.11 hoặc 3.12 vì `numpy==1.26.4` không hỗ trợ Python 3.13.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Tạo `.env` từ `.env.example` và cung cấp các key cần dùng:

```text
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
JINA_API_KEY=...
PAGEINDEX_API_KEY=...
```

Không commit `.env` hoặc API key.

## Chuẩn bị dữ liệu và index

```bash
.venv/Scripts/python.exe -m src.task1_collect_legal_docs
.venv/Scripts/python.exe -m src.task2_crawl_news
.venv/Scripts/python.exe -m src.task3_convert_markdown
.venv/Scripts/python.exe -m src.task4_chunking_indexing
```

Để dùng PageIndex fallback, tải PDF lên một lần:

```bash
.venv/Scripts/python.exe -m src.task8_pageindex_vectorless
```

Document IDs được lưu trong `pageindex_doc_ids.json`; file này đã được gitignore.

## Chạy chatbot

```bash
.venv/Scripts/streamlit.exe run app.py
```

UI hỗ trợ lịch sử hội thoại, chọn `top_k`, bật query expansion, xem source URL, retrieval score và excerpt.

## Evaluation

Golden dataset gồm 16 câu song ngữ có expected answer, expected context và source URL. Script chạy bốn metric RAGAS cho dense-only và hybrid:

- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision

```bash
.venv/Scripts/python.exe -m group_project.evaluation.eval_pipeline
```

`evaluation/results.md` chỉ được ghi sau khi cả hai cấu hình hoàn tất; script không sinh score giả nếu API hoặc quota lỗi.

## Kiểm thử

```bash
.venv/Scripts/python.exe -m pytest tests/test_individual.py tests/test_pipeline_contracts.py -v
```

## Phân công vai trò

| Vai trò | Phạm vi đã tích hợp |
|---|---|
| Architecture / Integration | Task 4, 9 và luồng end-to-end |
| Data Collection / Conversion | Task 1–3 và dữ liệu RMIT chính thức |
| Dense Retrieval | Task 4–5, OpenAI Embeddings và ChromaDB |
| Sparse Retrieval / Reranking | Task 6–8, BM25, RRF, Jina và PageIndex |
| Generation / Frontend | Task 10 và Streamlit chatbot |
| Evaluation / QA | Golden dataset, RAGAS A/B và test suite |

Tên thành viên và MSSV cần được nhóm điền theo danh sách thực tế; repository không suy đoán thông tin cá nhân.
