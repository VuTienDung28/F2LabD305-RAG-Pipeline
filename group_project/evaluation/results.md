# RAG Evaluation Results

Framework: RAGAS 0.1.21. Scores below were produced by a completed run over 16 bilingual questions. PageIndex returned `InsufficientCredits` during evaluation, so the retrieval pipeline continued with local Chroma/BM25 results.

## Overall Scores

| Metric | dense_only | hybrid | Δ hybrid - dense |
|---|---:|---:|---:|
| Faithfulness | 0.609 | 0.516 | -0.093 |
| Answer Relevancy | 0.343 | 0.346 | +0.003 |
| Context Recall | 0.688 | 0.750 | +0.062 |
| Context Precision | 0.673 | 0.649 | -0.024 |

## A/B Analysis

Hybrid retrieval improved context recall by 0.062 and produced nearly identical answer relevancy (+0.003), but reduced faithfulness by 0.093 and context precision by 0.024. Dense-only performed better on answer-groundedness and context precision in this run, while hybrid retrieved more of the expected evidence.

## Worst Performers (Bottom 3)

| # | Config | Question | Average | Failure stage | Root cause |
|---:|---|---|---:|---|---|
| 1 | dense_only | Sinh viên đại học được mượn tối đa bao nhiêu tài liệu và trong bao lâu? | 0.000 | Retrieval / generation | Exact borrowing quota and period were not consistently grounded in the selected context. |
| 2 | dense_only | Thời gian gia hạn tài liệu thư viện là bao lâu? | 0.000 | Retrieval / generation | The renewal-period evidence was not consistently selected for this query. |
| 3 | dense_only | Phí trả tài liệu trễ là bao nhiêu? | 0.000 | Retrieval / generation | The fine amount was not consistently grounded in the selected context. |

## Recommendations

1. Add more borrowing-policy examples or increase lexical weight for exact quotas, periods and fines.
2. Tune retrieval depth and reranking to improve recall without lowering context precision.
3. Review zero-score questions after each corpus update and add missing official source passages.

## Reproduce

```bash
.venv/Scripts/python.exe -m group_project.evaluation.eval_pipeline
```
