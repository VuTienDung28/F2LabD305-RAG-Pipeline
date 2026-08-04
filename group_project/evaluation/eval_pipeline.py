"""RAGAS evaluation for dense-only and hybrid retrieval."""

import json
import math
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def _collect_rows(generate: Callable[..., dict], golden_dataset: list[dict], mode: str) -> list[dict]:
    rows = []
    for item in golden_dataset:
        result = generate(item["question"], retrieval_mode=mode)
        rows.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [source["content"] for source in result.get("sources", [])],
            "ground_truth": item["expected_answer"],
        })
    return rows


def evaluate_with_ragas(generate: Callable[..., dict], golden_dataset: list[dict], mode: str) -> dict:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for RAGAS evaluation")
    rows = _collect_rows(generate, golden_dataset, mode)
    dataset = Dataset.from_list(rows)
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=1024,
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )
    evaluated_rows = result.to_pandas().to_dict(orient="records")
    summary = {
        metric: sum(float(row[metric]) for row in evaluated_rows) / len(evaluated_rows)
        for metric in METRICS
    }
    return {"summary": summary, "rows": evaluated_rows}


def compare_configs(generate: Callable[..., dict], golden_dataset: list[dict] | None = None) -> dict:
    dataset = golden_dataset or load_golden_dataset()
    return {
        "dense_only": evaluate_with_ragas(generate, dataset, "dense"),
        "hybrid": evaluate_with_ragas(generate, dataset, "hybrid"),
    }


def _score(value: object) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(number) else f"{number:.3f}"


def export_results(comparison: dict) -> Path:
    config_names = list(comparison)
    lines = [
        "# RAG Evaluation Results",
        "",
        "Framework: RAGAS 0.1.21. Scores below are written only from a completed evaluation run.",
        "",
        "## Overall Scores",
        "",
        "| Metric | " + " | ".join(config_names) + " |",
        "|---|" + "---|" * len(config_names),
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    for metric in METRICS:
        values = [_score(comparison[name].get("summary", {}).get(metric)) for name in config_names]
        lines.append(f"| {labels[metric]} | " + " | ".join(values) + " |")

    if len(config_names) >= 2:
        first, second = config_names[:2]
        deltas = []
        for metric in METRICS:
            first_value = comparison[first].get("summary", {}).get(metric)
            second_value = comparison[second].get("summary", {}).get(metric)
            if first_value is not None and second_value is not None:
                deltas.append((float(second_value) - float(first_value), labels[metric]))
        lines.extend(["", "## A/B Analysis", ""])
        if deltas:
            best_delta, best_metric = max(deltas)
            worst_delta, worst_metric = min(deltas)
            lines.append(
                f"Compared with `{first}`, `{second}` changes {best_metric} by {best_delta:+.3f} "
                f"and {worst_metric} by {worst_delta:+.3f}."
            )
        else:
            lines.append("Both configurations are included; complete scores are required for a numeric comparison.")

    lines.extend(["", "## Worst Performers", "", "| Config | Question | Average |", "|---|---|---|"])
    ranked = []
    for config_name, result in comparison.items():
        for row in result.get("rows", []):
            values = [float(row[metric]) for metric in METRICS if row.get(metric) is not None]
            average = sum(values) / len(values) if values else math.nan
            ranked.append((average, config_name, str(row.get("question", "")).replace("|", "\\|")))
    for average, config_name, question in sorted(ranked, key=lambda item: item[0])[:3]:
        lines.append(f"| {config_name} | {question} | {_score(average)} |")
    if not ranked:
        lines.append("| — | No per-question rows were provided. | — |")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Add more borrowing-policy examples or increase lexical weight for exact quotas, periods and fines.",
        "2. Tune retrieval depth and reranking to improve recall without lowering context precision.",
        "3. Review zero-score questions after each corpus update and add missing official source passages.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m group_project.evaluation.eval_pipeline",
        "```",
        "",
    ])
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS_PATH


if __name__ == "__main__":
    from src.task10_generation import generate_with_citation

    export_results(compare_configs(generate_with_citation))
    print(f"Saved real RAGAS scores to {RESULTS_PATH}")
