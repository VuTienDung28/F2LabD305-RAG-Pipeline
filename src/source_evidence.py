"""Safely load and highlight standardized source documents."""

from html import escape
from pathlib import Path

from .task4_chunking_indexing import STANDARDIZED_DIR


def load_document(document_path: str, root: Path = STANDARDIZED_DIR) -> str | None:
    path = Path(document_path)
    if path.is_absolute() or path.suffix.casefold() != ".md":
        return None
    root = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8")


def highlight_document(document: str, chunks: list[dict]) -> tuple[str, int]:
    ranges = []
    for chunk in chunks:
        content = chunk.get("content", "")
        start = chunk.get("metadata", {}).get("chunk_start")
        if not isinstance(start, int) or isinstance(start, bool) or not content:
            continue
        end = start + len(content)
        if start < 0 or document[start:end] != content:
            continue
        ranges.append((start, end))

    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts = []
    cursor = 0
    for start, end in merged:
        parts.append(escape(document[cursor:start]))
        parts.append(f'<mark class="rag-chunk">{escape(document[start:end])}</mark>')
        cursor = end
    parts.append(escape(document[cursor:]))
    return f'<div class="source-document">{"".join(parts)}</div>', len(ranges)
