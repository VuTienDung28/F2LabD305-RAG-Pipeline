"""Task 3 — Convert landing documents to source-attributed Markdown."""

import json
from pathlib import Path

from markitdown import MarkItDown

from .task1_collect_legal_docs import LEGAL_DOCUMENTS

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_SOURCES = {item["filename"]: item["url"] for item in LEGAL_DOCUMENTS}


def _write_markdown(path: Path, title: str, source_url: str, doc_type: str, body: str, crawled: str = "") -> Path:
    body = body.strip()
    if len(body) <= 200:
        raise ValueError(f"Converted content is too short: {path.name}")
    header = [f"# {title}", "", f"**Source:** {source_url}"]
    if crawled:
        header.append(f"**Crawled:** {crawled}")
    header.extend([f"**Type:** {doc_type}", "", "---", ""])
    path.write_text("\n".join(header) + body + "\n", encoding="utf-8")
    return path


def convert_legal_docs() -> list[Path]:
    source_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    outputs = []
    for filepath in sorted(source_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        result = converter.convert(str(filepath))
        output = _write_markdown(
            output_dir / f"{filepath.stem}.md",
            filepath.stem.replace("-", " ").title(),
            LEGAL_SOURCES.get(filepath.name, ""),
            "legal",
            result.text_content,
        )
        outputs.append(output)
        print(f"Saved: {output}")
    return outputs


def convert_news_articles() -> list[Path]:
    source_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filepath in sorted(source_dir.glob("*.json")):
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output = _write_markdown(
            output_dir / f"{filepath.stem}.md",
            data.get("title", "Unknown"),
            data.get("url", ""),
            "news",
            data.get("content_markdown", ""),
            data.get("date_crawled", ""),
        )
        outputs.append(output)
        print(f"Saved: {output}")
    return outputs


def convert_all() -> list[Path]:
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    outputs = convert_all()
    print(f"Converted {len(outputs)} files into {OUTPUT_DIR}")
