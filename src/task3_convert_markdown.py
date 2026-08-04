"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in ``data/landing/legal`` to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {legal_dir}")

    md = MarkItDown()
    output_files: list[Path] = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            content = result.text_content.strip()
            if not content:
                raise ValueError(f"MarkItDown returned empty content for {filepath}")

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content + "\n", encoding="utf-8")
            output_files.append(output_path)
            print(f"  [OK] Saved: {output_path}")

    return output_files


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in ``data/landing/news`` to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {news_dir}")

    output_files: list[Path] = []

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            article_content = data.get("content_markdown") or data.get("content")
            if not isinstance(article_content, str) or not article_content.strip():
                raise ValueError(f"Missing article content in {filepath}")

            header = (
                f"# {data.get('title', 'Unknown')}\n\n"
                f"**Source:** {data.get('url', 'N/A')}  \n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
                "---\n\n"
            )
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(
                header + article_content.strip() + "\n",
                encoding="utf-8",
            )
            output_files.append(output_path)
            print(f"  [OK] Saved: {output_path}")

    return output_files


def convert_all() -> tuple[list[Path], list[Path]]:
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    print(
        f"\n[OK] Done! Converted {len(legal_files)} legal documents and "
        f"{len(news_files)} news articles to: {OUTPUT_DIR}"
    )
    return legal_files, news_files


if __name__ == "__main__":
    convert_all()
