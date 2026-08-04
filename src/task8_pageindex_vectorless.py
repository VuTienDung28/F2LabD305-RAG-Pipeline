"""Task 8: PageIndex vectorless retrieval fallback.

The PageIndex Python SDK accepts PDF files.  This module converts each
standardized Markdown document to a small PDF, uploads it once, stores document
IDs in a local manifest, and queries every ready document through PageIndex's
retrieval API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
PDF_CACHE_DIR = PROJECT_DIR / "pageindex_pdfs"
MANIFEST_PATH = PROJECT_DIR / "pageindex_doc_ids.json"

RETRIEVAL_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 2.0

load_dotenv(PROJECT_DIR / ".env")


def _api_key() -> str:
    key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PAGEINDEX_API_KEY is not configured. Add it to .env before using Task 8."
        )
    return key


@lru_cache(maxsize=1)
def _client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=_api_key())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}

    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid PageIndex manifest: {MANIFEST_PATH}") from exc

    documents = raw.get("documents", raw) if isinstance(raw, dict) else {}
    if not isinstance(documents, dict):
        raise RuntimeError(f"Invalid PageIndex manifest structure: {MANIFEST_PATH}")

    normalized: dict[str, dict[str, str]] = {}
    for source, entry in documents.items():
        if isinstance(entry, str):
            normalized[str(source)] = {"doc_id": entry, "sha256": ""}
        elif isinstance(entry, dict) and entry.get("doc_id"):
            normalized[str(source)] = {
                "doc_id": str(entry["doc_id"]),
                "sha256": str(entry.get("sha256", "")),
            }
    return normalized


def _save_manifest(documents: dict[str, dict[str, str]]) -> None:
    payload = {"version": 1, "documents": documents}
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unicode_font_path() -> Path | None:
    """Find a common Unicode TrueType font on Windows, Linux, or macOS."""
    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = [
        windows_dir / "Fonts" / "arial.ttf",
        windows_dir / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    return next((path for path in candidates if path.is_file()), None)


def _markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Render Markdown as readable text in a PageIndex-compatible PDF."""
    from fpdf import FPDF

    text = (
        markdown_path.read_text(encoding="utf-8")
        .replace("\x00", "")
        .replace("\uf0b7", "\u2022")
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    font_path = _unicode_font_path()
    if font_path:
        pdf.add_font("PageIndexBody", fname=str(font_path))
        font_name = "PageIndexBody"
        render = lambda value: value
    else:
        font_name = "Helvetica"
        render = lambda value: value.encode("latin-1", errors="replace").decode("latin-1")

    for raw_line in text.splitlines() or [markdown_path.stem]:
        line = raw_line.expandtabs(4).rstrip()
        heading_level = len(line) - len(line.lstrip("#"))
        if heading_level and line[heading_level :].startswith(" "):
            line = line[heading_level:].strip()
            font_size = max(12, 18 - (heading_level - 1) * 2)
            line_height = font_size * 0.55
        else:
            font_size = 10
            line_height = 5.5

        pdf.set_font(font_name, size=font_size)
        pdf.multi_cell(
            0,
            line_height,
            text=render(line or " "),
            new_x="LMARGIN",
            new_y="NEXT",
            wrapmode="CHAR",
        )
        if heading_level:
            pdf.ln(1)

    pdf.output(str(pdf_path))


def upload_documents(force: bool = False) -> list[dict[str, str]]:
    """Upload all standardized Markdown documents and persist their IDs.

    Unchanged files already present in the local manifest are not uploaded
    again unless ``force=True``.  The API processes documents asynchronously;
    use :func:`document_statuses` to inspect readiness before searching.
    """
    markdown_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    markdown_files = [path for path in markdown_files if path.stat().st_size > 0]
    if not markdown_files:
        raise RuntimeError(f"No Markdown documents found under {STANDARDIZED_DIR}")

    client = _client()
    manifest = _load_manifest()
    uploaded: list[dict[str, str]] = []

    for markdown_path in markdown_files:
        source = markdown_path.relative_to(STANDARDIZED_DIR).as_posix()
        sha256 = _file_hash(markdown_path)
        current = manifest.get(source, {})

        if not force and current.get("doc_id") and current.get("sha256") == sha256:
            uploaded.append(
                {"source": source, "doc_id": current["doc_id"], "status": "cached"}
            )
            continue

        pdf_path = PDF_CACHE_DIR / Path(source).with_suffix(".pdf")
        _markdown_to_pdf(markdown_path, pdf_path)
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return a doc_id for {source}: {response}")

        manifest[source] = {"doc_id": str(doc_id), "sha256": sha256}
        _save_manifest(manifest)
        uploaded.append({"source": source, "doc_id": str(doc_id), "status": "uploaded"})
        print(f"[OK] Uploaded {source} -> {doc_id}")

    return uploaded


def document_statuses() -> list[dict[str, Any]]:
    """Return PageIndex processing state for every document in the manifest."""
    manifest = _load_manifest()
    if not manifest:
        return []

    client = _client()
    statuses: list[dict[str, Any]] = []
    for source, entry in manifest.items():
        doc_id = entry["doc_id"]
        try:
            tree = client.get_tree(doc_id)
            statuses.append(
                {
                    "source": source,
                    "doc_id": doc_id,
                    "status": tree.get("status", "unknown"),
                    "retrieval_ready": bool(tree.get("retrieval_ready", False)),
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "source": source,
                    "doc_id": doc_id,
                    "status": "error",
                    "retrieval_ready": False,
                    "error": str(exc),
                }
            )
    return statuses


def _wait_for_retrieval(retrieval_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + RETRIEVAL_TIMEOUT_SECONDS
    while True:
        response = _client().get_retrieval(retrieval_id)
        status = str(response.get("status", "")).casefold()
        if status in {"completed", "complete", "success", "succeeded"}:
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"PageIndex retrieval {retrieval_id} failed: {response}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"PageIndex retrieval {retrieval_id} exceeded "
                f"{RETRIEVAL_TIMEOUT_SECONDS:.0f} seconds."
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def _relevant_items(value: Any) -> Iterator[dict[str, Any]]:
    """Flatten both documented and legacy nested relevant_contents schemas."""
    if isinstance(value, dict):
        if value.get("relevant_content"):
            yield value
        else:
            for nested in value.values():
                yield from _relevant_items(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _relevant_items(nested)


def _parse_retrieval(
    response: dict[str, Any], source: str, doc_id: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    result_rank = 0

    for node in response.get("retrieved_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        for item in _relevant_items(node.get("relevant_contents", [])):
            content = str(item.get("relevant_content", "")).strip()
            if not content:
                continue

            result_rank += 1
            section = item.get("section_title") or node.get("title")
            results.append(
                {
                    "content": content,
                    # PageIndex legacy retrieval has no similarity score.  A
                    # reciprocal rank score preserves its returned ordering.
                    "score": 1.0 / result_rank,
                    "metadata": {
                        "source": source,
                        "filename": Path(source).name,
                        "doc_id": doc_id,
                        "section": section,
                        "node_id": node.get("node_id"),
                        "page_index": item.get("page_index", node.get("page_index")),
                    },
                    "source": "pageindex",
                }
            )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Query every ready PageIndex document and return ranked passages."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    manifest = _load_manifest()
    if not manifest:
        raise RuntimeError(
            "No PageIndex document IDs found. Run "
            "`python -m src.task8_pageindex_vectorless --upload` first."
        )

    client = _client()
    all_results: list[dict[str, Any]] = []
    errors: list[str] = []
    ready_count = 0

    for source, entry in manifest.items():
        doc_id = entry["doc_id"]
        try:
            if not client.is_retrieval_ready(doc_id):
                continue
            ready_count += 1

            submission = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submission.get("retrieval_id") or submission.get("id")
            if not retrieval_id:
                raise RuntimeError(f"Missing retrieval_id in response: {submission}")

            response = _wait_for_retrieval(str(retrieval_id))
            all_results.extend(_parse_retrieval(response, source, doc_id))
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if ready_count == 0:
        raise RuntimeError(
            "No uploaded PageIndex document is retrieval-ready yet. "
            "Run the status command and try again after processing completes."
        )
    if not all_results and errors:
        raise RuntimeError("PageIndex search failed: " + " | ".join(errors))

    # The same passage can appear in multiple nodes.  Keep its best rank score.
    deduplicated: dict[str, dict[str, Any]] = {}
    for result in all_results:
        key = " ".join(result["content"].split()).casefold()
        if key not in deduplicated or result["score"] > deduplicated[key]["score"]:
            deduplicated[key] = result

    return sorted(
        deduplicated.values(),
        key=lambda result: result["score"],
        reverse=True,
    )[:top_k]


def _print_statuses() -> None:
    statuses = document_statuses()
    if not statuses:
        print("No uploaded documents in the local manifest.")
        return
    for item in statuses:
        ready = "ready" if item["retrieval_ready"] else item["status"]
        print(f"[{ready}] {item['source']} -> {item['doc_id']}")


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Task 8 PageIndex vectorless retrieval")
    parser.add_argument("--upload", action="store_true", help="upload standardized documents")
    parser.add_argument("--force", action="store_true", help="re-upload unchanged documents")
    parser.add_argument("--status", action="store_true", help="show document processing status")
    parser.add_argument("--query", help="run a PageIndex retrieval query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.upload:
        uploaded = upload_documents(force=args.force)
        print(f"Processed {len(uploaded)} local documents.")
    if args.status:
        _print_statuses()
    if args.query:
        for rank, result in enumerate(pageindex_search(args.query, args.top_k), start=1):
            metadata = result["metadata"]
            print(f"{rank}. [{result['score']:.3f}] {metadata['source']}")
            print(f"   {result['content'][:160]}...")
    if not (args.upload or args.status or args.query):
        parser.print_help()
