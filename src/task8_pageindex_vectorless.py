"""Task 8 — PageIndex cloud fallback over uploaded RMIT PDFs."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is required")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents(timeout: int = 900) -> dict[str, str]:
    client = _client()
    doc_ids = {}
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        response = client.submit_document(str(path))
        doc_id = response.get("doc_id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {path.name}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.get_document(doc_id).get("status")
            if status == "completed":
                break
            if status in {"failed", "error"}:
                raise RuntimeError(f"PageIndex processing failed for {path.name}")
            time.sleep(5)
        else:
            raise TimeoutError(f"PageIndex processing timed out for {path.name}")
        doc_ids[path.name] = doc_id
        print(f"Uploaded: {path.name} -> {doc_id}")
    if not doc_ids:
        raise RuntimeError(f"No PDFs found in {LEGAL_DIR}")
    DOC_IDS_PATH.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    return doc_ids


def _load_doc_ids() -> dict[str, str]:
    if not DOC_IDS_PATH.exists():
        raise RuntimeError("Run upload_documents() before PageIndex search")
    return json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    doc_ids = _load_doc_ids()
    retrieval_prompt = (
        "Retrieve raw passages relevant to the query from the documents. "
        "Return JSON only as a list of objects with keys source, page, content. "
        f"Query: {query}"
    )
    response = _client().chat_completions(
        messages=[{"role": "user", "content": retrieval_prompt}],
        doc_id=list(doc_ids.values()),
    )
    content = response["choices"][0]["message"]["content"]
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end < start:
        raise RuntimeError("PageIndex did not return JSON passages")
    passages = json.loads(content[start:end + 1])
    if not isinstance(passages, list):
        raise RuntimeError("PageIndex returned an invalid passage list")
    return [
        {
            "content": item.get("content", ""),
            "score": 1.0 / rank,
            "metadata": {
                "source": item.get("source", "PageIndex"),
                "page": item.get("page", ""),
                "type": "legal",
            },
            "source": "pageindex",
        }
        for rank, item in enumerate(passages[:top_k], 1)
        if item.get("content")
    ]


if __name__ == "__main__":
    upload_documents()
    print(pageindex_search("What library resources can RMIT students access?", top_k=3))
