"""Task 1 — Download public RMIT Vietnam Library documents."""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
LEGAL_DOCUMENTS = [
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/study-at-rmit/programs/english-pdf/postgraduate-programs/postgraduate-guide-14032025.pdf",
        "filename": "rmit-postgraduate-library-guide-2025.pdf",
    },
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/library/en/tu-nguyen-en-information-literacy-and-the-use-of-library-resources-changes-in-the-generative-ai-age.pdf",
        "filename": "rmit-information-literacy-library-resources.pdf",
    },
    {
        "url": "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/vn-parents-guide/en-fc-parents-guide-050924.pdf",
        "filename": "rmit-vietnam-library-parents-guide.pdf",
    },
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str) -> Path:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF from {url}, received {content_type or 'unknown content'}")
    if len(response.content) <= 1024:
        raise ValueError(f"Downloaded file is too small: {url}")
    destination = DATA_DIR / filename
    destination.write_bytes(response.content)
    return destination


def collect_all() -> list[Path]:
    setup_directory()
    downloaded = []
    for item in LEGAL_DOCUMENTS:
        destination = download_file(item["url"], item["filename"])
        downloaded.append(destination)
        print(f"Downloaded: {destination}")
    return downloaded


if __name__ == "__main__":
    collect_all()
