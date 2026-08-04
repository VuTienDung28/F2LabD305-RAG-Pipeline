"""Task 2 — Crawl public RMIT Vietnam Library pages."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/libraryvn",
    "https://www.rmit.edu.vn/libraryvn/borrowing-and-resources/borrowing-and-returning",
    "https://www.rmit.edu.vn/libraryvn/borrowing-and-resources/library-resources",
    "https://www.rmit.edu.vn/libraryvn/student-support/book-a-study-room",
    "https://www.rmit.edu.vn/libraryvn/student-support/study-faq",
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(url: str) -> dict:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    response = await asyncio.to_thread(requests.get, url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup.select("script, style, nav, footer, header, noscript"):
        element.decompose()
    main = soup.select_one("main") or soup.body
    markdown = markdownify(str(main), heading_style="ATX").strip()
    if len(markdown) <= 500:
        raise ValueError(f"Crawled content is too short: {url}")
    title = soup.title.get_text(" ", strip=True) if soup.title else url.rstrip("/").rsplit("/", 1)[-1]
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all() -> list[Path]:
    setup_directory()
    outputs = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        path = DATA_DIR / f"article_{index:02d}.json"
        path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(path)
        print(f"Saved: {path}")
    return outputs


if __name__ == "__main__":
    asyncio.run(crawl_all())
