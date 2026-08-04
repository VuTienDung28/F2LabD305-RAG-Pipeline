"""Task 2 - Crawl university-service news from the public RMIT website.

Each article is saved as one UTF-8 JSON file in ``data/landing/news`` with
its source URL, title, crawl time, and cleaned text content.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
MIN_ARTICLES = 5
REQUEST_TIMEOUT_SECONDS = 30


def setup_directory() -> None:
    """Create ``data/landing/news`` when it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Public articles from RMIT Vietnam Library. A sixth URL gives the crawl one
# spare source while the assignment requires at least five successful files.
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/10-years-book-swap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/rmit-vietnam-library-launches-adobe-express-champions",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/library-welcomes-visitors-from-can-tho-university",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/beyond-the-pages-recap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/words-that-inspire-recap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/experience-day-2025-at-the-library",
]


class ArticleHTMLParser(HTMLParser):
    """Extract readable text from RMIT's nine-column article body."""

    BLOCK_TAGS = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "h5", "h6", "li", "p", "section", "table", "td", "th", "tr",
    }
    IGNORED_TAGS = {"button", "form", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.content_depth = 0
        self.ignored_depth = 0
        self.title_depth = 0
        self.title_parts: list[str] = []
        self.article_parts: list[str] = []
        self.meta_title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") == "og:title":
            self.meta_title = attributes.get("content", "") or ""
        if tag == "title":
            self.title_depth += 1
        classes = (attributes.get("class") or "").split()
        starts_content = tag == "div" and "aem-GridColumn--default--9" in classes
        if tag == "div":
            if self.content_depth:
                self.content_depth += 1
            elif starts_content:
                self.content_depth = 1
        if self.content_depth and tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
        if self.content_depth and not self.ignored_depth and tag in self.BLOCK_TAGS:
            self.article_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.content_depth and not self.ignored_depth and tag in self.BLOCK_TAGS:
            self.article_parts.append("\n")
        if self.content_depth and tag in self.IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1
        if tag == "div" and self.content_depth:
            self.content_depth -= 1
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if self.content_depth and not self.ignored_depth:
            text = re.sub(r"\s+", " ", unescape(data)).strip()
            if text:
                self.article_parts.append(text + " ")

    @property
    def title(self) -> str:
        html_title = " ".join("".join(self.title_parts).split())
        return " ".join((self.meta_title or html_title or "Unknown").split())

    @property
    def content(self) -> str:
        lines = []
        for line in "".join(self.article_parts).splitlines():
            clean_line = " ".join(line.split())
            if clean_line and (not lines or clean_line != lines[-1]):
                lines.append(clean_line)
        return "\n\n".join(lines)


def _fetch_article(url: str) -> dict[str, str]:
    """Fetch and parse one article synchronously (run in a worker thread)."""
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; UniversityServicesRAG/1.0; "
                "+educational-project)"
            )
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    parser = ArticleHTMLParser()
    parser.feed(response.text)
    content = parser.content
    if len(content) < 500:
        raise ValueError(f"Extracted content is too short ({len(content)} characters)")

    return {
        "url": response.url,
        "title": parser.title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict[str, str]:
    """Crawl one article without blocking the asyncio event loop."""
    return await asyncio.to_thread(_fetch_article, url)


async def crawl_all() -> list[Path]:
    """Crawl all configured URLs and require at least five successes."""
    setup_directory()
    saved_files: list[Path] = []

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except (requests.RequestException, ValueError) as exc:
            print(f"  x Failed: {exc}")
            continue

        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_files.append(filepath)
        print(f"  + Saved: {filepath}")

    if len(saved_files) < MIN_ARTICLES:
        raise RuntimeError(
            f"Only {len(saved_files)} articles were saved; at least {MIN_ARTICLES} are required."
        )

    print(f"Completed: saved {len(saved_files)} articles in {DATA_DIR}")
    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
