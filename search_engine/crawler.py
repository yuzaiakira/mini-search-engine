from __future__ import annotations

import gzip
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from html.parser import HTMLParser
from typing import Sequence
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from .engine import Document


DEFAULT_USER_AGENT = "MiniGoogleBot/0.2 (+educational search engine)"


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)

        if tag == "title":
            self._in_title = True
        elif tag in {"script", "style", "noscript", "svg", "canvas"}:
            self._skip_depth += 1
        elif tag == "a" and "href" in attrs_dict:
            self.links.append(attrs_dict["href"])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._in_title = False
        elif tag in {"script", "style", "noscript", "svg", "canvas"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not data or self._skip_depth:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.text_parts)).strip()


def add_default_scheme(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return "https://" + url
    return url


def clean_url(url: str) -> str:
    url = add_default_scheme(url)
    url, _fragment = urldefrag(url)
    return url.strip()


def base_url_for(url: str) -> str:
    parsed = urlparse(clean_url(url))
    if not parsed.scheme or not parsed.netloc:
        return clean_url(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(clean_url(url_a)).netloc.lower() == urlparse(clean_url(url_b)).netloc.lower()


def _request(url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> Request:
    return Request(
        clean_url(url),
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        },
    )


def fetch_bytes(
    url: str,
    *,
    timeout: int = 10,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[bytes, str, str]:
    with urlopen(_request(url, user_agent=user_agent), timeout=timeout) as response:
        raw = response.read()
        content_type = response.headers.get("Content-Type", "")
        encoding = response.headers.get_content_charset() or "utf-8"
        return raw, content_type, encoding


def fetch_html(url: str, *, timeout: int = 10, user_agent: str = DEFAULT_USER_AGENT) -> str:
    raw, content_type, encoding = fetch_bytes(url, timeout=timeout, user_agent=user_agent)
    lowered_content_type = content_type.lower()
    if lowered_content_type and not any(
        item in lowered_content_type for item in ("text/html", "application/xhtml+xml")
    ):
        return ""
    return raw.decode(encoding, errors="replace")


def fetch_page_document_and_links(
    url: str,
    *,
    timeout: int = 10,
    min_text_chars: int = 40,
) -> tuple[Document | None, list[str]]:
    url = clean_url(url)
    html = fetch_html(url, timeout=timeout)
    if not html:
        return None, []

    parser = SimpleHTMLTextExtractor()
    parser.feed(html)

    title = parser.title or url
    text = parser.text
    document = None
    if len(text) >= min_text_chars:
        document = Document(id=url, title=title, url=url, content=text)

    absolute_links = [clean_url(urljoin(url, href)) for href in parser.links]
    return document, absolute_links


def fetch_page_document(url: str, *, timeout: int = 10) -> Document | None:
    document, _links = fetch_page_document_and_links(url, timeout=timeout)
    return document


def _limit_reached(count: int, max_pages: int | None, safety_max_pages: int) -> bool:
    if count >= safety_max_pages:
        return True
    if max_pages is None:
        return False
    return count >= max_pages


def crawl(
    seed_urls: Sequence[str],
    *,
    max_pages: int | None = 30,
    delay_seconds: float = 0.2,
    stay_on_same_domain: bool = True,
    safety_max_pages: int = 10_000,
) -> list[Document]:
    """Very small educational crawler.

    If max_pages is None, the crawler keeps going until the same-domain queue is
    empty or safety_max_pages is reached. For real websites, also respect
    robots.txt and the target site's terms.
    """

    queue = deque(clean_url(url) for url in seed_urls if clean_url(url))
    seen: set[str] = set()
    documents: list[Document] = []
    seed = queue[0] if queue else ""

    while queue and not _limit_reached(len(documents), max_pages, safety_max_pages):
        url = clean_url(queue.popleft())
        if not url or url in seen:
            continue
        seen.add(url)

        try:
            document, links = fetch_page_document_and_links(url)
        except Exception:
            continue

        if document is not None:
            document.id = str(len(documents) + 1)
            documents.append(document)

        for href in links:
            next_url = clean_url(href)
            parsed = urlparse(next_url)

            if parsed.scheme not in {"http", "https"}:
                continue
            if stay_on_same_domain and seed and not same_domain(seed, next_url):
                continue
            if next_url not in seen:
                queue.append(next_url)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return documents


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _decode_sitemap_bytes(raw: bytes, url: str, content_type: str, encoding: str) -> str:
    if url.lower().endswith(".gz") or "gzip" in content_type.lower():
        raw = gzip.decompress(raw)
    return raw.decode(encoding or "utf-8", errors="replace")


def _direct_child_text(element: ET.Element, child_name: str) -> str | None:
    for child in list(element):
        if _local_name(child.tag) == child_name and child.text:
            return child.text.strip()
    return None


def parse_sitemap(xml_text: str) -> tuple[str, list[str]]:
    root = ET.fromstring(xml_text)
    root_name = _local_name(root.tag)
    locations: list[str] = []

    if root_name == "sitemapindex":
        for sitemap in list(root):
            if _local_name(sitemap.tag) == "sitemap":
                loc = _direct_child_text(sitemap, "loc")
                if loc:
                    locations.append(clean_url(loc))
        return "sitemapindex", locations

    if root_name == "urlset":
        for url_element in list(root):
            if _local_name(url_element.tag) == "url":
                loc = _direct_child_text(url_element, "loc")
                if loc:
                    locations.append(clean_url(loc))
        return "urlset", locations

    # Fallback for simple XML files that still contain <loc> tags.
    for element in root.iter():
        if _local_name(element.tag) == "loc" and element.text:
            locations.append(clean_url(element.text.strip()))
    return root_name, locations


def urls_from_sitemap(
    sitemap_url: str,
    *,
    max_urls: int | None = None,
    timeout: int = 10,
    safety_max_sitemaps: int = 500,
) -> list[str]:
    sitemap_url = clean_url(sitemap_url)
    sitemap_queue = deque([sitemap_url])
    seen_sitemaps: set[str] = set()
    seen_pages: set[str] = set()
    page_urls: list[str] = []

    while sitemap_queue and len(seen_sitemaps) < safety_max_sitemaps:
        current_sitemap = clean_url(sitemap_queue.popleft())
        if current_sitemap in seen_sitemaps:
            continue
        seen_sitemaps.add(current_sitemap)

        raw, content_type, encoding = fetch_bytes(current_sitemap, timeout=timeout)
        xml_text = _decode_sitemap_bytes(raw, current_sitemap, content_type, encoding)
        sitemap_type, locations = parse_sitemap(xml_text)

        if sitemap_type == "sitemapindex":
            for location in locations:
                if location not in seen_sitemaps:
                    sitemap_queue.append(location)
            continue

        for location in locations:
            parsed = urlparse(location)
            if parsed.scheme not in {"http", "https"} or location in seen_pages:
                continue
            seen_pages.add(location)
            page_urls.append(location)
            if max_urls is not None and len(page_urls) >= max_urls:
                return page_urls

    return page_urls


def crawl_sitemap(
    sitemap_url: str,
    *,
    max_pages: int | None = None,
    delay_seconds: float = 0.1,
) -> list[Document]:
    page_urls = urls_from_sitemap(sitemap_url, max_urls=max_pages)
    documents: list[Document] = []

    for page_url in page_urls:
        try:
            document = fetch_page_document(page_url)
        except Exception:
            continue

        if document is not None:
            document.id = str(len(documents) + 1)
            documents.append(document)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return documents
