from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .engine import Document


@dataclass(frozen=True)
class Site:
    id: int
    base_url: str
    source_url: str
    source_type: str
    limit_pages: int | None
    status: str
    page_count: int
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Page:
    id: int
    site_id: int
    url: str
    title: str
    content_length: int
    crawled_at: str


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                base_url TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                limit_pages INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                page_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_length INTEGER NOT NULL DEFAULT 0,
                crawled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE,
                UNIQUE(site_id, url)
            );

            CREATE INDEX IF NOT EXISTS idx_pages_site_id ON pages(site_id);
            CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
            """
        )


def _site_from_row(row: sqlite3.Row) -> Site:
    return Site(
        id=int(row["id"]),
        base_url=str(row["base_url"]),
        source_url=str(row["source_url"]),
        source_type=str(row["source_type"]),
        limit_pages=None if row["limit_pages"] is None else int(row["limit_pages"]),
        status=str(row["status"]),
        page_count=int(row["page_count"]),
        error=None if row["error"] is None else str(row["error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        id=int(row["id"]),
        site_id=int(row["site_id"]),
        url=str(row["url"]),
        title=str(row["title"]),
        content_length=int(row["content_length"]),
        crawled_at=str(row["crawled_at"]),
    )


def create_site(
    conn: sqlite3.Connection,
    *,
    base_url: str,
    source_url: str,
    source_type: str,
    limit_pages: int | None,
    status: str = "running",
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO sites (base_url, source_url, source_type, limit_pages, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (base_url, source_url, source_type, limit_pages, status),
    )
    return int(cursor.lastrowid)


def upsert_page(conn: sqlite3.Connection, *, site_id: int, document: Document) -> None:
    content = document.content.strip()
    conn.execute(
        """
        INSERT INTO pages (site_id, url, title, content, content_length)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(site_id, url) DO UPDATE SET
            title = excluded.title,
            content = excluded.content,
            content_length = excluded.content_length,
            crawled_at = CURRENT_TIMESTAMP
        """,
        (site_id, document.url, document.title or document.url, content, len(content)),
    )


def refresh_site_stats(conn: sqlite3.Connection, site_id: int) -> None:
    conn.execute(
        """
        UPDATE sites
        SET page_count = (SELECT COUNT(*) FROM pages WHERE site_id = ?),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (site_id, site_id),
    )


def update_site_status(
    conn: sqlite3.Connection,
    site_id: int,
    *,
    status: str,
    error: str | None = None,
) -> None:
    refresh_site_stats(conn, site_id)
    conn.execute(
        """
        UPDATE sites
        SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status, error, site_id),
    )


def list_sites(conn: sqlite3.Connection) -> list[Site]:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                base_url,
                source_url,
                source_type,
                limit_pages,
                status,
                page_count,
                error,
                created_at,
                updated_at,
                ROW_NUMBER() OVER (PARTITION BY base_url ORDER BY id DESC) AS row_number,
                SUM(page_count) OVER (PARTITION BY base_url) AS total_page_count
            FROM sites
        )
        SELECT
            id,
            base_url,
            source_url,
            source_type,
            limit_pages,
            status,
            total_page_count AS page_count,
            error,
            created_at,
            updated_at
        FROM ranked
        WHERE row_number = 1
        ORDER BY id DESC
        """
    ).fetchall()
    return [_site_from_row(row) for row in rows]


def get_site(conn: sqlite3.Connection, site_id: int) -> Site | None:
    row = conn.execute(
        """
        WITH selected AS (
            SELECT base_url
            FROM sites
            WHERE id = ?
        ),
        ranked AS (
            SELECT
                s.id,
                s.base_url,
                s.source_url,
                s.source_type,
                s.limit_pages,
                s.status,
                s.error,
                s.created_at,
                s.updated_at,
                ROW_NUMBER() OVER (PARTITION BY s.base_url ORDER BY s.id DESC) AS row_number,
                SUM(s.page_count) OVER (PARTITION BY s.base_url) AS total_page_count
            FROM sites s
            JOIN selected ON selected.base_url = s.base_url
        )
        SELECT
            id,
            base_url,
            source_url,
            source_type,
            limit_pages,
            status,
            total_page_count AS page_count,
            error,
            created_at,
            updated_at
        FROM ranked
        WHERE row_number = 1
        """,
        (site_id,),
    ).fetchone()
    return None if row is None else _site_from_row(row)


def list_pages(conn: sqlite3.Connection, site_id: int) -> list[Page]:
    rows = conn.execute(
        """
        WITH selected AS (
            SELECT base_url
            FROM sites
            WHERE id = ?
        )
        SELECT id, site_id, url, title, content_length, crawled_at
        FROM pages
        WHERE site_id IN (
            SELECT id
            FROM sites
            WHERE base_url = (SELECT base_url FROM selected)
        )
        ORDER BY id DESC
        """,
        (site_id,),
    ).fetchall()
    return [_page_from_row(row) for row in rows]


def load_documents(conn: sqlite3.Connection) -> list[Document]:
    rows = conn.execute(
        """
        SELECT id, title, url, content
        FROM pages
        WHERE TRIM(content) <> ''
        ORDER BY id ASC
        """
    ).fetchall()
    return [
        Document(
            id=str(row["id"]),
            title=str(row["title"]),
            url=str(row["url"]),
            content=str(row["content"]),
        )
        for row in rows
    ]


def total_pages(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM pages").fetchone()
    return int(row["count"])


def import_documents_as_site(
    conn: sqlite3.Connection,
    documents: Sequence[Document],
    *,
    base_url: str,
    source_url: str,
    source_type: str = "sample",
) -> int:
    site_id = create_site(
        conn,
        base_url=base_url,
        source_url=source_url,
        source_type=source_type,
        limit_pages=len(documents),
        status="running",
    )
    for document in documents:
        upsert_page(conn, site_id=site_id, document=document)
    update_site_status(conn, site_id, status="done")
    return site_id
