# Mini Search Engine

A lightweight educational search engine built with **Flask**, **SQLite**, and **scikit-learn (TF-IDF + cosine similarity)**.

This project demonstrates a complete search workflow:
- collect content (single page, sitemap, or crawler),
- store pages in SQLite,
- build and refresh a TF-IDF index,
- serve search results through a web UI and a JSON API.

> This is a learning project, not a production-grade web search system.

## Features

- Flask-based web interface with pages for:
  - search (`/` or `/search`)
  - adding new content sources (`/add-site`)
  - viewing indexed sources and pages (`/sites`, `/sites/<site_id>`)
- SQLite persistence for sources and indexed pages
- Automatic index rebuild after new pages are ingested
- Multiple ingestion modes:
  1. **Single page URL**
  2. **Sitemap URL**
  3. **Site crawl** (fallback when page/sitemap is not provided)
- Optional page limit for controlled crawling/indexing
- Search ranking using TF-IDF + cosine similarity
- Result snippets and query term highlighting
- Simple JSON API: `/api/search?q=...`

## Tech Stack

- Python 3
- Flask
- SQLite
- scikit-learn
- NLTK
- BeautifulSoup / requests-style crawling utilities (project crawler module)

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the app

```bash
python app.py --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

On first run, if the database is empty, sample data from `data/documents.jsonl` is imported automatically.

## Web Routes

```text
/search or /         Search page
/add-site            Add a source (site, sitemap, or single page)
/sites               List indexed sources
/sites/<site_id>     List pages for a source
/api/search?q=...    JSON search endpoint
```

## Add Source Workflow

The Add Source form accepts:

| Field | Description |
|---|---|
| Site URL | Base URL used for crawl mode, e.g. `https://example.com` |
| Sitemap URL (optional) | Sitemap source, e.g. `https://example.com/sitemap.xml` |
| Single Page URL (optional) | If provided, only this page is fetched/indexed |
| Page Limit (optional) | Max number of pages to ingest |

Priority order:
1. If **Single Page URL** is provided, only that page is indexed.
2. Otherwise, if **Sitemap URL** is provided, sitemap URLs are indexed.
3. Otherwise, crawler starts from **Site URL**.

## Data and Index

Default paths:

```text
Database: data/search.db
Index:    index/search_index.joblib
```

Database tables:

```text
sites   metadata for each ingestion source
pages   indexed pages and extracted content
```

## Build Index Manually

From JSONL:

```bash
python build_index.py --input data/documents.jsonl --output index/search_index.joblib
```

From SQLite:

```bash
python build_index.py --db data/search.db --output index/search_index.joblib
```

From crawler:

```bash
python build_index.py --crawl https://example.com --max-pages 20 --output index/search_index.joblib
```

## Run Tests

```bash
python -m pytest
```

## Project Structure

```text
mini-search-engine/
├── app.py
├── build_index.py
├── search_cli.py
├── requirements.txt
├── data/
│   └── documents.jsonl
├── index/
│   └── search_index.joblib
├── search_engine/
│   ├── __init__.py
│   ├── crawler.py
│   ├── database.py
│   ├── engine.py
│   └── preprocess.py
├── static/
│   └── style.css
├── templates/
│   ├── add_site.html
│   ├── base.html
│   ├── search.html
│   ├── site_pages.html
│   └── sites.html
└── tests/
    ├── test_engine.py
    └── test_flask_database.py
```

## Notes for Real-World Usage

If you plan to use this beyond local experimentation, consider adding:
- `robots.txt` compliance
- request rate limiting and retries
- background jobs for crawling (Celery/RQ)
- scheduled re-crawling and stale-page cleanup
- stronger multilingual normalization/tokenization
- ranking improvements (link-based signals, typo tolerance, autocomplete)
