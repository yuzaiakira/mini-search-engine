from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from markupsafe import Markup, escape

from search_engine.crawler import base_url_for, clean_url, crawl, crawl_sitemap, fetch_page_document
from search_engine.database import (
    connect,
    create_site,
    get_site,
    import_documents_as_site,
    init_db,
    list_pages,
    list_sites,
    load_documents,
    total_pages,
    update_site_status,
    upsert_page,
)
from search_engine.engine import SearchEngine, load_jsonl


DEFAULT_DB_PATH = "data/search.db"
DEFAULT_INDEX_PATH = "index/search_index.joblib"


def parse_limit(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None

    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError("لیمیت باید عدد صحیح باشد.") from exc

    if limit <= 0:
        raise ValueError("لیمیت باید بزرگ‌تر از صفر باشد.")
    return limit


def build_engine_from_database(db_path: str | Path, index_path: str | Path | None = None) -> SearchEngine | None:
    with connect(db_path) as conn:
        documents = load_documents(conn)

    if not documents:
        return None

    engine = SearchEngine()
    engine.fit(documents)
    if index_path is not None:
        engine.save(index_path)
    return engine


def ensure_sample_data(db_path: str | Path) -> None:
    sample_path = Path("data/documents.jsonl")
    if not sample_path.exists():
        return

    with connect(db_path) as conn:
        if total_pages(conn) > 0:
            return
        documents = load_jsonl(sample_path)
        import_documents_as_site(
            conn,
            documents,
            base_url="sample://documents",
            source_url=str(sample_path),
            source_type="sample",
        )


def rebuild_app_engine(app: Flask) -> None:
    engine = build_engine_from_database(app.config["DB_PATH"], app.config["INDEX_PATH"])
    app.extensions["search_engine"] = engine


def index_documents_for_source(
    *,
    db_path: str | Path,
    site_url: str,
    sitemap_url: str,
    page_url: str,
    limit_pages: int | None,
) -> tuple[int, int, str]:
    site_url = clean_url(site_url) if site_url else ""
    sitemap_url = clean_url(sitemap_url) if sitemap_url else ""
    page_url = clean_url(page_url) if page_url else ""

    if page_url:
        source_type = "single_page"
        source_url = page_url
        base_url = base_url_for(site_url or page_url)
    elif sitemap_url:
        source_type = "sitemap"
        source_url = sitemap_url
        base_url = base_url_for(site_url or sitemap_url)
    elif site_url:
        source_type = "crawl"
        source_url = site_url
        base_url = base_url_for(site_url)
    else:
        raise ValueError("حداقل یکی از فیلدهای URL سایت، sitemap یا لینک صفحه خاص را وارد کن.")

    with connect(db_path) as conn:
        site_id = create_site(
            conn,
            base_url=base_url,
            source_url=source_url,
            source_type=source_type,
            limit_pages=limit_pages,
            status="running",
        )

        try:
            if source_type == "single_page":
                document = fetch_page_document(page_url)
                documents = [document] if document is not None else []
            elif source_type == "sitemap":
                documents = crawl_sitemap(sitemap_url, max_pages=limit_pages)
            else:
                documents = crawl([site_url], max_pages=limit_pages)

            for document in documents:
                upsert_page(conn, site_id=site_id, document=document)

            if not documents:
                message = "هیچ صفحه قابل ایندکسی پیدا نشد. شاید صفحه HTML نبوده یا متن کافی نداشته است."
                update_site_status(conn, site_id, status="empty", error=message)
            else:
                update_site_status(conn, site_id, status="done")

            return site_id, len(documents), source_type
        except Exception as exc:
            update_site_status(conn, site_id, status="failed", error=str(exc))
            raise


def create_app(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    import_sample: bool = True,
) -> Flask:
    app = Flask(__name__)
    app.secret_key = "dev-secret-change-me"
    app.config["DB_PATH"] = str(db_path)
    app.config["INDEX_PATH"] = str(index_path)

    init_db(db_path)
    if import_sample:
        ensure_sample_data(db_path)
    rebuild_app_engine(app)

    @app.template_filter("highlight")
    def highlight(text: str, query: str) -> Markup:
        safe_text = escape(text or "")
        terms = [re.escape(term) for term in (query or "").split() if len(term) > 1]
        if not terms:
            return Markup(safe_text)

        pattern = re.compile("(" + "|".join(terms) + ")", re.IGNORECASE)
        return Markup(pattern.sub(r"<mark>\1</mark>", str(safe_text)))

    @app.context_processor
    def inject_counts() -> dict[str, int]:
        with connect(app.config["DB_PATH"]) as conn:
            return {"indexed_pages_count": total_pages(conn)}

    @app.get("/")
    @app.get("/search")
    def search_page():
        query = request.args.get("q", "").strip()
        top_k = request.args.get("top_k", 10, type=int) or 10
        top_k = max(1, min(top_k, 50))

        engine: SearchEngine | None = app.extensions.get("search_engine")
        results = engine.search(query, top_k=top_k) if query and engine is not None else []

        return render_template(
            "search.html",
            query=query,
            top_k=top_k,
            results=results,
            engine_ready=engine is not None,
        )

    @app.get("/api/search")
    def search_api():
        query = request.args.get("q", "").strip()
        top_k = request.args.get("top_k", 10, type=int) or 10
        top_k = max(1, min(top_k, 50))
        engine: SearchEngine | None = app.extensions.get("search_engine")
        results = engine.search(query, top_k=top_k) if query and engine is not None else []
        return jsonify({"query": query, "results": [asdict(result) for result in results]})

    @app.route("/add-site", methods=["GET", "POST"])
    def add_site_page():
        if request.method == "POST":
            site_url = request.form.get("site_url", "").strip()
            sitemap_url = request.form.get("sitemap_url", "").strip()
            page_url = request.form.get("page_url", "").strip()

            try:
                limit_pages = parse_limit(request.form.get("limit_pages"))
                site_id, count, source_type = index_documents_for_source(
                    db_path=app.config["DB_PATH"],
                    site_url=site_url,
                    sitemap_url=sitemap_url,
                    page_url=page_url,
                    limit_pages=limit_pages,
                )
                rebuild_app_engine(app)

                if count > 0:
                    flash(f"{count} صفحه با حالت {source_type} ایندکس و به دیتابیس اضافه شد.", "success")
                else:
                    flash("درخواستی ثبت شد اما صفحه قابل ایندکسی پیدا نشد.", "warning")
                return redirect(url_for("site_pages_page", site_id=site_id))
            except ValueError as exc:
                flash(str(exc), "error")
            except Exception as exc:
                flash(f"خطا در گرفتن و ایندکس کردن سایت: {exc}", "error")

        return render_template("add_site.html")

    @app.get("/sites")
    def sites_page():
        with connect(app.config["DB_PATH"]) as conn:
            sites = list_sites(conn)
        return render_template("sites.html", sites=sites)

    @app.get("/sites/<int:site_id>")
    def site_pages_page(site_id: int):
        with connect(app.config["DB_PATH"]) as conn:
            site = get_site(conn, site_id)
            if site is None:
                flash("سایت موردنظر پیدا نشد.", "error")
                return redirect(url_for("sites_page"))
            pages = list_pages(conn, site_id)
        return render_template("site_pages.html", site=site, pages=pages)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flask search engine UI.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path.")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="TF-IDF index path.")
    parser.add_argument("--host", default="127.0.0.1", help="Host.")
    parser.add_argument("--port", type=int, default=8000, help="Port.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(db_path=args.db, index_path=args.index)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
