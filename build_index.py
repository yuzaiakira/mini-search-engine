from __future__ import annotations

import argparse

from search_engine.crawler import crawl
from search_engine.database import connect, init_db, load_documents as load_db_documents
from search_engine.engine import SearchEngine, load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TF-IDF search index.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to documents JSONL file.")
    source.add_argument("--crawl", nargs="+", help="Seed URL(s) for the simple crawler.")
    source.add_argument("--db", help="SQLite database path, for example data/search.db.")

    parser.add_argument("--output", default="index/search_index.joblib", help="Output index path.")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages when crawling.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency.")
    parser.add_argument("--max-features", type=int, default=100_000, help="Maximum TF-IDF features.")
    parser.add_argument("--crawled-jsonl", default="data/crawled_documents.jsonl", help="Where to save crawled docs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        documents = load_jsonl(args.input)
    elif args.db:
        init_db(args.db)
        with connect(args.db) as conn:
            documents = load_db_documents(conn)
        if not documents:
            raise SystemExit("Database has no indexed pages.")
    else:
        documents = crawl(args.crawl, max_pages=args.max_pages)
        if not documents:
            raise SystemExit("Crawler returned no documents.")
        save_jsonl(documents, args.crawled_jsonl)
        print(f"Saved crawled documents to {args.crawled_jsonl}")

    engine = SearchEngine(min_df=args.min_df, max_features=args.max_features)
    engine.fit(documents)
    engine.save(args.output)

    print(f"Indexed {len(documents)} documents.")
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    main()
