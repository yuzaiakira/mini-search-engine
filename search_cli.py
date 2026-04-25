from __future__ import annotations

import argparse

from search_engine.engine import SearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query", nargs="*", help="Search query. Leave empty for interactive mode.")
    parser.add_argument("--index", default="index/search_index.joblib", help="Path to index file.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    return parser.parse_args()


def print_results(query: str, engine: SearchEngine, top_k: int) -> None:
    results = engine.search(query, top_k=top_k)

    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Score: {result.score:.4f} | TF-IDF: {result.tfidf_score:.4f}")
        print(f"   {result.snippet}")


def main() -> None:
    args = parse_args()
    engine = SearchEngine.load(args.index)

    if args.query:
        query = " ".join(args.query)
        print_results(query, engine, args.top_k)
        return

    print("Interactive search. Type 'exit' or press Ctrl+C to quit.")
    while True:
        try:
            query = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break

        if query:
            print_results(query, engine, args.top_k)


if __name__ == "__main__":
    main()
