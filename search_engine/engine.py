from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import TextPreprocessor


@dataclass
class Document:
    id: str
    title: str
    url: str
    content: str


@dataclass
class SearchResult:
    id: str
    title: str
    url: str
    snippet: str
    score: float
    tfidf_score: float


def load_jsonl(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc

            missing = {"id", "title", "url", "content"} - set(item)
            if missing:
                raise ValueError(f"Line {line_no} is missing fields: {sorted(missing)}")

            docs.append(
                Document(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    url=str(item["url"]),
                    content=str(item["content"]),
                )
            )

    if not docs:
        raise ValueError(f"No documents found in {path}")

    return docs


def save_jsonl(documents: Sequence[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


class SearchEngine:
    def __init__(
        self,
        *,
        min_df: int = 1,
        max_features: int | None = 100_000,
        ngram_range: tuple[int, int] = (1, 2),
        title_weight: int = 3,
    ) -> None:
        self.preprocessor = TextPreprocessor(ngram_range=ngram_range)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.preprocessor.analyze,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
            min_df=min_df,
            max_features=max_features,
        )
        self.title_weight = title_weight
        self.documents: list[Document] = []
        self.matrix = None

    def _document_text(self, doc: Document) -> str:
        # Repeat title so title matches naturally get more TF-IDF weight.
        weighted_title = " ".join([doc.title] * self.title_weight)
        return f"{weighted_title}\n{doc.url}\n{doc.content}"

    def fit(self, documents: Sequence[Document]) -> "SearchEngine":
        if not documents:
            raise ValueError("SearchEngine.fit() needs at least one document.")

        self.documents = list(documents)
        corpus = [self._document_text(doc) for doc in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    def save(self, path: str | Path) -> None:
        if self.matrix is None:
            raise RuntimeError("Build the index with fit() before saving.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "documents": self.documents,
                "matrix": self.matrix,
                "vectorizer": self.vectorizer,
                "preprocessor": self.preprocessor,
                "title_weight": self.title_weight,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SearchEngine":
        payload = joblib.load(path)
        engine = cls(title_weight=payload.get("title_weight", 3))
        engine.documents = payload["documents"]
        engine.matrix = payload["matrix"]
        engine.vectorizer = payload["vectorizer"]
        engine.preprocessor = payload.get("preprocessor", engine.preprocessor)
        return engine

    def _metadata_boost(self, query: str, doc: Document) -> float:
        query_norm = self.preprocessor.normalize(query)
        title_norm = self.preprocessor.normalize(doc.title)
        url_norm = self.preprocessor.normalize(doc.url)

        query_tokens = set(self.preprocessor.tokenize(query))
        title_tokens = set(self.preprocessor.tokenize(doc.title))

        boost = 0.0

        # Exact phrase in title is useful for a small Google-like boost.
        if query_norm and query_norm in title_norm:
            boost += 0.12

        # Token overlap with title.
        if query_tokens:
            overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
            boost += 0.08 * overlap

        # Query terms in URL can be mildly useful.
        for token in query_tokens:
            if token in url_norm:
                boost += 0.01

        return min(boost, 0.25)

    def _make_snippet(self, query: str, doc: Document, max_chars: int = 260) -> str:
        content = re.sub(r"\s+", " ", doc.content).strip()
        if not content:
            return ""

        query_tokens = set(self.preprocessor.tokenize(query))
        if not query_tokens:
            return content[:max_chars]

        # Split around sentence punctuation for a readable snippet.
        sentences = re.split(r"(?<=[.!؟?])\s+", content)
        best_sentence = ""
        best_score = -1

        for sentence in sentences:
            sentence_tokens = set(self.preprocessor.tokenize(sentence))
            score = len(query_tokens & sentence_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        snippet = best_sentence if best_sentence else content
        if len(snippet) <= max_chars:
            return snippet

        # Keep the middle around the first matching term when possible.
        normalized_snippet = self.preprocessor.normalize(snippet)
        positions = [normalized_snippet.find(token) for token in query_tokens if token in normalized_snippet]
        positions = [p for p in positions if p >= 0]

        if positions:
            center = min(positions)
            start = max(0, center - max_chars // 3)
        else:
            start = 0

        end = min(len(snippet), start + max_chars)
        clipped = snippet[start:end].strip()

        if start > 0:
            clipped = "..." + clipped
        if end < len(snippet):
            clipped = clipped + "..."

        return clipped

    def search(self, query: str, *, top_k: int = 10, min_score: float = 0.0) -> list[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Index is empty. Run fit() or load an index first.")

        query = query.strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.matrix).ravel()

        final_scores = []
        for i, doc in enumerate(self.documents):
            final_scores.append(float(tfidf_scores[i]) + self._metadata_boost(query, doc))

        ranked_indices = np.argsort(final_scores)[::-1]

        results: list[SearchResult] = []
        for idx in ranked_indices[:top_k * 3]:
            final_score = float(final_scores[idx])
            tfidf_score = float(tfidf_scores[idx])

            if final_score <= min_score:
                continue

            doc = self.documents[int(idx)]
            results.append(
                SearchResult(
                    id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    snippet=self._make_snippet(query, doc),
                    score=round(final_score, 6),
                    tfidf_score=round(tfidf_score, 6),
                )
            )

            if len(results) >= top_k:
                break

        return results
