from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from nltk.stem import PorterStemmer

try:
    from nltk.corpus import stopwords
except Exception:  # pragma: no cover
    stopwords = None


# Fallback list so the project still works before downloading NLTK corpora.
FALLBACK_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "of", "on", "or", "our", "she", "that", "the", "their", "them", "this",
    "to", "was", "we", "were", "will", "with", "you", "your"
}

# Small built-in Persian stopword list. Extend it for your own corpus.
PERSIAN_STOPWORDS = {
    "از", "به", "با", "برای", "در", "که", "این", "آن", "را", "و", "یا", "یک",
    "هر", "هم", "همه", "های", "ها", "تر", "ترین", "تا", "بر", "بین", "اما",
    "اگر", "چون", "پس", "نه", "نیز", "می", "شود", "شده", "شد", "است", "هست",
    "هستند", "بود", "بودند", "باشد", "باشند", "کرد", "کرده", "کنند", "کند",
    "کنم", "کنی", "ما", "من", "تو", "او", "شما", "ایشان", "آنها", "آن‌ها",
    "خود", "دیگر", "باید", "نباید", "روی", "زیر", "داخل", "خارج", "بدون",
    "مثل", "مانند", "فقط", "حتی", "خیلی", "کم", "زیاد", "چند", "چه", "چرا",
    "کجا", "کدام", "وقتی", "زمانی"
}

TOKEN_RE = re.compile(r"[A-Za-z]+|[\u0600-\u06FF]+|\d+")


@lru_cache(maxsize=1)
def english_stopwords() -> set[str]:
    if stopwords is not None:
        try:
            return set(stopwords.words("english"))
        except LookupError:
            pass
    return set(FALLBACK_ENGLISH_STOPWORDS)


def normalize_persian_arabic(text: str) -> str:
    """Normalize Persian/Arabic variants and remove diacritics."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "آ": "آ",
        "\u200c": " ",  # zero-width non-joiner
        "\u200f": " ",
        "\u200e": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove Arabic/Persian diacritics.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


def contains_persian(token: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in token)


def contains_english(token: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in token)


@dataclass
class TextPreprocessor:
    """Tokenizer/normalizer used by TfidfVectorizer.

    For English tokens, Porter stemming is applied.
    For Persian tokens, a lightweight normalizer and stopword remover is used.
    """

    min_token_len: int = 2
    ngram_range: Tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self._stemmer = PorterStemmer()

    def normalize(self, text: str) -> str:
        text = normalize_persian_arabic(text)
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        tokens: list[str] = []

        for raw in TOKEN_RE.findall(text):
            token = raw.strip().lower()

            if len(token) < self.min_token_len:
                continue

            if contains_english(token):
                if token in english_stopwords():
                    continue
                token = self._stemmer.stem(token)
            elif contains_persian(token):
                if token in PERSIAN_STOPWORDS:
                    continue

            if len(token) >= self.min_token_len:
                tokens.append(token)

        return tokens

    def make_ngrams(self, tokens: Sequence[str]) -> List[str]:
        min_n, max_n = self.ngram_range
        output: list[str] = []

        for n in range(min_n, max_n + 1):
            if n <= 0:
                continue
            if n == 1:
                output.extend(tokens)
                continue

            for i in range(0, len(tokens) - n + 1):
                output.append("_".join(tokens[i:i + n]))

        return output

    def analyze(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        return self.make_ngrams(tokens)
