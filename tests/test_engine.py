from search_engine.engine import Document, SearchEngine


def test_search_returns_relevant_document():
    docs = [
        Document(id="1", title="Python Machine Learning", url="https://example.com/ml", content="A guide about machine learning with Python."),
        Document(id="2", title="Persian cooking", url="https://example.com/cook", content="A recipe for rice and stew."),
    ]

    engine = SearchEngine()
    engine.fit(docs)

    results = engine.search("machine learning", top_k=1)
    assert results
    assert results[0].id == "1"


def test_persian_search_returns_relevant_document():
    docs = [
        Document(id="1", title="موتور جستجو", url="https://example.com/search", content="ایندکس سازی و رتبه بندی اسناد در موتور جستجو مهم است."),
        Document(id="2", title="آشپزی", url="https://example.com/cook", content="طرز تهیه غذاهای ایرانی."),
    ]

    engine = SearchEngine()
    engine.fit(docs)

    results = engine.search("رتبه بندی جستجو", top_k=1)
    assert results
    assert results[0].id == "1"
