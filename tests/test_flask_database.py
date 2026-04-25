from app import create_app
from search_engine.database import connect, create_site, list_pages, load_documents, upsert_page
from search_engine.engine import Document


def test_database_stores_pages(tmp_path):
    db_path = tmp_path / "search.db"
    app = create_app(db_path=db_path, index_path=tmp_path / "index.joblib", import_sample=False)

    with connect(db_path) as conn:
        site_id = create_site(
            conn,
            base_url="https://example.com/",
            source_url="https://example.com/",
            source_type="single_page",
            limit_pages=1,
        )
        upsert_page(
            conn,
            site_id=site_id,
            document=Document(
                id="1",
                title="Example Search Page",
                url="https://example.com/",
                content="This page is about search engines and indexing.",
            ),
        )
        pages = list_pages(conn, site_id)
        documents = load_documents(conn)

    assert app is not None
    assert len(pages) == 1
    assert documents[0].title == "Example Search Page"


def test_flask_search_page_loads(tmp_path):
    app = create_app(db_path=tmp_path / "search.db", index_path=tmp_path / "index.joblib", import_sample=False)
    client = app.test_client()

    response = client.get("/search")

    assert response.status_code == 200
    assert "Mini Google" in response.get_data(as_text=True)
