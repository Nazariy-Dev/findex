"""ЛР7: маршрути через TestClient; індекс підставляється через dependency_overrides."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from findex.ranking import Index
from findex.web.app import create_app, get_index
from findex.web.settings import Settings


@pytest.fixture
def client(small_index: Index) -> Iterator[TestClient]:
    app = create_app(Settings(), load=False)  # lifespan не читає файл з диска
    app.dependency_overrides[get_index] = lambda: small_index
    app.state.index = small_index  # для /health і HTML-сторінок
    with TestClient(app) as c:
        yield c


def test_search_ok(client: TestClient) -> None:
    r = client.get("/search", params={"q": "event loop", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1 and body["results"][0]["path"] == "asyncio.rst"
    assert r.headers["x-request-id"]


def test_pagination(client: TestClient) -> None:
    a = client.get("/search", params={"q": "event OR the OR a", "k": 1, "page": 1}).json()
    b = client.get("/search", params={"q": "event OR the OR a", "k": 1, "page": 2}).json()
    assert a["pages"] >= 2 and a["results"][0]["doc_id"] != b["results"][0]["doc_id"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "k": "abc"},
        {"q": "x", "scorer": "nope"},
        {"q": "(a OR"},
        {},
    ],
)
def test_validation_422(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_doc_404_and_200(client: TestClient) -> None:
    assert client.get("/docs/9999").status_code == 404
    r = client.get("/docs/0")
    assert r.status_code == 200 and r.json()["doc_id"] == 0


def test_stats_and_health(client: TestClient) -> None:
    assert client.get("/stats").json()["documents"] == 5
    assert client.get("/health").json() == {"status": "ok"}


def test_health_503_when_not_loaded() -> None:
    app = create_app(Settings(), load=False)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/search", params={"q": "x"}).status_code == 503


def test_unexpected_error_is_clean_500(small_index: Index) -> None:
    app = create_app(Settings(), load=False)

    def broken(request: Request) -> Index:
        raise RuntimeError("secret stack detail")

    app.dependency_overrides[get_index] = broken
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/search", params={"q": "x"})
    assert r.status_code == 500
    assert "secret" not in r.text and "request_id" in r.json()


def test_html_page(client: TestClient) -> None:
    r = client.get("/", params={"q": "queryset"})
    assert r.status_code == 200 and "<mark>queryset</mark>" in r.text.lower()
    assert client.get("/doc/0").status_code == 200
