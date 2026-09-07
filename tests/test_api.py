import pytest
from fastapi.testclient import TestClient

from prag.api import app, get_store
from prag.storage import DocumentStore


@pytest.fixture
def client(tmp_path):
    store = DocumentStore(tmp_path / "store", max_upload_bytes=1_000_000)
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_returns_id_and_metadata(client, pdf_bytes):
    resp = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert body["size_bytes"] == len(pdf_bytes)
    assert len(body["document_id"]) == 16

    got = client.get(f"/documents/{body['document_id']}")
    assert got.status_code == 200
    assert got.json()["document_id"] == body["document_id"]


def test_upload_non_pdf_rejected(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("x.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 415


def test_upload_oversize_rejected(tmp_path, pdf_bytes):
    small = DocumentStore(tmp_path / "s", max_upload_bytes=5)
    app.dependency_overrides[get_store] = lambda: small
    try:
        resp = TestClient(app).post(
            "/documents/upload",
            files={"file": ("big.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 413
    finally:
        app.dependency_overrides.clear()


def test_unknown_document_404(client):
    assert client.get("/documents/nope").status_code == 404
