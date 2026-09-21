"""Multi-file PDF upload (issue #14): ``POST /documents`` stores one or more
PDFs under the configurable storage path, validates each independently, and
returns a stable document id per file."""

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from prag import api
from prag.api import _read_bounded, _settings, app, get_store
from prag.config import Settings, load_settings
from prag.storage import DocumentStore, UploadTooLargeError


def _pdf(tag: str) -> bytes:
    return b"%PDF-1.4\n% " + tag.encode() + b"\n%%EOF\n"


def _part(name: str, content: bytes, mime: str = "application/pdf"):
    return ("files", (name, content, mime))


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "store", max_upload_bytes=1_000)


@pytest.fixture
def client(store):
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[_settings] = lambda: Settings(max_files_per_upload=3)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_multiple_pdfs_are_stored_with_a_stable_id_each(client, store):
    resp = client.post("/documents", files=[_part("a.pdf", _pdf("a")), _part("b.pdf", _pdf("b"))])

    assert resp.status_code == 201
    body = resp.json()
    assert body["rejected"] == []
    assert [d["filename"] for d in body["uploaded"]] == ["a.pdf", "b.pdf"]
    for doc, content in zip(body["uploaded"], (_pdf("a"), _pdf("b"))):
        assert doc["document_id"] == hashlib.sha256(content).hexdigest()[:16]
        assert (store.docs_dir / f"{doc['document_id']}.pdf").read_bytes() == content
    assert len(store.list_documents()) == 2


def test_single_pdf_upload_works_through_the_same_endpoint(client):
    resp = client.post("/documents", files=[_part("only.pdf", _pdf("only"))])
    assert resp.status_code == 201
    assert len(resp.json()["uploaded"]) == 1


def test_document_id_is_stable_across_repeat_uploads(client, store):
    first = client.post("/documents", files=[_part("a.pdf", _pdf("same"))]).json()
    again = client.post("/documents", files=[_part("renamed.pdf", _pdf("same"))]).json()
    assert first["uploaded"][0]["document_id"] == again["uploaded"][0]["document_id"]
    assert len(store.list_documents()) == 1


def test_files_land_under_the_configured_storage_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAG_STORAGE_DIR", str(tmp_path / "custom-root"))
    resp = TestClient(app).post("/documents", files=[_part("a.pdf", _pdf("a"))])
    assert resp.status_code == 201
    assert list((tmp_path / "custom-root" / "documents").glob("*.pdf"))


def test_bad_file_in_a_batch_is_reported_not_silently_dropped(client, store):
    resp = client.post(
        "/documents",
        files=[
            _part("good.pdf", _pdf("good")),
            _part("notes.txt", b"plain text", "text/plain"),
            _part("fake.pdf", b"not really a pdf"),
        ],
    )

    assert resp.status_code == 207
    body = resp.json()
    assert [d["filename"] for d in body["uploaded"]] == ["good.pdf"]
    assert [(r["filename"], r["status"]) for r in body["rejected"]] == [("notes.txt", 415), ("fake.pdf", 415)]
    assert all(r["detail"] for r in body["rejected"])
    assert len(store.list_documents()) == 1


@pytest.mark.parametrize(
    "name, content, expected_status",
    [
        ("notes.txt", b"plain text", 415),
        ("fake.pdf", b"not really a pdf", 415),
        ("huge.pdf", b"%PDF-" + b"x" * 5_000, 413),
        ("empty.pdf", b"", 400),
    ],
)
def test_batch_where_every_file_is_rejected_is_an_error_with_a_clear_reason(
    client, store, name, content, expected_status
):
    resp = client.post("/documents", files=[_part(name, content)])

    assert resp.status_code == expected_status
    body = resp.json()
    assert body["uploaded"] == []
    assert body["rejected"][0]["filename"] == name
    assert body["rejected"][0]["detail"]
    assert store.list_documents() == []


def test_all_rejected_for_different_reasons_is_a_400(client):
    resp = client.post(
        "/documents",
        files=[_part("notes.txt", b"plain text"), _part("huge.pdf", b"%PDF-" + b"x" * 5_000)],
    )
    assert resp.status_code == 400
    assert {r["status"] for r in resp.json()["rejected"]} == {415, 413}


def test_too_many_files_in_one_request_is_rejected_before_storing_any(client, store):
    parts = [_part(f"{i}.pdf", _pdf(str(i))) for i in range(4)]  # limit is 3
    resp = client.post("/documents", files=parts)
    assert resp.status_code == 400
    assert "too many files" in resp.json()["detail"]
    assert store.list_documents() == []


def test_request_without_files_is_a_validation_error(client):
    assert client.post("/documents").status_code == 422


# --- the size limit bounds memory, not just what gets stored ---------------------

class ChunkedUpload:
    """UploadFile stand-in that records how much the endpoint actually read."""

    def __init__(self, total_bytes: int):
        self.remaining = total_bytes
        self.bytes_served = 0

    async def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        n = self.remaining if size < 0 else min(size, self.remaining)
        self.remaining -= n
        self.bytes_served += n
        return b"x" * n


def test_oversize_upload_is_abandoned_without_reading_the_whole_body():
    upload = ChunkedUpload(total_bytes=50 * api._READ_CHUNK_BYTES)

    with pytest.raises(UploadTooLargeError) as excinfo:
        asyncio.run(_read_bounded(upload, limit=api._READ_CHUNK_BYTES))

    assert upload.bytes_served <= 2 * api._READ_CHUNK_BYTES  # not all 50 chunks
    assert excinfo.value.at_least is True
    assert "at least" in str(excinfo.value)


def test_upload_exactly_at_the_limit_is_accepted():
    upload = ChunkedUpload(total_bytes=1_000)
    assert len(asyncio.run(_read_bounded(upload, limit=1_000))) == 1_000


# --- settings -------------------------------------------------------------------------

def test_settings_follow_the_environment_instead_of_freezing_at_first_use(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAG_STORAGE_DIR", str(tmp_path / "one"))
    assert _settings().storage_dir == tmp_path / "one"
    monkeypatch.setenv("PRAG_STORAGE_DIR", str(tmp_path / "two"))
    assert _settings().storage_dir == tmp_path / "two"


def test_max_files_per_upload_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAG_MAX_FILES_PER_UPLOAD", "7")
    assert load_settings().max_files_per_upload == 7

    config = tmp_path / "config.json"
    config.write_text('{"max_files_per_upload": 2}', encoding="utf-8")
    assert load_settings(config).max_files_per_upload == 2


@pytest.mark.parametrize("field", ["max_files_per_upload", "max_upload_bytes"])
@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_limits_are_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: bad})
