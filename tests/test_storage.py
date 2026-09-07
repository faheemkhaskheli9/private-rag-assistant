import json

import pytest

from prag.storage import (
    DocumentStore,
    NotPdfError,
    UploadTooLargeError,
)


def _store(tmp_path, max_bytes=1_000_000):
    return DocumentStore(tmp_path / "store", max_bytes)


def test_save_pdf_persists_bytes_and_metadata(tmp_path, pdf_bytes):
    store = _store(tmp_path)
    meta = store.save_pdf("report.pdf", pdf_bytes)

    stored = tmp_path / "store" / "documents" / f"{meta.document_id}.pdf"
    assert stored.read_bytes() == pdf_bytes
    assert meta.filename == "report.pdf"
    assert meta.size_bytes == len(pdf_bytes)

    index = json.loads((tmp_path / "store" / "metadata.json").read_text())
    assert meta.document_id in index
    assert index[meta.document_id]["content_sha256"] == meta.content_sha256


def test_reupload_same_bytes_is_idempotent(tmp_path, pdf_bytes):
    store = _store(tmp_path)
    a = store.save_pdf("a.pdf", pdf_bytes)
    b = store.save_pdf("a.pdf", pdf_bytes)
    assert a.document_id == b.document_id
    assert len(store.list_documents()) == 1


def test_non_pdf_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(NotPdfError):
        store.save_pdf("notes.txt", b"just text")
    with pytest.raises(NotPdfError):
        store.save_pdf("fake.pdf", b"<html>not a pdf</html>")


def test_oversize_rejected(tmp_path, pdf_bytes):
    store = _store(tmp_path, max_bytes=10)
    with pytest.raises(UploadTooLargeError):
        store.save_pdf("big.pdf", pdf_bytes)


def test_corrupt_index_is_not_fatal(tmp_path, pdf_bytes):
    store = _store(tmp_path)
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    store.index_path.write_text("{ this is not json")
    meta = store.save_pdf("recovered.pdf", pdf_bytes)
    assert store.get(meta.document_id) is not None


def test_get_unknown_returns_none(tmp_path):
    assert _store(tmp_path).get("deadbeef") is None
