"""Document storage: atomic PDF persistence + a JSON metadata index.

Design points (portfolio robustness rules):
* Every write is temp-file + ``os.replace`` so an interrupted upload never
  leaves a truncated PDF or a half-written index.
* Document id is a content hash, so re-uploading the same bytes is idempotent
  (returns the existing record) rather than creating a duplicate under a new
  counter/uuid key.
* ``%PDF-`` magic-byte validation — an HTTP upload succeeding says nothing
  about the payload being a real PDF.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_PDF_MAGIC = b"%PDF-"
_DOC_ID_LEN = 16


class NotPdfError(ValueError):
    """Payload is not a PDF (bad magic bytes / extension)."""


class UploadTooLargeError(ValueError):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"upload is {size} bytes; limit is {limit}")
        self.size = size
        self.limit = limit


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    filename: str
    size_bytes: int
    content_sha256: str
    uploaded_at: str
    stored_path: str


def _looks_like_pdf(filename: str, content: bytes) -> bool:
    if not content.startswith(_PDF_MAGIC):
        return False
    return filename.lower().endswith(".pdf")


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class DocumentStore:
    def __init__(self, storage_dir: Path, max_upload_bytes: int) -> None:
        self.storage_dir = Path(storage_dir)
        self.docs_dir = self.storage_dir / "documents"
        self.index_path = self.storage_dir / "metadata.json"
        self.max_upload_bytes = max_upload_bytes

    # --- index helpers ---------------------------------------------------

    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Poisoned index — a partial legacy write. Start clean rather than
            # wedging every future upload.
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict[str, dict]) -> None:
        _atomic_write_bytes(
            self.index_path,
            json.dumps(index, indent=2, sort_keys=True).encode("utf-8"),
        )

    # --- public API ----------------------------------------------------

    def get(self, document_id: str) -> DocumentMetadata | None:
        record = self._read_index().get(document_id)
        return DocumentMetadata(**record) if record else None

    def list_documents(self) -> list[DocumentMetadata]:
        return [DocumentMetadata(**r) for r in self._read_index().values()]

    def save_pdf(self, filename: str, content: bytes) -> DocumentMetadata:
        if len(content) > self.max_upload_bytes:
            raise UploadTooLargeError(len(content), self.max_upload_bytes)
        if not _looks_like_pdf(filename, content):
            raise NotPdfError(f"{filename!r} is not a valid PDF upload")

        sha = hashlib.sha256(content).hexdigest()
        document_id = sha[:_DOC_ID_LEN]

        index = self._read_index()
        existing = index.get(document_id)
        if existing is not None and (self.docs_dir / f"{document_id}.pdf").is_file():
            return DocumentMetadata(**existing)

        stored_path = self.docs_dir / f"{document_id}.pdf"
        _atomic_write_bytes(stored_path, content)

        meta = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            size_bytes=len(content),
            content_sha256=sha,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            stored_path=str(stored_path),
        )
        index[document_id] = asdict(meta)
        self._write_index(index)
        return meta
