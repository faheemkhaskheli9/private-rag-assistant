"""FastAPI ingestion surface (Phase 1).

``POST /documents/upload`` -> store a PDF, return its id + metadata.
``GET  /documents`` / ``GET /documents/{id}`` -> inspect what has been ingested.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .config import Settings, load_settings
from .storage import DocumentStore, NotPdfError, UploadTooLargeError

app = FastAPI(title="Private RAG Assistant", version="0.1.0")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_settings()


def get_store(settings: Settings = Depends(_settings)) -> DocumentStore:
    return DocumentStore(settings.storage_dir, settings.max_upload_bytes)


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    content_sha256: str
    uploaded_at: str
    stored_path: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile, store: DocumentStore = Depends(get_store)
) -> DocumentOut:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        meta = store.save_pdf(file.filename or "upload.pdf", content)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except NotPdfError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    return DocumentOut(**meta.__dict__)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(store: DocumentStore = Depends(get_store)) -> list[DocumentOut]:
    return [DocumentOut(**m.__dict__) for m in store.list_documents()]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: str, store: DocumentStore = Depends(get_store)
) -> DocumentOut:
    meta = store.get(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentOut(**meta.__dict__)
