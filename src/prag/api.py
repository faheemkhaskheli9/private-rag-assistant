"""FastAPI ingestion surface (Phase 1).

``POST /documents``        -> store one or more PDFs, per-file outcome in the body.
``POST /documents/upload`` -> store a single PDF, return its id + metadata.
``GET  /documents`` / ``GET /documents/{id}`` -> inspect what has been ingested.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import Settings, load_settings
from .storage import DocumentStore, NotPdfError, UploadTooLargeError

app = FastAPI(title="Private RAG Assistant", version="0.1.0")

_READ_CHUNK_BYTES = 1024 * 1024


def _settings() -> Settings:
    # Deliberately not memoised: load_settings() reads PRAG_* environment
    # variables, which a cache keyed on "no arguments" would freeze at the
    # first request and silently ignore afterwards.
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


class RejectedUpload(BaseModel):
    filename: str
    status: int
    detail: str


class BatchUploadOut(BaseModel):
    uploaded: list[DocumentOut]
    rejected: list[RejectedUpload]


async def _read_bounded(file: UploadFile, limit: int) -> bytes:
    """Read an upload, giving up as soon as it exceeds ``limit`` bytes.

    ``await file.read()`` would buffer the entire body before any size check
    could run, so the limit would cap what gets *stored* but not what one
    request can make the server hold in memory.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise UploadTooLargeError(total, limit, at_least=True)
        chunks.append(chunk)
    return b"".join(chunks)


async def _store_upload(file: UploadFile, store: DocumentStore) -> DocumentOut:
    """Validate and persist one upload; raises ``HTTPException`` on rejection."""
    try:
        content = await _read_bounded(file, store.max_upload_bytes)
        if not content:
            raise HTTPException(status_code=400, detail="empty upload")
        meta = store.save_pdf(file.filename or "upload.pdf", content)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except NotPdfError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    return DocumentOut(**meta.__dict__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile, store: DocumentStore = Depends(get_store)
) -> DocumentOut:
    return await _store_upload(file, store)


@app.post(
    "/documents",
    response_model=BatchUploadOut,
    status_code=201,
    responses={
        207: {"model": BatchUploadOut, "description": "Some files stored, some rejected."},
        400: {"description": "No file was stored, or too many files in one request."},
    },
)
async def upload_documents(
    files: list[UploadFile],
    store: DocumentStore = Depends(get_store),
    settings: Settings = Depends(_settings),
) -> JSONResponse:
    """Store one or more PDFs. Each file is validated independently, so one bad
    file does not discard the good ones -- and is never silently dropped: it is
    listed under ``rejected`` and the status code reflects it (201 all stored,
    207 mixed, 4xx none stored)."""
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"too many files: {len(files)}; limit is {settings.max_files_per_upload} per request",
        )

    uploaded: list[DocumentOut] = []
    rejected: list[RejectedUpload] = []
    for file in files:
        try:
            uploaded.append(await _store_upload(file, store))
        except HTTPException as exc:
            rejected.append(
                RejectedUpload(filename=file.filename or "", status=exc.status_code, detail=str(exc.detail))
            )

    if not rejected:
        status = 201
    elif uploaded:
        status = 207
    else:
        # Every file failed: surface the shared cause if there is one.
        statuses = {item.status for item in rejected}
        status = statuses.pop() if len(statuses) == 1 else 400
    body = BatchUploadOut(uploaded=uploaded, rejected=rejected)
    return JSONResponse(status_code=status, content=body.model_dump())


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
