# Private GPT with RAG

> LLM, RAG & Agentic AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-phase%201%20in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Organizations want document Q&A over their own private files without sending sensitive data to third parties unnecessarily, and with visible citations.

## 2. Architecture

```text
PDF -> Chunk -> Embed -> Vector DB -> Retrieve -> LLM Answer + Citations -> Chat UI
```

## 3. Technology Stack

- Python
- LangChain
- FAISS/Chroma
- Local LLM (e.g., Llama) or OpenAI API
- FastAPI

## 4. Feature List

- PDF upload
- Chunking
- Embedding generation
- Vector database storage
- Semantic retrieval
- Answer citations
- Local or cloud LLM backend option
- Persistent chat memory

## 5. Implementation Plan

1. Phase 1: Ingestion and chunking pipeline
2. Phase 2: Vector store integration and retrieval
3. Phase 3: Answer generation with citation tracking
4. Phase 4: Chat memory and multi-document sessions

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not in this file. To see what's open:

```bash
gh issue list --repo faheemkhaskheli9/private-rag-assistant --state open --label type:user-story
```

Implement Phase 1 issues first (later phases depend on it). When you start one, add label `status:in-progress`. When you finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
private-rag-assistant/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd private-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Phase 1 ships the ingestion entrypoints (PDF upload + metadata index). No API
key or model is required.

```bash
pip install -r requirements.txt

# HTTP API
PYTHONPATH=src uvicorn prag.api:app --reload
curl -s -F 'file=@examples/sample.pdf' localhost:8000/documents/upload
# one or more PDFs in a single request (repeat the `files` field)
curl -s -F 'files=@first.pdf' -F 'files=@second.pdf' localhost:8000/documents

# CLI
PYTHONPATH=src python -m prag.cli upload examples/sample.pdf
PYTHONPATH=src python -m prag.cli list
```

VS Code: **PRAG: FastAPI (uvicorn)**, **PRAG: CLI (upload)**, **PRAG: pytest**
in `.vscode/launch.json`.

`POST /documents` validates each file independently (`%PDF-` magic bytes +
`.pdf` name, `PRAG_MAX_UPLOAD_BYTES`) and reports a per-file outcome:
`{"uploaded": [...], "rejected": [{"filename", "status", "detail"}]}` with
`201` when every file was stored, `207` when some were rejected, and a `4xx`
when none were — a bad file never silently disappears from a batch. Document
ids are a hash of the file's bytes, so re-uploading the same PDF returns the
same id. Uploads are read in bounded chunks and abandoned as soon as they pass
the size limit, so the limit caps memory as well as disk; a request may carry
at most `PRAG_MAX_FILES_PER_UPLOAD` files (default 20). Files are stored under
`PRAG_STORAGE_DIR` (default `data/store`).

Every chunk produced during ingestion carries citation-ready metadata
(`prag.chunking.ChunkRecord`): `document_id`, 1-indexed `page_number`, and a
`(start_offset, end_offset)` character range into that page's own extracted
text — so a later answer can cite the exact source location, not just "this
document". Offsets are the single source of truth `chunk_text` builds on
(`chunk_text_with_offsets`), so the plain-text and metadata-tagged chunking
paths can never drift apart:

```python
from prag.chunking import ChunkConfig, chunk_pages_with_metadata, extract_pdf_pages

pages = extract_pdf_pages(pdf_bytes)
records = chunk_pages_with_metadata("doc-1", pages, ChunkConfig())
# ChunkRecord(document_id='doc-1', page_number=2, text='...', start_offset=0, end_offset=612)
```

A malformed/unreadable PDF (multi-column layouts pypdf can't extract, or a
corrupt file) logs a warning and yields no pages/chunks rather than crashing
the ingestion pipeline — see `extract_pdf_pages`.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t private-rag-assistant .
docker run -p 8000:8000 private-rag-assistant
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
