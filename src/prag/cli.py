"""Ingestion CLI: ``python -m prag.cli upload FILE...`` / ``list``.

Thin wrapper over :class:`~prag.storage.DocumentStore` for driving Phase 1 by
hand without starting the API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .storage import DocumentStore, NotPdfError, UploadTooLargeError


def _build_store(config_path: str | None) -> DocumentStore:
    settings = load_settings(config_path)
    return DocumentStore(settings.storage_dir, settings.max_upload_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prag", description="Private RAG ingestion CLI")
    parser.add_argument("--config", help="path to a JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="ingest one or more PDF files")
    up.add_argument("paths", nargs="+", type=Path)

    sub.add_parser("list", help="list ingested documents")

    args = parser.parse_args(argv)
    store = _build_store(args.config)

    if args.command == "list":
        for meta in store.list_documents():
            print(f"{meta.document_id}  {meta.size_bytes:>9}B  {meta.filename}")
        return 0

    exit_code = 0
    for path in args.paths:
        try:
            content = path.read_bytes()
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            meta = store.save_pdf(path.name, content)
        except (NotPdfError, UploadTooLargeError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"ingested {path.name} -> {meta.document_id}")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
