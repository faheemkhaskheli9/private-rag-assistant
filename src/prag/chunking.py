"""Config-driven text chunking for RAG ingestion.

Chunk size, overlap, and splitter type come from :class:`ChunkConfig` — read
from `configs/*.yaml` via :mod:`prag.config`, never hardcoded — so different
chunking strategies can be compared on the same document without a code
change (issue #2). Attaching page/offset metadata to each chunk for
citations is issue #3's job; this module only turns raw page text into chunk
strings.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

_logger = logging.getLogger(__name__)

_VALID_SPLITTERS = {"fixed", "recursive"}


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    splitter: str = "fixed"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.splitter not in _VALID_SPLITTERS:
            raise ValueError(
                f"splitter must be one of {sorted(_VALID_SPLITTERS)}, got {self.splitter!r}"
            )


def extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
    """Extract per-page text from a PDF.

    A malformed/unreadable PDF logs a warning and returns ``[]`` rather than
    crashing the ingestion pipeline (a multi-column or corrupt PDF is
    expected input here, not an exceptional one). Page-numbered metadata for
    citations is added in issue #3; this returns plain page text in order.
    """

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as exc:
        _logger.warning("failed to parse PDF: %s", exc)
        return []


def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    """Split ``text`` into chunks per ``config``.

    ``splitter="fixed"`` slides a fixed-size window with the configured
    overlap, ignoring content boundaries. ``splitter="recursive"`` first
    splits on paragraph/sentence boundaries and greedily packs them up to
    ``chunk_size``, falling back to a hard split only for a single unit
    longer than ``chunk_size``. The two strategies land chunk boundaries at
    different points, so they produce measurably different chunk counts on
    the same input.
    """

    text = text.strip()
    if not text:
        return []
    if config.splitter == "fixed":
        return _chunk_fixed(text, config)
    return _chunk_recursive(text, config)


def chunk_pages(pages: list[str], config: ChunkConfig) -> list[str]:
    """Chunk each extracted page independently and flatten the result.

    Chunking per page (rather than concatenating the whole document first)
    keeps this ready for issue #3 to attach a page number to each chunk
    without having to re-derive page boundaries after the fact.
    """

    chunks: list[str] = []
    for page_text in pages:
        chunks.extend(chunk_text(page_text, config))
    return chunks


def _chunk_fixed(text: str, config: ChunkConfig) -> list[str]:
    step = config.chunk_size - config.chunk_overlap
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + config.chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str) -> list[str]:
    units: list[str] = []
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        units.extend(s for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return units


def _chunk_recursive(text: str, config: ChunkConfig) -> list[str]:
    units = _split_units(text) or [text]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= config.chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(unit) > config.chunk_size:
            # A single sentence longer than chunk_size: hard-split it rather
            # than emit an oversized chunk.
            chunks.extend(_chunk_fixed(unit, config))
            current = ""
        else:
            current = unit
    if current:
        chunks.append(current)

    if config.chunk_overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-config.chunk_overlap :]
            overlapped.append(f"{tail} {cur}".strip())
        chunks = overlapped
    return chunks
