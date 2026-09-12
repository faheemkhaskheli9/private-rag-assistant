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

    return [chunk for chunk, _, _ in chunk_text_with_offsets(text, config)]


def chunk_text_with_offsets(text: str, config: ChunkConfig) -> list[tuple[str, int, int]]:
    """Like :func:`chunk_text`, but also returns each chunk's ``(start, end)``
    character offset into ``text`` — the single source of truth
    :func:`chunk_text` and :func:`chunk_pages_with_metadata` both build on,
    so the two can never drift apart.

    For ``splitter="recursive"``, the returned offsets describe each chunk's
    own (non-overlapping) source span — the leading overlap text prepended
    from the previous chunk is context, not new source content, so citing
    the offset range should point at what's actually new here.
    """

    lead = len(text) - len(text.lstrip())
    stripped = text.strip()
    if not stripped:
        return []
    if config.splitter == "fixed":
        spans = _chunk_fixed_with_offsets(stripped, config)
    else:
        spans = _chunk_recursive_with_offsets(stripped, config)
    return [(chunk, lead + start, lead + end) for chunk, start, end in spans]


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


@dataclass(frozen=True)
class ChunkRecord:
    """One citation-ready chunk: text plus everything needed to point back
    at its exact source location (issue #3)."""

    document_id: str
    page_number: int  # 1-indexed, matching how PDF viewers number pages
    text: str
    start_offset: int  # character offset into that page's own extracted text
    end_offset: int


def chunk_pages_with_metadata(
    document_id: str, pages: list[str], config: ChunkConfig
) -> list[ChunkRecord]:
    """Chunk each page and tag every chunk with document/page/offset metadata.

    Offsets are relative to that *page's* own extracted text, not the whole
    document — paired with ``page_number``, that's what makes a chunk
    resolvable back to an exact location for citations.
    """

    if not document_id.strip():
        raise ValueError("document_id must not be empty")

    records: list[ChunkRecord] = []
    for page_index, page_text in enumerate(pages):
        page_number = page_index + 1
        for chunk, start, end in chunk_text_with_offsets(page_text, config):
            records.append(
                ChunkRecord(
                    document_id=document_id,
                    page_number=page_number,
                    text=chunk,
                    start_offset=start,
                    end_offset=end,
                )
            )
    return records


def _chunk_fixed(text: str, config: ChunkConfig) -> list[str]:
    return [chunk for chunk, _, _ in _chunk_fixed_with_offsets(text, config)]


def _chunk_fixed_with_offsets(text: str, config: ChunkConfig) -> list[tuple[str, int, int]]:
    step = config.chunk_size - config.chunk_overlap
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + config.chunk_size, n)
        chunks.append((text[start:end], start, end))
        if end == n:
            break
        start += step
    return chunks


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace off both ends of ``text[start:end]``, in absolute offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    prev_end = 0
    for match in _PARAGRAPH_SPLIT.finditer(text):
        spans.append((prev_end, match.start()))
        prev_end = match.end()
    spans.append((prev_end, len(text)))
    return [trimmed for a, b in spans if (trimmed := _trim_span(text, a, b)) is not None]


def _sentence_spans(text: str, para_start: int, para_end: int) -> list[tuple[int, int]]:
    para_text = text[para_start:para_end]
    spans = []
    prev_end = 0
    for match in _SENTENCE_SPLIT.finditer(para_text):
        spans.append((prev_end, match.start()))
        prev_end = match.end()
    spans.append((prev_end, len(para_text)))
    result = []
    for a, b in spans:
        trimmed = _trim_span(para_text, a, b)
        if trimmed is not None:
            result.append((para_start + trimmed[0], para_start + trimmed[1]))
    return result


def _split_units_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into (paragraph, then sentence) units, each tagged with
    its absolute ``(start, end)`` offset in ``text`` — the position-tracking
    counterpart of ``_split_units`` that offset-aware chunking builds on."""
    units = []
    for p_start, p_end in _paragraph_spans(text):
        for s_start, s_end in _sentence_spans(text, p_start, p_end):
            units.append((text[s_start:s_end], s_start, s_end))
    return units


def _split_units(text: str) -> list[str]:
    return [unit for unit, _, _ in _split_units_with_offsets(text)]


def _chunk_recursive(text: str, config: ChunkConfig) -> list[str]:
    return [chunk for chunk, _, _ in _chunk_recursive_with_offsets(text, config)]


def _chunk_recursive_with_offsets(text: str, config: ChunkConfig) -> list[tuple[str, int, int]]:
    units = _split_units_with_offsets(text) or [(text, 0, len(text))]
    chunks: list[tuple[str, int, int]] = []  # (text, start, end) -- core content, no overlap
    current = ""
    current_start = 0
    current_end = 0
    for unit_text, u_start, u_end in units:
        candidate = f"{current} {unit_text}".strip() if current else unit_text
        if len(candidate) <= config.chunk_size:
            if not current:
                current_start = u_start
            current, current_end = candidate, u_end
            continue
        if current:
            chunks.append((current, current_start, current_end))
        if len(unit_text) > config.chunk_size:
            # A single sentence longer than chunk_size: hard-split it rather
            # than emit an oversized chunk, offsetting into the parent text.
            for sub_text, sub_start, sub_end in _chunk_fixed_with_offsets(unit_text, config):
                chunks.append((sub_text, u_start + sub_start, u_start + sub_end))
            current = ""
        else:
            current, current_start, current_end = unit_text, u_start, u_end
    if current:
        chunks.append((current, current_start, current_end))

    if config.chunk_overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for (prev_text, _, _), (cur_text, cur_start, cur_end) in zip(chunks, chunks[1:]):
            tail = prev_text[-config.chunk_overlap :]
            combined = f"{tail} {cur_text}".strip()
            # Offset stays the chunk's own span -- the prepended overlap is
            # repeated context from the previous chunk, not new content.
            overlapped.append((combined, cur_start, cur_end))
        chunks = overlapped
    return chunks
