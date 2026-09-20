import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from prag.chunking import (
    ChunkConfig,
    ChunkRecord,
    chunk_pages,
    chunk_pages_with_metadata,
    chunk_text,
    chunk_text_with_offsets,
    extract_pdf_pages,
)

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 20
    + "\n\n"
    + "A second paragraph follows with more sentences here. " * 20
)


def _make_text_pdf(pages_text: list[str]) -> bytes:
    """Build a real, parseable multi-page PDF with actual text content.

    No PDF-authoring dependency (e.g. reportlab) is available in this
    CPU-only, offline sweep environment, so this constructs the minimum
    viable page objects directly via pypdf's low-level writer API instead of
    a hand-rolled xref table.
    """

    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        content_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = content_ref

        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)

        fontdict = DictionaryObject()
        fontdict[NameObject("/F1")] = font_ref
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fontdict
        page[NameObject("/Resources")] = resources

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# --- ChunkConfig validation ---------------------------------------------


def test_default_config_is_valid():
    ChunkConfig()


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_negative_overlap_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(chunk_overlap=-1)


def test_non_positive_chunk_size_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=0)


def test_unknown_splitter_rejected():
    with pytest.raises(ValueError, match="splitter"):
        ChunkConfig(splitter="magic")


# --- chunk_text -----------------------------------------------------------


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", ChunkConfig()) == []


def test_fixed_splitter_respects_chunk_size_and_overlap():
    config = ChunkConfig(chunk_size=50, chunk_overlap=10, splitter="fixed")
    chunks = chunk_text(LONG_TEXT, config)
    assert len(chunks) > 1
    assert all(len(c) <= config.chunk_size for c in chunks)
    # Overlap: the tail of one chunk reappears at the head of the next.
    assert chunks[0][-10:] == chunks[1][:10]


def test_fixed_and_recursive_splitters_produce_different_chunk_counts():
    config_fixed = ChunkConfig(chunk_size=100, chunk_overlap=20, splitter="fixed")
    config_recursive = ChunkConfig(chunk_size=100, chunk_overlap=20, splitter="recursive")
    fixed_chunks = chunk_text(LONG_TEXT, config_fixed)
    recursive_chunks = chunk_text(LONG_TEXT, config_recursive)
    assert len(fixed_chunks) != len(recursive_chunks)


def test_recursive_splitter_keeps_sentences_whole_when_they_fit():
    config = ChunkConfig(chunk_size=200, chunk_overlap=0, splitter="recursive")
    chunks = chunk_text("First sentence here. Second sentence here.", config)
    assert chunks == ["First sentence here. Second sentence here."]


def test_recursive_splitter_hard_splits_an_oversized_sentence():
    config = ChunkConfig(chunk_size=20, chunk_overlap=0, splitter="recursive")
    oversized = "A" * 100 + "."
    chunks = chunk_text(oversized, config)
    assert len(chunks) > 1
    assert all(len(c) <= config.chunk_size for c in chunks)


# --- PDF extraction + end-to-end ingestion --------------------------------


def test_extract_pdf_pages_returns_real_text_per_page():
    pdf = _make_text_pdf(["Hello world page one.", "Second page content here."])
    pages = extract_pdf_pages(pdf)
    assert len(pages) == 2
    assert "Hello world" in pages[0]
    assert "Second page" in pages[1]


def test_malformed_pdf_logs_and_returns_empty_list_instead_of_raising():
    assert extract_pdf_pages(b"not a pdf at all") == []


def test_chunk_pages_switching_splitter_changes_chunk_count_on_same_pdf():
    pdf = _make_text_pdf([LONG_TEXT, LONG_TEXT])
    pages = extract_pdf_pages(pdf)

    fixed = chunk_pages(pages, ChunkConfig(chunk_size=100, chunk_overlap=20, splitter="fixed"))
    recursive = chunk_pages(
        pages, ChunkConfig(chunk_size=100, chunk_overlap=20, splitter="recursive")
    )
    assert len(fixed) != len(recursive)


# --- offset tracking / page metadata (issue #3) ----------------------------


def test_chunk_text_with_offsets_matches_chunk_text():
    config = ChunkConfig(chunk_size=50, chunk_overlap=10, splitter="fixed")
    plain = chunk_text(LONG_TEXT, config)
    with_offsets = chunk_text_with_offsets(LONG_TEXT, config)
    assert [c for c, _, _ in with_offsets] == plain


def test_fixed_offsets_are_exact_slice_bounds():
    text = "abcdefghij"
    config = ChunkConfig(chunk_size=4, chunk_overlap=1, splitter="fixed")
    chunks = chunk_text_with_offsets(text, config)
    for chunk, start, end in chunks:
        assert text[start:end] == chunk


def test_recursive_offsets_point_at_real_source_text():
    text = "First sentence here. Second sentence here."
    config = ChunkConfig(chunk_size=200, chunk_overlap=0, splitter="recursive")
    (chunk, start, end), = chunk_text_with_offsets(text, config)
    assert text[start:end] == chunk == text


def test_recursive_offsets_are_monotonic_across_chunks():
    config = ChunkConfig(chunk_size=60, chunk_overlap=0, splitter="recursive")
    chunks = chunk_text_with_offsets(LONG_TEXT, config)
    starts = [start for _, start, _ in chunks]
    assert starts == sorted(starts)


def test_offsets_account_for_leading_whitespace():
    text = "   \n  Hello world."
    config = ChunkConfig(chunk_size=100, chunk_overlap=0, splitter="fixed")
    (chunk, start, end), = chunk_text_with_offsets(text, config)
    assert text[start:end] == chunk == "Hello world."


def test_chunk_pages_with_metadata_tags_document_and_page_number():
    pdf = _make_text_pdf(["Hello world page one.", "Second page content here."])
    pages = extract_pdf_pages(pdf)
    config = ChunkConfig(chunk_size=200, chunk_overlap=0, splitter="fixed")

    records = chunk_pages_with_metadata("doc-1", pages, config)

    assert all(isinstance(r, ChunkRecord) for r in records)
    assert {r.document_id for r in records} == {"doc-1"}
    page_numbers = [r.page_number for r in records]
    assert page_numbers == sorted(page_numbers)
    assert set(page_numbers) == {1, 2}
    assert any("Hello world" in r.text and r.page_number == 1 for r in records)
    assert any("Second page" in r.text and r.page_number == 2 for r in records)


def test_chunk_pages_with_metadata_offsets_resolve_back_into_the_page_text():
    pdf = _make_text_pdf([LONG_TEXT, LONG_TEXT])
    pages = extract_pdf_pages(pdf)
    config = ChunkConfig(chunk_size=80, chunk_overlap=0, splitter="recursive")

    records = chunk_pages_with_metadata("doc-2", pages, config)
    assert records  # sanity: something was produced
    for record in records:
        page_text = pages[record.page_number - 1]
        assert page_text[record.start_offset : record.end_offset] == record.text


def test_chunk_pages_with_metadata_multipage_document_gets_distinct_page_numbers():
    pdf = _make_text_pdf([LONG_TEXT, LONG_TEXT, LONG_TEXT])
    pages = extract_pdf_pages(pdf)
    config = ChunkConfig(chunk_size=100, chunk_overlap=0, splitter="fixed")

    records = chunk_pages_with_metadata("doc-3", pages, config)
    assert {r.page_number for r in records} == {1, 2, 3}


def test_chunk_pages_with_metadata_empty_document_id_rejected():
    with pytest.raises(ValueError, match="document_id"):
        chunk_pages_with_metadata("   ", ["some text"], ChunkConfig())


def test_chunk_pages_with_metadata_handles_malformed_pdf_gracefully():
    # extract_pdf_pages already returns [] for an unparseable PDF (logged
    # warning, no crash); chunking that empty page list must not crash either.
    pages = extract_pdf_pages(b"not a pdf at all")
    assert chunk_pages_with_metadata("doc-4", pages, ChunkConfig()) == []


# --- exact known chunk counts / boundaries (issue #4 regression coverage) --
#
# The tests above check relative properties (counts differ between
# splitters, overlap holds between the *first* pair). These pin down an
# exact, hand-computed expected chunk count and boundary list for a known
# input, and check overlap across *every* consecutive pair -- so a future
# off-by-one in the fixed-window step calculation fails here with the exact
# expected vs. actual boundaries, not just "some count changed".


def _digits(n: int) -> str:
    """`n` deterministic characters ("0123456789012...") -- easy to eyeball
    slice boundaries against by hand."""
    return "".join(str(i % 10) for i in range(n))


def test_fixed_splitter_exact_chunk_count_and_boundaries_for_known_input():
    text = _digits(50)
    config = ChunkConfig(chunk_size=10, chunk_overlap=3, splitter="fixed")
    chunks = chunk_text_with_offsets(text, config)

    # Hand-computed: step = 10 - 3 = 7; starts 0,7,14,21,28,35,42 (42+10=52
    # clamps to the text's 50 chars, and the loop stops once a chunk reaches
    # the end) -- 7 chunks total.
    expected_starts = [0, 7, 14, 21, 28, 35, 42]
    assert [start for _, start, _ in chunks] == expected_starts, (
        f"expected fixed-window starts {expected_starts}, "
        f"got {[start for _, start, _ in chunks]}"
    )
    assert len(chunks) == 7
    assert chunks[-1][2] == len(text)  # last chunk's end reaches the text end
    for chunk, start, end in chunks:
        assert text[start:end] == chunk


def test_fixed_splitter_overlap_is_exact_between_every_consecutive_pair():
    text = _digits(200)
    config = ChunkConfig(chunk_size=30, chunk_overlap=8, splitter="fixed")
    chunks = chunk_text(text, config)

    assert len(chunks) > 2, "need >2 chunks so this isn't only checking the first pair"
    for i, (prev_chunk, next_chunk) in enumerate(zip(chunks, chunks[1:])):
        assert prev_chunk[-config.chunk_overlap :] == next_chunk[: config.chunk_overlap], (
            f"overlap broken between chunk {i} and {i + 1}: "
            f"tail={prev_chunk[-config.chunk_overlap:]!r} "
            f"head={next_chunk[:config.chunk_overlap]!r}"
        )


