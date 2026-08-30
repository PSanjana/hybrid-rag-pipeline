"""Tests for rag_pipeline.chunking.recursive."""

from rag_pipeline.chunking.recursive import chunk_recursive, pack_structural
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.ingestion.models import Segment
from rag_pipeline.ingestion.normalization import normalize_text

from .conftest import make_document

# Rich technical-documentation text used for content-preservation checks:
# multiple paragraphs, normal sentence punctuation, URLs, technical
# identifiers, and version numbers — none of which should be corrupted by
# splitting on paragraph/newline/sentence/space separators and rejoining.
TECH_TEXT = (
    "Introduction paragraph explaining the system in plain prose with several "
    "complete sentences. It references https://example.com/docs/v1 for more "
    "details, and mentions the config key CONFIG_MAX_RETRIES explicitly.\n\n"
    "Second paragraph covers versioning. Upgrade from v1.2.3 to v2.10.0 requires "
    "restarting the service. Error code ERR-4042 indicates a timeout, while "
    "ERR-5003 indicates an authentication failure.\n\n"
    "Third paragraph has a longer sentence structure, e.g. this one includes an "
    "abbreviation and a trailing URL reference: https://docs.example.org/api?ver=2."
)

# Same spirit as TECH_TEXT but also exercises single-newline splitting
# (distinct from the "\n\n" paragraph separator) within a paragraph.
RECONSTRUCTION_TEXT = normalize_text(
    "Introduction paragraph explaining the system in plain prose with several "
    "complete sentences. It references https://example.com/docs/v1 for more "
    "details, and mentions the config key CONFIG_MAX_RETRIES explicitly.\n\n"
    "Second paragraph covers versioning.\n"
    "Upgrade from v1.2.3 to v2.10.0 requires restarting the service.\n"
    "Error code ERR-4042 indicates a timeout, while ERR-5003 indicates an "
    "authentication failure.\n\n"
    "Third paragraph has a longer sentence structure, e.g. this one includes an "
    "abbreviation and a trailing URL reference: https://docs.example.org/api?ver=2."
)


def test_prefers_paragraph_boundaries_over_mid_paragraph_cuts() -> None:
    paragraphs = [f"Paragraph {i}. " + " ".join(["word"] * 10) for i in range(5)]
    text = "\n\n".join(paragraphs)
    pieces = pack_structural(text, chunk_size=120)
    # Every piece should be an exact join of one or more whole paragraphs,
    # never a mid-paragraph cut.
    for piece in pieces:
        for sub in piece.split("\n\n"):
            assert sub.strip() in paragraphs or sub.strip() == ""


def test_long_paragraph_without_breaks_falls_back_to_word_packing() -> None:
    # No "\n\n", no "\n", no ". " — must fall back to space-splitting.
    text = "word " * 400
    pieces = pack_structural(text, chunk_size=100)
    assert len(pieces) > 1
    assert all(len(piece) <= 100 for piece in pieces)


def test_respects_practical_maximum_size() -> None:
    text = (f"Sentence number {i} ends here. " for i in range(200))
    joined = "".join(text)
    pieces = pack_structural(joined, chunk_size=150)
    assert all(len(piece) <= 150 for piece in pieces)


def test_no_empty_pieces() -> None:
    text = "\n\n\n".join(["", "First.", "", "Second."])
    pieces = pack_structural(text, chunk_size=1000)
    assert all(piece.strip() for piece in pieces)


def test_deterministic_output() -> None:
    text = "\n\n".join(f"Paragraph {i}." * 5 for i in range(10))
    first = pack_structural(text, chunk_size=80)
    second = pack_structural(text, chunk_size=80)
    assert first == second


def test_overlap_is_applied_between_consecutive_chunks() -> None:
    document = make_document(
        segments=(
            Segment(
                text="\n\n".join(f"Paragraph number {i} with some content." for i in range(20)),
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_recursive(document, settings)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        tail = previous.text[-20:]
        assert current.text.startswith(tail)


def test_provenance_preserved_for_markdown_heading() -> None:
    document = make_document(
        file_type="markdown",
        segments=(
            Segment(
                text="\n\n".join(f"Paragraph {i}." * 5 for i in range(10)),
                section_heading="Guide > Setup",
                page_number=None,
            ),
        ),
    )
    settings = Settings(_env_file=None, chunk_size=80, chunk_overlap=10)
    chunks = chunk_recursive(document, settings)
    assert chunks
    assert all(chunk.section_heading == "Guide > Setup" for chunk in chunks)
    assert all(chunk.chunking_strategy == ChunkingStrategy.RECURSIVE for chunk in chunks)


def test_max_size_invariant_holds_after_overlap_stitching() -> None:
    # The bug this guards against: pack_structural to the full chunk_size,
    # then prepending `overlap` more characters, used to produce chunks up
    # to chunk_size + overlap long. Every emitted chunk must now satisfy
    # len(text) <= chunk_size, unconditionally, for a range of configs.
    text = "\n\n".join(
        f"Paragraph number {i} with some meaningful content here." for i in range(40)
    )
    document = make_document(segments=(Segment(text=text, section_heading=None, page_number=None),))
    for chunk_size, overlap in [(100, 20), (150, 50), (1000, 200), (1000, 999)]:
        settings = Settings(_env_file=None, chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunk_recursive(document, settings)
        assert chunks
        assert all(len(chunk.text) <= chunk_size for chunk in chunks), (
            f"violated max size at chunk_size={chunk_size}, overlap={overlap}"
        )


def test_subsequent_chunk_holds_overlap_context_plus_new_content_within_budget() -> None:
    # Matches the example in the spec: chunk_size=1000, overlap=200 -> a
    # subsequent chunk may hold up to 200 chars of previous context plus up
    # to 800 chars of new content, but never exceeds 1000 total.
    text = "\n\n".join(f"Paragraph {i}: " + ("word " * 30) for i in range(30))
    document = make_document(segments=(Segment(text=text, section_heading=None, page_number=None),))
    settings = Settings(_env_file=None, chunk_size=1000, chunk_overlap=200)
    chunks = chunk_recursive(document, settings)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 1000
    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap_region = previous.text[-200:]
        assert current.text.startswith(overlap_region)
        new_content = current.text[200:]
        assert len(new_content) <= 800


def test_max_size_invariant_holds_with_pending_separator_carried_across_oversized_parts() -> None:
    # Small chunk sizes force multiple structural parts (sentences, even
    # paragraphs) to be individually oversized, exercising the "pending"
    # separator carry-forward mechanism most heavily -- exactly where a
    # naive fix could let a prepended separator push a piece over budget.
    for chunk_size in (10, 20, 45, 60):
        pieces = pack_structural(RECONSTRUCTION_TEXT, chunk_size=chunk_size)
        assert all(len(piece) <= chunk_size for piece in pieces), f"chunk_size={chunk_size}"
        assert "".join(pieces) == RECONSTRUCTION_TEXT, f"chunk_size={chunk_size}"


def test_pack_structural_preserves_word_tokens_exactly() -> None:
    # No separator/rejoin step should silently drop, duplicate, or corrupt
    # characters within a word/URL/identifier/version-number token. Tokens
    # are recovered from the *reconstructed* (joined) text, not per piece:
    # a piece boundary can legitimately fall exactly between "sentences"
    # and ". It" (the ". " carried forward onto the next piece rather than
    # trailing the current one), which would make naive per-piece
    # tokenization see two pieces ending/starting mid "sentence." even
    # though the joined text is byte-for-byte identical to the source.
    for chunk_size in (60, 120, 250, 1000):
        pieces = pack_structural(TECH_TEXT, chunk_size=chunk_size)
        assert "".join(pieces).split() == TECH_TEXT.split()


def test_pack_structural_preserves_specific_technical_tokens_verbatim() -> None:
    pieces = pack_structural(TECH_TEXT, chunk_size=120)
    combined = " ".join(pieces)
    for token in (
        "https://example.com/docs/v1",
        "CONFIG_MAX_RETRIES",
        "v1.2.3",
        "v2.10.0",
        "ERR-4042",
        "ERR-5003",
        "https://docs.example.org/api?ver=2.",
    ):
        assert token in combined


def test_pack_structural_reconstructs_normalized_text_exactly() -> None:
    # Character-for-character, not just word-token, reconstruction: every
    # separator (paragraph breaks, single newlines, spaces, ". ") must
    # survive splitting and rejoining, across a range of chunk sizes
    # (including ones small enough to force the character-level fallback
    # for an oversized structural part).
    for chunk_size in (20, 60, 120, 250, 1000):
        pieces = pack_structural(RECONSTRUCTION_TEXT, chunk_size=chunk_size)
        assert "".join(pieces) == RECONSTRUCTION_TEXT, f"failed at chunk_size={chunk_size}"


def _duplicated_prefix_length(previous_text: str, current_text: str, max_overlap: int) -> int:
    """The actual length `_apply_overlap` duplicated onto `current_text`'s start.

    `_apply_overlap` prepends `previous[-overlap:]`, which Python silently
    shrinks to `len(previous)` when the *pre-stitch* previous piece is
    shorter than `overlap` -- so the real duplicated length isn't always
    exactly `overlap`. It's well-defined as the largest `length` (up to
    `max_overlap`) for which `current_text` starts with the same substring
    `previous_text` ends with.
    """
    upper_bound = min(max_overlap, len(previous_text), len(current_text))
    for length in range(upper_bound, 0, -1):
        if current_text[:length] == previous_text[-length:]:
            return length
    return 0


def test_chunk_recursive_reconstructs_normalized_text_exactly_after_removing_overlap() -> None:
    document = make_document(
        segments=(Segment(text=RECONSTRUCTION_TEXT, section_heading=None, page_number=None),)
    )
    for chunk_size, overlap in [(60, 15), (150, 30), (300, 50)]:
        settings = Settings(_env_file=None, chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunk_recursive(document, settings)
        assert len(chunks) > 1

        parts = [chunks[0].text]
        for previous, current in zip(chunks, chunks[1:], strict=False):
            dup_len = _duplicated_prefix_length(previous.text, current.text, overlap)
            parts.append(current.text[dup_len:])
        reconstructed = "".join(parts)

        assert reconstructed == RECONSTRUCTION_TEXT, f"failed at chunk_size={chunk_size}"


def test_chunk_recursive_content_recoverable_after_removing_known_overlap() -> None:
    # Beyond deliberate overlap duplication, the chunk sequence must not
    # lose or corrupt content: stripping the known `overlap`-length prefix
    # from every chunk after the first and concatenating recovers exactly
    # the original word-token sequence.
    document = make_document(
        segments=(Segment(text=TECH_TEXT, section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=150, chunk_overlap=30)
    chunks = chunk_recursive(document, settings)
    assert len(chunks) > 1

    recovered_words: list[str] = []
    for index, chunk in enumerate(chunks):
        new_content = chunk.text if index == 0 else chunk.text[30:]
        recovered_words.extend(new_content.split())

    assert recovered_words == TECH_TEXT.split()


def test_does_not_merge_across_segments() -> None:
    document = make_document(
        file_type="pdf",
        segments=(
            Segment(text="Short page one text.", section_heading=None, page_number=1),
            Segment(text="Short page two text.", section_heading=None, page_number=2),
        ),
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_overlap=100)
    chunks = chunk_recursive(document, settings)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
