"""Recursive / structure-aware chunking.

Unlike `fixed.py` (which slices raw character windows on a fixed stride),
this strategy tries progressively finer separators — paragraph breaks, line
breaks, sentence/space breaks, then a character-level fallback — and
greedily packs the resulting pieces up to `chunk_size` characters. This
keeps natural structural boundaries (paragraphs, list items, sentences)
intact wherever the text allows it, only falling back to a blind character
cut when no structural separator can produce a small-enough piece.

Each ingestion `Segment` is split independently: pieces are never merged
across segments, so a chunk's `section_heading`/`page_number` provenance is
never ambiguous.

Overlap and the `chunk_size` maximum: pieces are first packed structurally
to a *reduced* budget of `chunk_size - chunk_overlap` characters — reserving
room up front for the overlap that gets added next — then a post-process
stitches the trailing `chunk_overlap` characters of each piece onto the
*start* of the next piece within the same segment. Every emitted chunk
(including the first, which is also packed to the reduced budget even
though it has no previous-chunk context to prepend) therefore satisfies
`len(chunk.text) <= chunk_size`, exactly, unconditionally — never
`chunk_size + chunk_overlap`. A subsequent chunk holds up to
`chunk_overlap` characters of previous context plus up to
`chunk_size - chunk_overlap` characters of new content. The packed pieces
are never re-stripped after stitching, so that overlap region is preserved
byte-for-byte; the stitched text may therefore start with whitespace if the
overlapped tail happened to end mid-whitespace in the source.
"""

from __future__ import annotations

from ..config import ChunkingStrategy, Settings
from ..ingestion.models import NormalizedDocument
from .fixed import _fixed_windows
from .models import Chunk, build_chunk

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def pack_structural(
    text: str, chunk_size: int, separators: tuple[str, ...] = _SEPARATORS
) -> list[str]:
    """Greedily pack `text` into pieces of at most `chunk_size` characters.

    Tries `separators` from largest/most-structural to smallest, splitting
    on the first one present in the text and falling back to the next when
    it isn't. Once no separator remains, falls back to raw fixed-size
    windows (with no overlap — overlap is applied once, across the whole
    result, by the caller).

    Outer whitespace is stripped exactly once, here, at the public entry
    point. The recursive splitting/packing below (`_pack`) never strips
    again, and never truncates a piece before storing it — `.strip()` is
    used only as a *predicate* to decide whether a candidate has any
    non-whitespace content, never applied to the value that gets stored.
    This matters because `str.split(separator)` consumes the separator text
    itself, and a separator like `". "` contains a meaningful, non-whitespace
    character (the period) — discarding it at a piece boundary would
    silently corrupt the sentence (e.g. "sentences" instead of "sentences.").
    Each part has its separator re-attached (to every part but the last)
    before being grouped and stored unmodified, so `"".join(pieces)`
    reconstructs the (once-stripped) input exactly.
    """
    text = text.strip()
    if not text:
        return []
    return _pack(text, chunk_size, separators)


def _pack(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursive packing helper. Assumes `text` is already outer-stripped.

    Splits `text` on `separator` into `raw_parts`, and processes each
    raw part together with the (separator) *suffix* that trailed it in the
    original text — never both together as one string handed to a smaller
    separator's split, which would let that separator match *inside* the
    just-consumed one (e.g. "\\n" matching inside a trailing "\\n\\n") and
    orphan a whitespace-only fragment.

    Instead, a raw part recurses on its own (suffix-free) content, and its
    separator suffix is tracked as `pending`: a small (<= 2 char) string
    carried forward — never split further, never dropped — until it can be
    attached to the front of the next piece. Whenever a piece is about to
    receive a `pending` prefix, its own packing budget is pre-reduced by
    `len(pending)` so the combined result still respects `chunk_size`
    exactly. This guarantees both invariants simultaneously: no separator
    character is ever lost, and no emitted piece exceeds `chunk_size`.
    """
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return _fixed_windows(text, chunk_size, overlap=0)

    separator, *rest = separators
    remaining = tuple(rest)
    if separator not in text:
        return _pack(text, chunk_size, remaining)

    raw_parts = text.split(separator)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    pending = ""  # a separator suffix not yet attached to any piece

    def flush_current() -> None:
        nonlocal current, current_len, pending
        if not current:
            return
        pieces.append(pending + "".join(current))
        pending = ""
        current, current_len = [], 0

    def reserved_budget() -> int:
        return chunk_size - len(pending) if pending else chunk_size

    def emit_oversized(content: str, trailing_separator: str) -> None:
        nonlocal pending
        budget = max(1, chunk_size - len(pending))
        sub_pieces = _pack(content, budget, remaining)
        sub_pieces[0] = pending + sub_pieces[0]
        pending = ""
        pieces.extend(sub_pieces)
        pending = trailing_separator

    current_available = reserved_budget()

    for index, raw_part in enumerate(raw_parts):
        suffix = separator if index < len(raw_parts) - 1 else ""

        if len(raw_part) > chunk_size:
            flush_current()
            emit_oversized(raw_part, suffix)
            current_available = reserved_budget()
            continue

        part = raw_part + suffix
        part_len = len(part)

        if current and current_len + part_len > current_available:
            flush_current()
            current_available = reserved_budget()

        if not current and part_len > current_available:
            emit_oversized(raw_part, suffix)
            current_available = reserved_budget()
            continue

        current.append(part)
        current_len += part_len

    flush_current()
    if pending:
        # Unreachable in practice (the loop above always resolves `pending`
        # by the final iteration, since the last raw_part's suffix is always
        # ""), but kept as a defensive guarantee against losing characters.
        if pieces:
            pieces[-1] += pending
        else:
            pieces.append(pending)

    return pieces


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the trailing `overlap` characters of each piece onto the next.

    Pieces are expected to already be packed to a `chunk_size - overlap`
    budget (see `chunk_recursive`), so the stitched result never exceeds
    `chunk_size`.
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    stitched = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:], strict=False):
        stitched.append(previous[-overlap:] + current)
    return stitched


def chunk_recursive(document: NormalizedDocument, settings: Settings) -> list[Chunk]:
    """Chunk each segment independently using structure-aware recursive packing."""
    chunks: list[Chunk] = []
    chunk_index = 0
    overlap = settings.chunk_overlap
    # Reserve room for the overlap prepended below, so every stitched piece
    # (including the first, packed to the same reduced budget for
    # consistency) stays within settings.chunk_size.
    pack_budget = settings.chunk_size - overlap if overlap > 0 else settings.chunk_size
    for segment in document.segments:
        # pack_structural preserves separator characters exactly (including
        # a trailing "\n\n"/"\n"/" "/". " on an intermediate piece), so no
        # `.strip()` is applied to a piece before storing it here either:
        # doing so would either corrupt a legitimate trailing separator, or
        # — after `_apply_overlap` — trim into the deliberately duplicated
        # overlap region whenever it happens to start with whitespace,
        # silently shrinking the actual overlap below `chunk_overlap`.
        # `.strip()` is used only below, as a predicate, to detect (and
        # skip) a piece that turned out to be pure whitespace.
        pieces = pack_structural(segment.text, pack_budget)
        pieces = _apply_overlap(pieces, overlap)
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                build_chunk(
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=piece,
                    source_file=document.source_file,
                    section_heading=segment.section_heading,
                    page_number=segment.page_number,
                    strategy=ChunkingStrategy.RECURSIVE,
                )
            )
            chunk_index += 1
    return chunks
