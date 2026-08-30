"""Pure conversion: final reranked results -> numbered Evidence -> a delimited evidence block.

No I/O anywhere in this module -- given an already-reranked
`RerankedRetrievalResult` list, it deterministically assigns citation
numbers and renders clearly-delimited evidence text for the generation
prompt. Never includes retrieval-diagnostic scores (see `Evidence`).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..retrieval.models import RerankedRetrievalResult
from .models import Evidence


def build_evidence(results: Sequence[RerankedRetrievalResult]) -> list[Evidence]:
    """Convert final reranked results into numbered `Evidence`.

    Citation numbering is deterministic and follows reranked order
    exactly: `results[i].rank` becomes `citation_number` (reranked rank 1
    -> citation `[1]`, rank 2 -> `[2]`, ...). Never merges two results
    into one citation number, and never reorders `results`.
    """
    return [
        Evidence(
            citation_number=result.rank,
            chunk_id=result.chunk_id,
            text=result.text,
            source_file=result.source_file,
            document_id=result.document_id,
            chunk_index=result.chunk_index,
            section_heading=result.section_heading,
            page_number=result.page_number,
            chunking_strategy=result.chunking_strategy,
            reranked_rank=result.rank,
        )
        for result in results
    ]


def format_evidence_block(evidence: Sequence[Evidence]) -> str:
    """Render numbered evidence as clearly-delimited blocks, e.g.:

        [1]
        Source: authentication-api.md
        Section: Token Lifetime
        Content:
        API access tokens expire after 60 minutes...

    `Section:`/`Page:` lines are omitted (not printed as empty/None) when
    that provenance is absent, rather than fabricating a placeholder
    value. Chunk text is included verbatim (not truncated, paraphrased,
    or otherwise altered) so grounding is judged against the real
    content. The `[n]` / `Source:` / `Section:` / `Page:` / `Content:`
    labels and blank-line block separators are fixed, never derived from
    document content, so a chunk's own text can never be mistaken for a
    new evidence header or citation number -- evidence content is DATA,
    delimited by structure the document itself cannot produce.
    """
    blocks: list[str] = []
    for item in evidence:
        lines = [f"[{item.citation_number}]", f"Source: {item.source_file}"]
        if item.section_heading:
            lines.append(f"Section: {item.section_heading}")
        if item.page_number is not None:
            lines.append(f"Page: {item.page_number}")
        lines.append("Content:")
        lines.append(item.text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
