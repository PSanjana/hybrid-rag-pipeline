"""Internal helpers shared by dense and sparse retrieval.

Not part of the public `rag_pipeline.retrieval` API. `retrieval.dense` and
`retrieval.sparse` both read the same kind of Chroma-stored per-chunk
metadata (written once, at indexing time, by
`rag_pipeline.indexing.dense._chunk_metadata`) and need the same query/
top_k/active-manifest handling, so this module is the one place that
implements each of those, rather than dense and sparse quietly
maintaining two subtly different interpretations of the same rules.
"""

from __future__ import annotations

from chromadb.api.types import Metadata

from ..config import ChunkingStrategy, Settings
from ..indexing.manifest import load_manifest
from ..indexing.models import IndexManifest
from .exceptions import IndexNotReadyError, InvalidQueryError, RetrievalError


def validate_query(query: str) -> None:
    if not query.strip():
        raise InvalidQueryError("Query must not be empty or whitespace-only.")


def resolve_top_k(top_k: int | None, default: int) -> int:
    resolved = default if top_k is None else top_k
    if resolved <= 0:
        raise RetrievalError(f"top_k must be positive, got {resolved!r}.")
    return resolved


def load_active_manifest(settings: Settings, strategy: ChunkingStrategy) -> IndexManifest:
    manifest = load_manifest(settings, strategy)
    if manifest is None:
        raise IndexNotReadyError(
            f"No active index snapshot for strategy={strategy.value!r}. Build one with "
            "rag_pipeline.indexing.index_chunks() first."
        )
    return manifest


def require_str(metadata: Metadata, field: str, chunk_id: str, error_cls: type[Exception]) -> str:
    value = metadata.get(field)
    if not isinstance(value, str):
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has a non-string {field!r}: {value!r}."
        )
    return value


def require_int(metadata: Metadata, field: str, chunk_id: str, error_cls: type[Exception]) -> int:
    value = metadata.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has a non-integer {field!r}: {value!r}."
        )
    return value


def optional_str(
    metadata: Metadata, field: str, chunk_id: str, error_cls: type[Exception]
) -> str | None:
    """`field` absent -> None (indexing omits it entirely for a None source value).

    `field` present but not a string is a corruption signal, not an
    absence -- raises rather than silently coercing to None.
    """
    if field not in metadata:
        return None
    value = metadata[field]
    if not isinstance(value, str):
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has a non-string {field!r}: {value!r}."
        )
    return value


def optional_int(
    metadata: Metadata, field: str, chunk_id: str, error_cls: type[Exception]
) -> int | None:
    """`field` absent -> None (indexing omits it entirely for a None source value).

    `field` present but not a non-bool int is a corruption signal, not an
    absence -- raises rather than silently coercing to None.
    """
    if field not in metadata:
        return None
    value = metadata[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has a non-integer {field!r}: {value!r}."
        )
    return value


def parse_chunking_strategy(
    metadata: Metadata,
    chunk_id: str,
    requested_strategy: ChunkingStrategy,
    error_cls: type[Exception],
) -> ChunkingStrategy:
    strategy_value = require_str(metadata, "chunking_strategy", chunk_id, error_cls)
    try:
        chunking_strategy = ChunkingStrategy(strategy_value)
    except ValueError as exc:
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has an invalid "
            f"chunking_strategy={strategy_value!r}."
        ) from exc
    if chunking_strategy != requested_strategy:
        raise error_cls(
            f"Chroma stored metadata for id {chunk_id!r} has chunking_strategy="
            f"{chunking_strategy.value!r}, but strategy={requested_strategy.value!r} was "
            "requested."
        )
    return chunking_strategy
