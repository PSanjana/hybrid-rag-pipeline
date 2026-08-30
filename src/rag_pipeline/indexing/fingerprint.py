"""Deterministic snapshot fingerprinting for an index build."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ..config import ChunkingStrategy


def compute_snapshot_id(
    chunk_ids: Sequence[str],
    strategy: ChunkingStrategy,
    embedding_model: str,
    tokenizer_version: str,
    dedup_algorithm_version: str,
    dedup_similarity_threshold: float,
) -> str:
    """SHA-256 fingerprint over (ordered chunk IDs, strategy, model, tokenizer, dedup config).

    Each `chunk_id` already encodes its document, chunking strategy,
    position, and final text (see `rag_pipeline.chunking.models.
    compute_chunk_id`), so hashing the *ordered* list of chunk IDs is
    sufficient to represent the corpus's contents — there's no need to
    re-hash chunk text directly here. Changing the order, adding/removing a
    chunk, changing the embedding model, changing the BM25 tokenizer
    version, or changing the deduplication algorithm/threshold all change
    the fingerprint.

    Used for two distinct purposes with two different `chunk_ids` inputs:
    a *pre-dedup request fingerprint* (over the raw, canonically-ordered
    corpus, before deduplication runs) that gates snapshot reuse without
    ever calling the embedding provider, and the *final snapshot_id* (over
    the post-dedup *kept* corpus) that identifies the activated snapshot.
    Including `dedup_similarity_threshold` (a float) is safe here: `json`
    serializes floats via Python's own `repr`-equivalent encoding, which is
    deterministic and locale-independent (unlike e.g. `str.format` under a
    non-"C" locale), so the same threshold value always serializes to the
    same bytes regardless of where this runs.

    Not a random UUID: the same inputs always produce the same ID, which is
    what makes snapshot reuse/idempotence possible.
    """
    payload = json.dumps(
        {
            "chunking_strategy": strategy.value,
            "embedding_model": embedding_model,
            "bm25_tokenizer_version": tokenizer_version,
            "dedup_algorithm_version": dedup_algorithm_version,
            "dedup_similarity_threshold": dedup_similarity_threshold,
            "chunk_ids": list(chunk_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
