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
) -> str:
    """SHA-256 fingerprint over (ordered chunk IDs, strategy, model, tokenizer version).

    Each `chunk_id` already encodes its document, chunking strategy,
    position, and final text (see `rag_pipeline.chunking.models.
    compute_chunk_id`), so hashing the *ordered* list of chunk IDs is
    sufficient to represent the corpus's contents — there's no need to
    re-hash chunk text directly here. Changing the order, adding/removing a
    chunk, changing the embedding model, or changing the BM25 tokenizer
    version all change the fingerprint.

    Not a random UUID: the same inputs always produce the same ID, which is
    what makes snapshot reuse/idempotence possible.
    """
    payload = json.dumps(
        {
            "chunking_strategy": strategy.value,
            "embedding_model": embedding_model,
            "bm25_tokenizer_version": tokenizer_version,
            "chunk_ids": list(chunk_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
