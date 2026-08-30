"""Synchronized dense (Chroma) + sparse (BM25) indexing over canonical Chunk corpora.

Given one ordered list of `Chunk` objects, `index_chunks()` builds a
snapshot-identified Chroma collection and a matching BM25 sparse snapshot,
verifies they represent the exact same chunk corpus, and atomically
activates a per-strategy manifest recording the active snapshot. Query-side
retrieval (dense, sparse, or hybrid) is a later pipeline stage and is not
implemented here.
"""

from .exceptions import (
    DenseIndexError,
    IndexingError,
    InvalidChunkCorpusError,
    ManifestError,
    SparseIndexError,
    SynchronizationError,
)
from .fingerprint import compute_snapshot_id
from .manifest import load_manifest, manifest_path, write_manifest
from .models import IndexManifest, canonical_order
from .service import IndexingResult, index_chunks, verify_synchronization
from .tokenizer import TOKENIZER_VERSION, tokenize

__all__ = [
    "TOKENIZER_VERSION",
    "DenseIndexError",
    "IndexManifest",
    "IndexingError",
    "IndexingResult",
    "InvalidChunkCorpusError",
    "ManifestError",
    "SparseIndexError",
    "SynchronizationError",
    "canonical_order",
    "compute_snapshot_id",
    "index_chunks",
    "load_manifest",
    "manifest_path",
    "tokenize",
    "verify_synchronization",
    "write_manifest",
]
