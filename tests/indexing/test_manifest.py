"""Tests for rag_pipeline.indexing.manifest."""

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing.manifest import load_manifest, manifest_path, write_manifest
from rag_pipeline.indexing.models import MANIFEST_SCHEMA_VERSION, IndexManifest


def _manifest(strategy: ChunkingStrategy, snapshot_id: str = "a" * 64) -> IndexManifest:
    return IndexManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        request_fingerprint="f" * 64,
        chunking_strategy=strategy,
        embedding_model="text-embedding-3-small",
        embedding_dimension=8,
        bm25_tokenizer_version="technical_v1",
        dedup_algorithm_version="cosine_v1",
        dedup_similarity_threshold=0.95,
        pre_dedup_chunk_count=2,
        chunk_count=2,
        duplicate_count=0,
        chunk_ids=("id1", "id2"),
        chroma_collection_name=f"rag-{strategy.value}-{snapshot_id[:12]}",
        sparse_snapshot_path="/tmp/corpus.json",
        dedup_report_path="/tmp/duplicates.json",
    )


def test_load_missing_manifest_returns_none(index_settings: Settings) -> None:
    assert load_manifest(index_settings, ChunkingStrategy.RECURSIVE) is None


def test_write_then_load_round_trips(index_settings: Settings) -> None:
    manifest = _manifest(ChunkingStrategy.RECURSIVE)
    write_manifest(index_settings, manifest)

    loaded = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)

    assert loaded == manifest


def test_manifest_path_is_per_strategy(index_settings: Settings) -> None:
    recursive_path = manifest_path(index_settings, ChunkingStrategy.RECURSIVE)
    fixed_path = manifest_path(index_settings, ChunkingStrategy.FIXED)
    assert recursive_path != fixed_path
    assert recursive_path.name == "recursive.json"
    assert fixed_path.name == "fixed.json"


def test_fixed_and_recursive_manifests_coexist(index_settings: Settings) -> None:
    write_manifest(index_settings, _manifest(ChunkingStrategy.RECURSIVE, "a" * 64))
    write_manifest(index_settings, _manifest(ChunkingStrategy.FIXED, "b" * 64))

    recursive = load_manifest(index_settings, ChunkingStrategy.RECURSIVE)
    fixed = load_manifest(index_settings, ChunkingStrategy.FIXED)

    assert recursive is not None
    assert fixed is not None
    assert recursive.snapshot_id != fixed.snapshot_id
    assert recursive.chunking_strategy == ChunkingStrategy.RECURSIVE
    assert fixed.chunking_strategy == ChunkingStrategy.FIXED


def test_writing_recursive_manifest_does_not_overwrite_fixed(index_settings: Settings) -> None:
    write_manifest(index_settings, _manifest(ChunkingStrategy.FIXED, "b" * 64))
    write_manifest(index_settings, _manifest(ChunkingStrategy.RECURSIVE, "a" * 64))
    write_manifest(index_settings, _manifest(ChunkingStrategy.RECURSIVE, "c" * 64))

    fixed = load_manifest(index_settings, ChunkingStrategy.FIXED)
    assert fixed is not None
    assert fixed.snapshot_id == "b" * 64


def test_write_is_atomic_no_tmp_file_left_behind(index_settings: Settings) -> None:
    manifest = _manifest(ChunkingStrategy.SEMANTIC)
    path = write_manifest(index_settings, manifest)

    tmp_path = path.with_name(f"{path.name}.tmp")
    assert path.exists()
    assert not tmp_path.exists()
