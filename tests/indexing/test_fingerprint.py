"""Tests for rag_pipeline.indexing.fingerprint."""

from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.deduplication import DEDUP_ALGORITHM_VERSION
from rag_pipeline.indexing.fingerprint import compute_snapshot_id

_IDS = ("id1", "id2", "id3")
_MODEL = "text-embedding-3-small"
_TOKENIZER = "technical_v1"
_THRESHOLD = 0.95


def _fp(
    ids: tuple[str, ...] = _IDS,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    model: str = _MODEL,
    tokenizer: str = _TOKENIZER,
    dedup_version: str = DEDUP_ALGORITHM_VERSION,
    threshold: float = _THRESHOLD,
) -> str:
    return compute_snapshot_id(ids, strategy, model, tokenizer, dedup_version, threshold)


def test_identical_inputs_produce_identical_snapshot_id() -> None:
    assert _fp() == _fp()


def test_snapshot_id_is_sha256_hex() -> None:
    snapshot_id = _fp()
    assert len(snapshot_id) == 64
    int(snapshot_id, 16)  # raises ValueError if not valid hex


def test_changed_chunk_id_changes_snapshot_id() -> None:
    assert _fp() != _fp(ids=("id1", "id2", "DIFFERENT"))


def test_changed_chunk_order_changes_snapshot_id() -> None:
    assert _fp() != _fp(ids=("id3", "id2", "id1"))


def test_changed_embedding_model_changes_snapshot_id() -> None:
    assert _fp() != _fp(model="text-embedding-3-large")


def test_changed_tokenizer_version_changes_snapshot_id() -> None:
    assert _fp() != _fp(tokenizer="technical_v2")


def test_different_chunking_strategy_changes_snapshot_id() -> None:
    recursive = _fp(strategy=ChunkingStrategy.RECURSIVE)
    fixed = _fp(strategy=ChunkingStrategy.FIXED)
    semantic = _fp(strategy=ChunkingStrategy.SEMANTIC)
    assert len({recursive, fixed, semantic}) == 3


def test_changed_dedup_algorithm_version_changes_snapshot_id() -> None:
    assert _fp() != _fp(dedup_version="cosine_v2")


def test_changed_dedup_threshold_changes_snapshot_id() -> None:
    assert _fp() != _fp(threshold=0.90)


def test_request_fingerprint_over_raw_ids_is_stable_for_unchanged_config() -> None:
    # Simulates the pre-dedup request fingerprint: the same raw chunk_ids +
    # config always produce the same fingerprint, independent of any later
    # deduplication outcome.
    raw_ids = ("raw1", "raw2", "raw3")
    assert _fp(ids=raw_ids) == _fp(ids=raw_ids)


def test_changed_raw_chunk_corpus_changes_request_fingerprint() -> None:
    raw_ids = ("raw1", "raw2", "raw3")
    changed_ids = ("raw1", "raw2", "raw3", "raw4")
    assert _fp(ids=raw_ids) != _fp(ids=changed_ids)


def test_same_dedup_result_and_config_produce_same_final_snapshot_id() -> None:
    kept_ids = ("kept1", "kept2")
    assert _fp(ids=kept_ids) == _fp(ids=kept_ids)


def test_final_snapshot_id_changes_when_dedup_threshold_changes_even_with_same_kept_chunks() -> (
    None
):
    # Two builds that happen to keep the exact same post-dedup chunk set
    # under different thresholds must still get different snapshot IDs --
    # configuration provenance is explicit, not incidental.
    kept_ids = ("kept1", "kept2")
    assert _fp(ids=kept_ids, threshold=0.95) != _fp(ids=kept_ids, threshold=0.90)
