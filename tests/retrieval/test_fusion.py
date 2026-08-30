"""Tests for rag_pipeline.retrieval.fusion (pure weighted RRF, no I/O)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.retrieval.exceptions import FusionError
from rag_pipeline.retrieval.fusion import _FusedCandidate, _sort_key, fuse_rankings

from .conftest import make_dense_result, make_sparse_result

_DENSE_WEIGHT = 0.7
_SPARSE_WEIGHT = 0.3
_RANK_CONSTANT = 60
_TOP_K = 10


def _fuse(dense, sparse, **overrides):
    params = {
        "dense_weight": _DENSE_WEIGHT,
        "sparse_weight": _SPARSE_WEIGHT,
        "rank_constant": _RANK_CONSTANT,
        "top_k": _TOP_K,
        **overrides,
    }
    return fuse_rankings(dense, sparse, **params)


# --- mathematical correctness (formula) -----------------------------------------


def test_dense_only_chunk_gets_correct_dense_contribution() -> None:
    # Contiguous 1..3 dense ranking (required -- see _validate_ranking);
    # the chunk of interest ("a") sits at rank 3.
    dense = [
        make_dense_result(chunk_id="filler-1", rank=1),
        make_dense_result(chunk_id="filler-2", rank=2),
        make_dense_result(chunk_id="a", rank=3),
    ]
    results = _fuse(dense, [])
    target = next(r for r in results if r.chunk_id == "a")
    assert target.dense_contribution == pytest.approx(_DENSE_WEIGHT / (_RANK_CONSTANT + 3))
    assert target.sparse_contribution == 0.0
    assert target.rrf_score == pytest.approx(target.dense_contribution)


def test_sparse_only_chunk_gets_correct_sparse_contribution() -> None:
    sparse = [
        make_sparse_result(chunk_id="filler-1", rank=1),
        make_sparse_result(chunk_id="filler-2", rank=2),
        make_sparse_result(chunk_id="filler-3", rank=3),
        make_sparse_result(chunk_id="filler-4", rank=4),
        make_sparse_result(chunk_id="a", rank=5),
    ]
    results = _fuse([], sparse)
    target = next(r for r in results if r.chunk_id == "a")
    assert target.sparse_contribution == pytest.approx(_SPARSE_WEIGHT / (_RANK_CONSTANT + 5))
    assert target.dense_contribution == 0.0
    assert target.rrf_score == pytest.approx(target.sparse_contribution)


def test_overlap_gets_sum_of_both_contributions() -> None:
    dense = [
        make_dense_result(chunk_id="filler-1", rank=1),
        make_dense_result(chunk_id="a", rank=2),
    ]
    sparse = [
        make_sparse_result(chunk_id="filler-1", rank=1),
        make_sparse_result(chunk_id="filler-2", rank=2),
        make_sparse_result(chunk_id="filler-3", rank=3),
        make_sparse_result(chunk_id="a", rank=4),
    ]
    results = _fuse(dense, sparse)
    target = next(r for r in results if r.chunk_id == "a")
    expected_dense = _DENSE_WEIGHT / (_RANK_CONSTANT + 2)
    expected_sparse = _SPARSE_WEIGHT / (_RANK_CONSTANT + 4)
    assert target.dense_contribution == pytest.approx(expected_dense)
    assert target.sparse_contribution == pytest.approx(expected_sparse)
    assert target.rrf_score == pytest.approx(expected_dense + expected_sparse)


def test_exact_formula_for_dense_rank_one() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1)]
    results = _fuse(dense, [])
    assert results[0].dense_contribution == pytest.approx(0.7 / 61)


def test_exact_formula_for_sparse_rank_one() -> None:
    sparse = [make_sparse_result(chunk_id="a", rank=1)]
    results = _fuse([], sparse)
    assert results[0].sparse_contribution == pytest.approx(0.3 / 61)


def test_rank_constant_is_used_in_denominator() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1)]
    results_k60 = _fuse(dense, [], rank_constant=60)
    results_k0 = _fuse(dense, [], rank_constant=0)
    assert results_k60[0].dense_contribution == pytest.approx(0.7 / 61)
    assert results_k0[0].dense_contribution == pytest.approx(0.7 / 1)


def test_dense_weight_change_changes_dense_contribution() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1)]
    low = _fuse(dense, [], dense_weight=0.1)
    high = _fuse(dense, [], dense_weight=0.9)
    assert low[0].dense_contribution == pytest.approx(0.1 / 61)
    assert high[0].dense_contribution == pytest.approx(0.9 / 61)
    assert high[0].dense_contribution > low[0].dense_contribution


def test_sparse_weight_change_changes_sparse_contribution() -> None:
    sparse = [make_sparse_result(chunk_id="a", rank=1)]
    low = _fuse([], sparse, sparse_weight=0.05)
    high = _fuse([], sparse, sparse_weight=0.95)
    assert low[0].sparse_contribution == pytest.approx(0.05 / 61)
    assert high[0].sparse_contribution == pytest.approx(0.95 / 61)
    assert high[0].sparse_contribution > low[0].sparse_contribution


def test_native_score_changes_do_not_alter_rrf_score_when_ranks_unchanged() -> None:
    dense_low = [
        make_dense_result(chunk_id="filler", rank=1),
        make_dense_result(chunk_id="a", rank=2, distance=0.9),
    ]
    dense_high = [
        make_dense_result(chunk_id="filler", rank=1),
        make_dense_result(chunk_id="a", rank=2, distance=0.01),
    ]
    sparse_low = [make_sparse_result(chunk_id="b", rank=1, bm25_score=0.001)]
    sparse_high = [make_sparse_result(chunk_id="b", rank=1, bm25_score=500.0)]

    results_low = _fuse(dense_low, sparse_low)
    results_high = _fuse(dense_high, sparse_high)

    by_id_low = {r.chunk_id: r.rrf_score for r in results_low}
    by_id_high = {r.chunk_id: r.rrf_score for r in results_high}
    assert by_id_low == pytest.approx(by_id_high)


def test_candidate_union_includes_dense_only_and_sparse_only_chunks() -> None:
    dense = [make_dense_result(chunk_id="dense-only", rank=1)]
    sparse = [make_sparse_result(chunk_id="sparse-only", rank=1)]
    results = _fuse(dense, sparse)
    chunk_ids = {r.chunk_id for r in results}
    assert chunk_ids == {"dense-only", "sparse-only"}


def test_fuse_rankings_rejects_negative_dense_weight() -> None:
    with pytest.raises(FusionError):
        _fuse([], [], dense_weight=-0.1)


def test_fuse_rankings_rejects_negative_sparse_weight() -> None:
    with pytest.raises(FusionError):
        _fuse([], [], sparse_weight=-0.1)


def test_fuse_rankings_rejects_both_weights_zero() -> None:
    with pytest.raises(FusionError):
        _fuse([], [], dense_weight=0.0, sparse_weight=0.0)


def test_fuse_rankings_rejects_negative_rank_constant() -> None:
    with pytest.raises(FusionError):
        _fuse([], [], rank_constant=-1)


def test_fuse_rankings_rejects_non_positive_top_k() -> None:
    with pytest.raises(FusionError):
        _fuse([], [], top_k=0)


# --- ranking / ordering ----------------------------------------------------------


def test_higher_rrf_score_ranks_first() -> None:
    dense = [
        make_dense_result(chunk_id="a", rank=1),
        make_dense_result(chunk_id="filler-2", rank=2),
        make_dense_result(chunk_id="filler-3", rank=3),
        make_dense_result(chunk_id="filler-4", rank=4),
        make_dense_result(chunk_id="b", rank=5),
    ]
    results = _fuse(dense, [])
    by_id = {r.chunk_id: r.rrf_score for r in results}
    assert by_id["a"] > by_id["b"]
    assert results[0].chunk_id == "a"
    assert results[-1].chunk_id == "b"


def test_overlapping_chunk_can_outrank_single_channel_results() -> None:
    # "a" is dense-only at rank 1 (strong dense signal alone).
    # "b" is a weaker overlap (dense rank 5 + sparse rank 5), but the sum
    # of both weighted contributions can still exceed "a"'s single term.
    dense = [
        make_dense_result(chunk_id="c", rank=1),
        make_dense_result(chunk_id="d", rank=2),
        make_dense_result(chunk_id="e", rank=3),
        make_dense_result(chunk_id="f", rank=4),
        make_dense_result(chunk_id="b", rank=5),
    ]
    sparse = [make_sparse_result(chunk_id="b", rank=1)]
    results = _fuse(dense, sparse)
    by_id = {r.chunk_id: r.rrf_score for r in results}
    # b: dense rank 5 + sparse rank 1; c: dense rank 1 only.
    assert by_id["b"] == pytest.approx(0.7 / 65 + 0.3 / 61)
    assert by_id["c"] == pytest.approx(0.7 / 61)
    assert by_id["b"] > by_id["c"]
    assert [r.chunk_id for r in results][0] == "b"


def test_equal_rrf_score_and_best_rank_ties_broken_by_dense_rank_presence() -> None:
    # With equal weights, a dense-only rank-1 chunk and a sparse-only
    # rank-1 chunk score identically (0.5/61 each) and tie on "best
    # available rank" too (min(1, inf) == min(inf, 1) == 1) -- the next
    # tie-break level, dense rank ASC (missing treated as +inf), then
    # prefers the dense-only candidate, since 1 < inf.
    dense = [make_dense_result(chunk_id="zeta", rank=1)]
    sparse = [make_sparse_result(chunk_id="alpha", rank=1)]
    results = _fuse(dense, sparse, dense_weight=0.5, sparse_weight=0.5)
    assert results[0].rrf_score == pytest.approx(results[1].rrf_score)
    assert [r.chunk_id for r in results] == ["zeta", "alpha"]


def test_sort_key_falls_back_to_chunk_id_ascending_when_fully_tied() -> None:
    # Two distinct chunk_ids can never actually tie on both dense rank and
    # sparse rank when produced by fuse_rankings() itself: ranks are
    # unique *within* each single-channel ranking (enforced by
    # _validate_ranking), so two different chunk_ids sharing the same
    # dense_rank -- or the same sparse_rank -- is already rejected before
    # sorting. This directly unit-tests the sort key's final fallback
    # (chunk_id ASC) in isolation, since it cannot be triggered end-to-end
    # through the public fuse_rankings() API with valid inputs.
    candidate_a = _FusedCandidate(
        chunk_id="zzz",
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=None,
        dense_contribution=0.5,
        sparse_contribution=0.0,
        dense_result=None,
        sparse_result=None,
    )
    candidate_b = _FusedCandidate(
        chunk_id="aaa",
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=None,
        dense_contribution=0.5,
        sparse_contribution=0.0,
        dense_result=None,
        sparse_result=None,
    )
    ordered = sorted([candidate_a, candidate_b], key=_sort_key)
    assert [c.chunk_id for c in ordered] == ["aaa", "zzz"]


def test_hybrid_ranks_start_at_one_and_are_sequential() -> None:
    dense = [make_dense_result(chunk_id=f"d{i}", rank=i) for i in range(1, 4)]
    results = _fuse(dense, [])
    assert [r.rank for r in results] == [1, 2, 3]


def test_hybrid_rank_is_independent_of_source_rank() -> None:
    # "target" sits at dense rank 3 (worst of three dense results) but at
    # sparse rank 1 (best) -- the summed contribution still makes it
    # hybrid rank 1, demonstrating hybrid rank is computed from fused
    # rrf_score rather than copied/reused from either source rank.
    dense = [
        make_dense_result(chunk_id="filler-1", rank=1),
        make_dense_result(chunk_id="filler-2", rank=2),
        make_dense_result(chunk_id="target", rank=3),
    ]
    sparse = [make_sparse_result(chunk_id="target", rank=1)]
    results = _fuse(dense, sparse)
    top = results[0]
    assert top.chunk_id == "target"
    assert top.dense_rank == 3
    assert top.sparse_rank == 1
    assert top.rank == 1


def test_top_k_truncates_after_fusion() -> None:
    dense = [make_dense_result(chunk_id=f"d{i}", rank=i) for i in range(1, 6)]
    results = _fuse(dense, [], top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]


def test_top_k_larger_than_union_returns_all() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1)]
    sparse = [make_sparse_result(chunk_id="b", rank=1)]
    results = _fuse(dense, sparse, top_k=1000)
    assert len(results) == 2


def test_same_inputs_always_yield_same_ordering() -> None:
    dense = [make_dense_result(chunk_id=f"d{i}", rank=i) for i in range(1, 5)]
    # Sparse ranks are deliberately the reverse mapping (d4 best, d1
    # worst), but the list itself is still built in ascending-rank order,
    # as _validate_ranking requires.
    sparse = [
        make_sparse_result(chunk_id="d4", rank=1),
        make_sparse_result(chunk_id="d3", rank=2),
        make_sparse_result(chunk_id="d2", rank=3),
        make_sparse_result(chunk_id="d1", rank=4),
    ]
    first = _fuse(dense, sparse)
    second = _fuse(dense, sparse)
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    assert [r.rrf_score for r in first] == [r.rrf_score for r in second]


# --- integrity: malformed single-channel rankings ---------------------------------


def test_duplicate_chunk_id_inside_dense_list_is_rejected() -> None:
    dense = [
        make_dense_result(chunk_id="a", rank=1),
        make_dense_result(chunk_id="a", rank=2),
    ]
    with pytest.raises(FusionError):
        _fuse(dense, [])


def test_duplicate_chunk_id_inside_sparse_list_is_rejected() -> None:
    sparse = [
        make_sparse_result(chunk_id="a", rank=1),
        make_sparse_result(chunk_id="a", rank=2),
    ]
    with pytest.raises(FusionError):
        _fuse([], sparse)


def test_non_positive_dense_rank_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=0)]
    with pytest.raises(FusionError):
        _fuse(dense, [])


def test_non_positive_sparse_rank_is_rejected() -> None:
    sparse = [make_sparse_result(chunk_id="a", rank=-1)]
    with pytest.raises(FusionError):
        _fuse([], sparse)


def test_duplicate_rank_positions_are_rejected() -> None:
    dense = [
        make_dense_result(chunk_id="a", rank=1),
        make_dense_result(chunk_id="b", rank=1),
    ]
    with pytest.raises(FusionError):
        _fuse(dense, [])


def test_non_contiguous_ranking_is_rejected() -> None:
    dense = [
        make_dense_result(chunk_id="a", rank=1),
        make_dense_result(chunk_id="b", rank=3),  # gap: rank 2 missing
    ]
    with pytest.raises(FusionError):
        _fuse(dense, [])


# --- integrity: overlapping provenance mismatch -----------------------------------


def test_overlapping_text_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, text="dense text")]
    sparse = [make_sparse_result(chunk_id="a", rank=1, text="different sparse text")]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_document_id_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, document_id="d" * 64)]
    sparse = [make_sparse_result(chunk_id="a", rank=1, document_id="e" * 64)]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_source_file_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, source_file="alpha.md")]
    sparse = [make_sparse_result(chunk_id="a", rank=1, source_file="beta.md")]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_chunk_index_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, chunk_index=0)]
    sparse = [make_sparse_result(chunk_id="a", rank=1, chunk_index=1)]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_section_heading_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, section_heading="Intro")]
    sparse = [make_sparse_result(chunk_id="a", rank=1, section_heading="Setup")]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_page_number_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, page_number=1)]
    sparse = [make_sparse_result(chunk_id="a", rank=1, page_number=2)]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


def test_overlapping_chunking_strategy_mismatch_is_rejected() -> None:
    dense = [make_dense_result(chunk_id="a", rank=1, chunking_strategy=ChunkingStrategy.RECURSIVE)]
    sparse = [make_sparse_result(chunk_id="a", rank=1, chunking_strategy=ChunkingStrategy.FIXED)]
    with pytest.raises(FusionError):
        _fuse(dense, sparse)


# --- configuration -----------------------------------------------------------------


def test_default_rrf_weights_are_0_7_and_0_3() -> None:
    settings = Settings(_env_file=None)
    assert settings.rrf_dense_weight == pytest.approx(0.7)
    assert settings.rrf_sparse_weight == pytest.approx(0.3)


def test_default_rrf_rank_constant_is_60() -> None:
    settings = Settings(_env_file=None)
    assert settings.rrf_rank_constant == 60


def test_default_hybrid_top_k_is_10() -> None:
    settings = Settings(_env_file=None)
    assert settings.hybrid_top_k == 10


def test_negative_dense_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="rrf_dense_weight must not be negative"):
        Settings(_env_file=None, rrf_dense_weight=-0.1)


def test_negative_sparse_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="rrf_sparse_weight must not be negative"):
        Settings(_env_file=None, rrf_sparse_weight=-0.1)


def test_both_weights_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Settings(_env_file=None, rrf_dense_weight=0.0, rrf_sparse_weight=0.0)


def test_negative_rank_constant_is_rejected() -> None:
    with pytest.raises(ValueError, match="rrf_rank_constant must not be negative"):
        Settings(_env_file=None, rrf_rank_constant=-1)


def test_non_positive_hybrid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="hybrid_top_k must be positive"):
        Settings(_env_file=None, hybrid_top_k=0)
