"""Cross-strategy behavior: canonical type, ID reproducibility, and non-mutation."""

from rag_pipeline.chunking.dispatcher import chunk_document
from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.ingestion.models import NormalizedDocument, Segment

from .conftest import FakeEmbeddingProvider, make_document

_LONG_TEXT = "\n\n".join(f"Paragraph {i}. Some content follows here." for i in range(15))


def _document() -> NormalizedDocument:
    return make_document(
        segments=(Segment(text=_LONG_TEXT, section_heading="Intro", page_number=None),)
    )


def test_all_strategies_return_the_same_canonical_chunk_type() -> None:
    document = _document()
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)

    fixed_chunks = chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=settings)
    recursive_chunks = chunk_document(
        document, strategy=ChunkingStrategy.RECURSIVE, settings=settings
    )
    semantic_chunks = chunk_document(
        document,
        strategy=ChunkingStrategy.SEMANTIC,
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
    )

    for chunks in (fixed_chunks, recursive_chunks, semantic_chunks):
        assert chunks
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_same_document_config_and_strategy_yields_identical_ids() -> None:
    document = _document()
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)

    first = chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
    second = chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_different_strategies_produce_distinct_chunk_ids() -> None:
    document = _document()
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)

    fixed_ids = {
        c.chunk_id
        for c in chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=settings)
    }
    recursive_ids = {
        c.chunk_id
        for c in chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
    }

    assert fixed_ids.isdisjoint(recursive_ids)


def test_max_size_invariant_holds_for_every_strategy() -> None:
    # len(chunk.text) <= chunk_size must hold unconditionally for all three
    # strategies -- this is the core invariant the recursive overlap bug
    # (chunk_size + chunk_overlap) violated.
    document = _document()
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)

    fixed_chunks = chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=settings)
    recursive_chunks = chunk_document(
        document, strategy=ChunkingStrategy.RECURSIVE, settings=settings
    )
    semantic_chunks = chunk_document(
        document,
        strategy=ChunkingStrategy.SEMANTIC,
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
    )

    for strategy_name, chunks in (
        ("fixed", fixed_chunks),
        ("recursive", recursive_chunks),
        ("semantic", semantic_chunks),
    ):
        assert chunks, strategy_name
        assert all(len(chunk.text) <= settings.chunk_size for chunk in chunks), strategy_name
        assert all(chunk.character_count <= settings.chunk_size for chunk in chunks), strategy_name


def test_no_strategy_mutates_the_normalized_document_input() -> None:
    document = _document()
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    original_segments = document.segments

    chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=settings)
    chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
    chunk_document(
        document,
        strategy=ChunkingStrategy.SEMANTIC,
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert document.segments is original_segments
