"""Tests for rag_pipeline.chunking.semantic.

All tests use `FakeEmbeddingProvider` (see conftest.py): deterministic,
in-process, no network access and no OpenAI API key required.
"""

from rag_pipeline.chunking.semantic import chunk_semantic
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.ingestion.models import Segment

from .conftest import FakeEmbeddingProvider, make_document


def test_similar_embeddings_force_no_topic_boundary() -> None:
    document = make_document(
        segments=(
            Segment(
                text="First idea here.\n\nSecond idea here.\n\nThird idea here.",
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)
    provider = FakeEmbeddingProvider(vector_for=lambda _text: [1.0, 0.0])  # all identical

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert len(chunks) == 1
    assert "First idea here." in chunks[0].text
    assert "Second idea here." in chunks[0].text
    assert "Third idea here." in chunks[0].text


def test_dissimilar_embeddings_force_a_topic_boundary() -> None:
    document = make_document(
        segments=(
            Segment(
                text="Alpha topic content.\n\nBeta topic content.",
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)

    def vector_for(text: str) -> list[float]:
        return [1.0, 0.0] if "Alpha" in text else [0.0, 1.0]

    provider = FakeEmbeddingProvider(vector_for=vector_for)

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert len(chunks) == 2
    assert "Alpha" in chunks[0].text
    assert "Beta" in chunks[1].text


def test_preserves_all_source_content_in_logical_order() -> None:
    units = [f"Unit {i} content." for i in range(6)]
    document = make_document(
        segments=(Segment(text="\n\n".join(units), section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)
    provider = FakeEmbeddingProvider(vector_for=lambda _text: [1.0, 0.0])

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    combined = "\n\n".join(chunk.text for chunk in chunks)
    for unit in units:
        assert unit in combined
    # Order preserved: each unit's position in the combined text is increasing.
    positions = [combined.index(unit) for unit in units]
    assert positions == sorted(positions)


def test_maximum_size_safeguard_forces_boundary_even_with_high_similarity() -> None:
    units = [f"Unit {i} " + ("word " * 10) for i in range(10)]
    document = make_document(
        segments=(Segment(text="\n\n".join(units), section_heading=None, page_number=None),)
    )
    # All embeddings identical (similarity == 1.0, never below threshold),
    # so only the size safeguard can create boundaries.
    settings = Settings(
        _env_file=None, chunk_size=120, chunk_overlap=20, chunk_semantic_similarity_threshold=0.5
    )
    provider = FakeEmbeddingProvider(vector_for=lambda _text: [1.0, 0.0])

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert len(chunks) > 1
    assert all(chunk.character_count <= 120 for chunk in chunks)


def test_max_size_invariant_holds_including_single_oversized_unit() -> None:
    # A single semantic unit (paragraph) that's already larger than
    # chunk_size on its own must still be split down to the limit, not just
    # groups of multiple units.
    huge_unit = "word " * 100  # ~500 chars, no internal blank-line breaks
    document = make_document(
        segments=(Segment(text=huge_unit, section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    provider = FakeEmbeddingProvider(vector_for=lambda _text: [1.0, 0.0])

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert chunks
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_provenance_preserved_including_pdf_page_number() -> None:
    document = make_document(
        file_type="pdf",
        segments=(
            Segment(
                text="Alpha content.\n\nMore alpha content.",
                section_heading=None,
                page_number=5,
            ),
        ),
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)
    provider = FakeEmbeddingProvider(vector_for=lambda _text: [1.0, 0.0])

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert chunks
    assert all(chunk.page_number == 5 for chunk in chunks)
    assert all(chunk.chunking_strategy == ChunkingStrategy.SEMANTIC for chunk in chunks)


def test_deterministic_fake_embeddings_produce_deterministic_chunks() -> None:
    document = make_document(
        segments=(
            Segment(
                text="Alpha one.\n\nAlpha two.\n\nBeta one.\n\nBeta two.",
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)

    def vector_for(text: str) -> list[float]:
        return [1.0, 0.0] if "Alpha" in text else [0.0, 1.0]

    first = chunk_semantic(document, settings, embedding_provider=FakeEmbeddingProvider(vector_for))
    second = chunk_semantic(
        document, settings, embedding_provider=FakeEmbeddingProvider(vector_for)
    )

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_single_unit_segment_produces_one_chunk_without_embedding_call() -> None:
    document = make_document(
        segments=(
            Segment(text="Just one paragraph, no breaks.", section_heading=None, page_number=None),
        )
    )
    settings = Settings(_env_file=None, chunk_size=1000, chunk_semantic_similarity_threshold=0.5)
    provider = FakeEmbeddingProvider()

    chunks = chunk_semantic(document, settings, embedding_provider=provider)

    assert len(chunks) == 1
    assert chunks[0].text == "Just one paragraph, no breaks."
    assert provider.calls == []  # no embedding call needed for a single unit


def test_empty_segment_produces_no_chunks() -> None:
    document = make_document(segments=(Segment(text="", section_heading=None, page_number=None),))
    settings = Settings(_env_file=None, chunk_size=1000)
    provider = FakeEmbeddingProvider()

    assert chunk_semantic(document, settings, embedding_provider=provider) == []


def test_no_network_or_api_key_required_with_injected_provider() -> None:
    # settings.openai_api_key is None (default), yet this must work because
    # a fake provider is injected — proving semantic chunking itself never
    # requires network access or an API key when a provider is supplied.
    document = make_document(
        segments=(Segment(text="Some content here.", section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None

    chunks = chunk_semantic(document, settings, embedding_provider=FakeEmbeddingProvider())
    assert chunks
