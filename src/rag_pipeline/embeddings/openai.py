"""OpenAI-backed embedding provider — the project's one production implementation.

Shared by semantic chunking and dense indexing; neither owns this module.
"""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI, OpenAIError

from ..config import Settings
from .exceptions import EmbeddingProviderError
from .validation import validate_consistent_dimensionality, validate_vector

_DEFAULT_BATCH_SIZE = 128


class OpenAIEmbeddingProvider:
    """Embedding provider backed by the OpenAI embeddings API.

    Requires `settings.openai_api_key`; raises `EmbeddingProviderError`
    immediately if it's missing, rather than failing on first use. Never
    logs raw text or the API key.

    Per batch response, vectors are reassociated with their input by the
    API's own `index` field (never assumed to already be in request order),
    and validated: exactly one vector per input, every vector non-empty and
    finite, and — across the *entire* call, not just one batch — all
    vectors share one consistent dimensionality. The embedding dimension is
    never hard-coded; it's whatever a valid response turns out to contain.

    "Exactly one vector per input" is enforced by two conditions checked
    together: `len(response.data) == len(batch)`, and the response's index
    values are exactly `{0, ..., len(batch) - 1}`. Neither condition alone
    is sufficient — a response could contain a duplicated index alongside
    an extra item (same *count* as expected, but not one-to-one) or could
    contain every valid index plus one duplicate-and-extra item (same
    *index set* as expected, but more items than inputs). Requiring both
    a matching count and a complete, exact index set is what rules out
    duplicates entirely: with N items and index values covering exactly N
    distinct values 0..N-1, no two items can share an index.
    """

    def __init__(self, settings: Settings, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        if settings.openai_api_key is None:
            raise EmbeddingProviderError(
                "Embedding generation requires OPENAI_API_KEY to be configured."
            )
        if batch_size <= 0:
            raise EmbeddingProviderError("batch_size must be positive.")
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.embedding_model
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
            except OpenAIError as exc:
                raise EmbeddingProviderError(
                    f"Embedding request failed for model {self._model!r}: {exc}"
                ) from exc

            if len(response.data) != len(batch):
                raise EmbeddingProviderError(
                    f"Embedding provider returned {len(response.data)} vectors for "
                    f"{len(batch)} inputs; expected exactly one vector per input."
                )

            by_index = {item.index: item.embedding for item in response.data}
            if set(by_index) != set(range(len(batch))):
                raise EmbeddingProviderError(
                    f"Embedding provider returned vectors for indexes {sorted(by_index)} but "
                    f"{len(batch)} inputs were sent; expected exactly one vector per input "
                    f"index 0..{len(batch) - 1}."
                )
            for index in range(len(batch)):
                vector = by_index[index]
                validate_vector(vector)
                embeddings.append(vector)

        validate_consistent_dimensionality(embeddings)
        return embeddings
