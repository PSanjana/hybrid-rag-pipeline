"""Production reranker: a local sentence-transformers cross-encoder.

A cross-encoder scores a (query, document) pair jointly through one
transformer forward pass, unlike embedding cosine similarity (which
scores query and document independently and only compares the resulting
vectors) -- this is generally meaningfully more accurate for reranking a
small candidate set, at the cost of being too slow to run over an entire
corpus. That tradeoff is exactly why it only ever runs over the already-
narrowed `rerank_candidate_k` hybrid candidates, downstream of fast
dense/sparse/RRF retrieval.

`sentence-transformers` (and its `torch` dependency) is an optional
extra (`pip install 'rag-pipeline[rerank]'`): importing this module never
imports it or downloads a model. That only happens lazily, on the first
`.score()` call, so importing `rag_pipeline` -- or even this module --
never triggers a network request or a multi-hundred-MB download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .exceptions import RerankerError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_BATCH_SIZE = 32


class CrossEncoderReranker:
    """A `Reranker` backed by a local `sentence_transformers.CrossEncoder` model.

    `model_name` and `batch_size` are configurable (see
    `Settings.reranker_model_name`/`Settings.reranker_batch_size`); the
    underlying model is loaded at most once, lazily, on first use.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "sentence-transformers is required for CrossEncoderReranker but is not "
                    "installed. Install it with: pip install 'rag-pipeline[rerank]'."
                ) from exc
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:
                raise RerankerError(
                    f"Failed to load cross-encoder model {self._model_name!r}: {exc}"
                ) from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, document) for document in documents]
        try:
            raw_scores = model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:
            raise RerankerError(f"Cross-encoder prediction failed: {exc}") from exc
        return [float(s) for s in raw_scores]
