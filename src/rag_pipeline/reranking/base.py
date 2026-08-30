"""Reranker abstraction: score (query, document) pairs without depending on a specific provider.

A `Reranker` assigns exactly one raw relevance score per input document,
in the same order the documents were given -- no hidden sorting, no
score normalization. Callers (see `retrieval.rerank`) depend only on
this protocol, never on a concrete provider, so a production
cross-encoder, a hosted reranking API, or an offline test double are all
interchangeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one raw relevance score per `documents[i]`, in input order.

        Higher is more relevant; the score domain is provider-specific
        (e.g. a cross-encoder logit) and must not be treated as a
        probability. Must not reorder, drop, or pad `documents` -- the
        returned list's length and order always match the input exactly.
        """
        ...
