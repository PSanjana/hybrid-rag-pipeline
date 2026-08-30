"""Provider abstractions for generation and citation verification.

`Generator` turns (system prompt, user prompt) into free text and is the
only thing `generation.service` depends on. `CitationJudge` turns the
same (system prompt, user prompt) shape into *structured* per-occurrence
verdicts (`RawJudgeVerdict`) and is the only thing `generation.verification`
depends on. In both cases, a production OpenAI provider and offline test
doubles are interchangeable implementations -- this keeps the generation
and verification services independently testable and never coupled to a
specific SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return generated text for the given system/user prompts.

        Must return the model's raw text response; empty-response and
        provider-failure handling is the provider implementation's
        responsibility (see `GenerationProviderError`).
        """
        ...


@dataclass(frozen=True, slots=True)
class RawJudgeVerdict:
    """One unvalidated verdict as returned by a `CitationJudge` provider, before validation.

    Deliberately untyped/untrusted at this boundary -- `verdict` is a
    raw `str`, not yet a `CitationVerdict` -- because a provider's
    output (real or fake) must never be assumed correct just because it
    came from a judge. `generation.verification.verify_grounded_answer`
    is the only place a `RawJudgeVerdict` is converted into a trusted
    `CitationVerification`, after validating it against the exact
    expected occurrence set.
    """

    occurrence_id: int
    citation_number: int
    verdict: str
    rationale: str


@runtime_checkable
class CitationJudge(Protocol):
    def judge(self, system_prompt: str, user_prompt: str) -> list[RawJudgeVerdict]:
        """Return one raw verdict per citation occurrence described in `user_prompt`.

        Mirrors `Generator.generate()`'s (system prompt, user prompt)
        shape on the input side, but returns structured per-occurrence
        data instead of free text, since a judge's entire job is to
        produce a discrete verdict + rationale per occurrence -- see
        `generation.judge_prompt` for how occurrence identity is made
        explicit in `user_prompt` via a deterministic annotated-answer
        marker. Must not filter, deduplicate, or reorder occurrences on
        its own; output-set validation happens entirely in
        `generation.verification`, never inside a provider.
        """
        ...
