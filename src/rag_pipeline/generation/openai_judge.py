"""Production citation judge: OpenAI Chat Completions structured outputs.

Mirrors `openai.OpenAIGenerator`'s shape: the client is built once in
`__init__` (never at module import), missing `OPENAI_API_KEY` fails fast
there with a clear project-specific error, and every OpenAI SDK failure
is wrapped with the cause chained. Unlike `OpenAIGenerator`, this
provider asks for *structured*, schema-validated output via
`chat.completions.parse(response_format=...)` rather than free text --
the schema constrains `verdict` to exactly the four `CitationVerdict`
string values, so the API's own structured-output decoding rejects a
malformed verdict before this provider ever sees one. That said,
`verification.verify_grounded_answer()` independently re-validates the
result regardless (occurrence set, citation numbers, rationale) -- this
provider's schema is a quality improvement, never the sole source of
trust (see `base.RawJudgeVerdict`).

Uses Chat Completions (not the Responses API) for the same reason as
`OpenAIGenerator`: consistency with the rest of this project's OpenAI
integration, and a smaller, more stable surface for a single-turn
system+user-prompt request.

Never logs the API key, prompt, evidence, or answer/judgment text.
"""

from __future__ import annotations

from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from ..config import Settings
from .base import RawJudgeVerdict
from .exceptions import CitationJudgeError

_DEFAULT_TEMPERATURE = 0.0

_VerdictLiteral = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class _JudgedOccurrenceSchema(BaseModel):
    occurrence_id: int
    citation_number: int
    verdict: _VerdictLiteral
    rationale: str


class _JudgeResponseSchema(BaseModel):
    verdicts: list[_JudgedOccurrenceSchema]


class OpenAICitationJudge:
    """A `CitationJudge` backed by the OpenAI Chat Completions structured-outputs API.

    Requires `settings.openai_api_key`; raises `CitationJudgeError`
    immediately if it's missing, rather than failing on first use.
    """

    def __init__(self, settings: Settings, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        if settings.openai_api_key is None:
            raise CitationJudgeError(
                "Citation verification requires OPENAI_API_KEY to be configured."
            )
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.citation_judge_model
        self._temperature = temperature

    def judge(self, system_prompt: str, user_prompt: str) -> list[RawJudgeVerdict]:
        try:
            response = self._client.chat.completions.parse(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_JudgeResponseSchema,
            )
        except OpenAIError as exc:
            raise CitationJudgeError(
                f"Citation judge request failed for model {self._model!r}: {exc}"
            ) from exc

        if not response.choices:
            raise CitationJudgeError(
                f"Citation judge returned no choices for model {self._model!r}."
            )

        message = response.choices[0].message
        if message.refusal:
            raise CitationJudgeError(
                f"Citation judge refused to respond for model {self._model!r}."
            )

        parsed = message.parsed
        if parsed is None:
            raise CitationJudgeError(
                f"Citation judge returned no parsed structured output for model {self._model!r}."
            )

        return [
            RawJudgeVerdict(
                occurrence_id=item.occurrence_id,
                citation_number=item.citation_number,
                verdict=item.verdict,
                rationale=item.rationale,
            )
            for item in parsed.verdicts
        ]
