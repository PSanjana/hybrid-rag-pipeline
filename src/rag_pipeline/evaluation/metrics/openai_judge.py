"""Production semantic evaluation judge: OpenAI Chat Completions structured outputs.

Mirrors `generation.openai_judge.OpenAICitationJudge`: the client is built
once in `__init__` (never at import), a missing `OPENAI_API_KEY` fails
fast there with a clear project-specific error, structured output is
requested via `chat.completions.parse(response_format=...)` with a
Pydantic schema, and every OpenAI SDK failure is wrapped with its cause
chained. The schema constrains verdicts to the exact allowed strings, but
`evaluate_correctness()` / `evaluate_faithfulness()` still independently
re-validate the id set, verdict enum, and rationales -- the schema is a
quality improvement, never the sole trust boundary.

One class implements BOTH the `CorrectnessJudge` and `FaithfulnessJudge`
protocols (their prompts stay logically separate; only the transport is
shared). The evaluation model is its own setting
(`Settings.evaluation_judge_model`) so it can be tuned independently of
generation / citation judging.

Never logs the API key, prompts, answer text, evidence, or verdicts.
"""

from __future__ import annotations

from typing import Literal, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from ...config import Settings
from ..exceptions import EvaluationJudgeError
from .correctness import RawCorrectnessAssessment, RawFactVerdict, RawGoldenContradiction
from .faithfulness import RawClaimVerdict

_DEFAULT_TEMPERATURE = 0.0

_FactVerdictLiteral = Literal["correct", "partially_correct", "missing", "contradicted"]
_ClaimVerdictLiteral = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class _FactVerdictSchema(BaseModel):
    fact_id: int
    verdict: _FactVerdictLiteral
    rationale: str


class _GoldenContradictionSchema(BaseModel):
    contradiction_id: int
    claim_text: str
    rationale: str
    conflicting_fact_ids: list[int]


class _CorrectnessResponseSchema(BaseModel):
    verdicts: list[_FactVerdictSchema]
    golden_contradictions: list[_GoldenContradictionSchema]


class _ClaimVerdictSchema(BaseModel):
    claim_id: int
    claim_text: str
    verdict: _ClaimVerdictLiteral
    rationale: str


class _FaithfulnessResponseSchema(BaseModel):
    claims: list[_ClaimVerdictSchema]


_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


class OpenAIEvaluationJudge:
    """A `CorrectnessJudge` + `FaithfulnessJudge` backed by OpenAI structured outputs.

    Requires `settings.openai_api_key`; raises `EvaluationJudgeError`
    immediately if it is missing, rather than on first use.
    """

    def __init__(self, settings: Settings, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        if settings.openai_api_key is None:
            raise EvaluationJudgeError(
                "Semantic evaluation judging requires OPENAI_API_KEY to be configured."
            )
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.evaluation_judge_model
        self._temperature = temperature

    def _parse(self, system_prompt: str, user_prompt: str, schema: type[_SchemaT]) -> _SchemaT:
        try:
            response = self._client.chat.completions.parse(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema,
            )
        except OpenAIError as exc:
            raise EvaluationJudgeError(
                f"Evaluation judge request failed for model {self._model!r}: {exc}"
            ) from exc

        if not response.choices:
            raise EvaluationJudgeError(
                f"Evaluation judge returned no choices for model {self._model!r}."
            )
        message = response.choices[0].message
        if message.refusal:
            raise EvaluationJudgeError(
                f"Evaluation judge refused to respond for model {self._model!r}."
            )
        parsed = message.parsed
        if parsed is None:
            raise EvaluationJudgeError(
                f"Evaluation judge returned no parsed structured output for model {self._model!r}."
            )
        return parsed

    def assess_correctness(self, system_prompt: str, user_prompt: str) -> RawCorrectnessAssessment:
        parsed = self._parse(system_prompt, user_prompt, _CorrectnessResponseSchema)
        return RawCorrectnessAssessment(
            fact_verdicts=[
                RawFactVerdict(fact_id=item.fact_id, verdict=item.verdict, rationale=item.rationale)
                for item in parsed.verdicts
            ],
            golden_contradictions=[
                RawGoldenContradiction(
                    contradiction_id=item.contradiction_id,
                    claim_text=item.claim_text,
                    rationale=item.rationale,
                    conflicting_fact_ids=tuple(item.conflicting_fact_ids),
                )
                for item in parsed.golden_contradictions
            ],
        )

    def assess_faithfulness(self, system_prompt: str, user_prompt: str) -> list[RawClaimVerdict]:
        parsed = self._parse(system_prompt, user_prompt, _FaithfulnessResponseSchema)
        return [
            RawClaimVerdict(
                claim_id=item.claim_id,
                claim_text=item.claim_text,
                verdict=item.verdict,
                rationale=item.rationale,
            )
            for item in parsed.claims
        ]
