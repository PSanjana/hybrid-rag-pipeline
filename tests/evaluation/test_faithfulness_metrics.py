"""Offline tests for rag_pipeline.evaluation.metrics.faithfulness (fake judge, no network)."""

from __future__ import annotations

import pytest

from rag_pipeline.evaluation.exceptions import EvaluationJudgeError, EvaluationJudgeOutputError
from rag_pipeline.evaluation.metrics.faithfulness import RawClaimVerdict, evaluate_faithfulness
from rag_pipeline.evaluation.models import Answerability, QuestionType
from rag_pipeline.generation.models import AnswerDecision

from .conftest import (
    FakeFaithfulnessJudge,
    make_final_answer,
    make_golden_case,
    make_grounded_answer,
)

_Q = "What is the access token lifetime?"


def _answered(sources: list[str] | None = None) -> object:
    grounded = make_grounded_answer(
        answer_text="Access tokens expire after 60 minutes [1].",
        sources=sources or ["authentication-api.md"],
    )
    return make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)


def test_all_supported_claims_score_one() -> None:
    judge = FakeFaithfulnessJudge(
        [(1, "TTL is 60 min", "supported", "r"), (2, "MFA is required", "supported", "r")]
    )

    report = evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)

    assert report.applicable is True
    assert report.score == 1.0
    assert report.supported_count == 2


def test_partial_claim_lowers_score() -> None:
    judge = FakeFaithfulnessJudge(
        [(1, "c1", "supported", "r"), (2, "c2", "partially_supported", "r")]
    )

    report = evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx(0.75)
    assert report.partially_supported_count == 1


def test_unsupported_claim_lowers_score() -> None:
    judge = FakeFaithfulnessJudge([(1, "c1", "supported", "r"), (2, "c2", "unsupported", "r")])

    report = evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx(0.5)
    assert report.unsupported_count == 1


def test_contradicted_claim_is_retained() -> None:
    judge = FakeFaithfulnessJudge([(1, "c1", "contradicted", "conflicts with evidence")])

    report = evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)

    assert report.score == 0.0
    assert report.contradicted_count == 1
    assert report.claim_assessments[0].verdict.value == "contradicted"


def test_mixed_claims_average_is_correct() -> None:
    judge = FakeFaithfulnessJudge(
        [
            (1, "c1", "supported", "r"),
            (2, "c2", "partially_supported", "r"),
            (3, "c3", "unsupported", "r"),
            (4, "c4", "contradicted", "r"),
        ]
    )

    report = evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)


def test_zero_claims_for_substantive_answered_output_rejected() -> None:
    judge = FakeFaithfulnessJudge([])

    with pytest.raises(EvaluationJudgeOutputError, match="zero claims"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)


def test_duplicate_claim_ids_rejected() -> None:
    judge = FakeFaithfulnessJudge([(1, "c1", "supported", "r"), (1, "c1b", "supported", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="duplicate claim_id"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)


def test_non_contiguous_claim_ids_rejected() -> None:
    judge = FakeFaithfulnessJudge([(1, "c1", "supported", "r"), (3, "c3", "supported", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="contiguous 1"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)


def test_invalid_verdict_rejected() -> None:
    judge = FakeFaithfulnessJudge([(1, "c1", "kinda", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="invalid verdict"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)


def test_blank_claim_text_and_blank_rationale_rejected() -> None:
    blank_text = FakeFaithfulnessJudge([(1, "   ", "supported", "r")])
    with pytest.raises(EvaluationJudgeOutputError, match="empty claim_text"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=blank_text)

    blank_rationale = FakeFaithfulnessJudge([(1, "c1", "supported", "  ")])
    with pytest.raises(EvaluationJudgeOutputError, match="empty rationale"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=blank_rationale)


def test_bool_claim_id_rejected() -> None:
    judge = FakeFaithfulnessJudge(
        raw=[RawClaimVerdict(claim_id=True, claim_text="c", verdict="supported", rationale="r")]
    )

    with pytest.raises(EvaluationJudgeOutputError, match="non-integer claim_id"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=judge)


def test_abstained_final_answer_is_not_applicable_and_judge_not_called() -> None:
    grounded = make_grounded_answer(answer_text="draft [1].", sources=["authentication-api.md"])
    final = make_final_answer(
        decision=AnswerDecision.ABSTAINED_CONTRADICTION,
        grounded=grounded,
        abstention_reason="reason",
    )
    judge = FakeFaithfulnessJudge([(1, "c1", "supported", "r")])

    report = evaluate_faithfulness(question=_Q, final_answer=final, judge=judge)

    assert report.applicable is False
    assert report.score is None
    assert judge.calls == []


def test_answered_unanswerable_case_can_still_receive_faithfulness_evaluation() -> None:
    # golden answerability is irrelevant to faithfulness; only the decision matters
    unanswerable_case = make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
    )
    assert unanswerable_case.answerability is Answerability.UNANSWERABLE
    grounded = make_grounded_answer(
        answer_text="It is 60 minutes [1].", sources=["authentication-api.md"]
    )
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)
    judge = FakeFaithfulnessJudge([(1, "TTL is 60 minutes", "unsupported", "not in evidence")])

    report = evaluate_faithfulness(question=_Q, final_answer=final, judge=judge)

    assert report.applicable is True
    assert report.score == 0.0
    assert judge.calls != []


def test_judge_failure_is_wrapped_and_prompt_forbids_golden_truth() -> None:
    boom = FakeFaithfulnessJudge(error=RuntimeError("boom"))
    with pytest.raises(EvaluationJudgeError, match="Faithfulness judge failed"):
        evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=boom)

    ok = FakeFaithfulnessJudge([(1, "c1", "supported", "r")])
    evaluate_faithfulness(question=_Q, final_answer=_answered(), judge=ok)
    system_prompt, user_prompt = ok.calls[0]
    assert "ONLY against the supplied evidence" in system_prompt
    assert "NO golden or reference answer" in system_prompt
    assert "UNTRUSTED" in user_prompt
    assert "Evidence blocks supplied to the generator" in user_prompt
