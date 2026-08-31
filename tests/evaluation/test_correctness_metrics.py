"""Offline tests for rag_pipeline.evaluation.metrics.correctness (fake judge, no network)."""

from __future__ import annotations

import pytest

from rag_pipeline.evaluation.exceptions import EvaluationJudgeError, EvaluationJudgeOutputError
from rag_pipeline.evaluation.metrics.correctness import (
    RawFactVerdict,
    RawGoldenContradiction,
    evaluate_correctness,
)
from rag_pipeline.evaluation.models import Answerability, QuestionType
from rag_pipeline.generation.models import AnswerDecision

from .conftest import (
    FakeCorrectnessJudge,
    make_final_answer,
    make_golden_case,
    make_grounded_answer,
)


def _answered(answer_text: str = "Access tokens expire after 60 minutes [1].") -> object:
    grounded = make_grounded_answer(answer_text=answer_text, sources=["authentication-api.md"])
    return make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)


def test_all_facts_correct_scores_one() -> None:
    case = make_golden_case(expected_facts=("f1", "f2", "f3"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "correct", "r"), (3, "correct", "r")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.applicable is True
    assert report.score == 1.0
    assert report.expected_fact_score == 1.0
    assert report.correct_count == 3
    assert report.has_golden_contradiction is False
    assert report.golden_contradiction_count == 0
    assert report.golden_contradictions == ()
    assert [a.verdict.value for a in report.fact_assessments] == ["correct", "correct", "correct"]


def test_one_partial_gives_expected_mean() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "partially_correct", "r")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx(0.75)
    assert report.partially_correct_count == 1


def test_missing_fact_lowers_score() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "missing", "r")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx(0.5)
    assert report.missing_count == 1


def test_contradicted_fact_lowers_score_and_count_is_retained() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "contradicted", "conflicts")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.score == pytest.approx(0.5)
    assert report.contradicted_count == 1
    assert report.fact_assessments[1].verdict.value == "contradicted"


def test_judge_output_order_does_not_matter_when_ids_align() -> None:
    case = make_golden_case(expected_facts=("alpha", "beta", "gamma"))
    judge = FakeCorrectnessJudge(
        [(3, "missing", "r"), (1, "correct", "r"), (2, "partially_correct", "r")]
    )

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert [a.fact_id for a in report.fact_assessments] == [1, 2, 3]
    assert [a.fact_text for a in report.fact_assessments] == ["alpha", "beta", "gamma"]
    assert [a.verdict.value for a in report.fact_assessments] == [
        "correct",
        "partially_correct",
        "missing",
    ]


def test_duplicate_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (1, "missing", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="duplicate fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_missing_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1", "f2", "f3"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "correct", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match=r"did not return a verdict for fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_unexpected_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "correct", "r"), (3, "correct", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="unexpected fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_bool_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(
        raw=[RawFactVerdict(fact_id=True, verdict="correct", rationale="r")]
    )

    with pytest.raises(EvaluationJudgeOutputError, match="non-integer fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_invalid_verdict_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge([(1, "mostly_right", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="invalid verdict"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_blank_rationale_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge([(1, "correct", "   ")])

    with pytest.raises(EvaluationJudgeOutputError, match="empty rationale"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_answerable_but_abstained_is_not_applicable_and_judge_not_called() -> None:
    case = make_golden_case(expected_facts=("f1",))
    grounded = make_grounded_answer(answer_text="draft [1].", sources=["authentication-api.md"])
    final = make_final_answer(
        decision=AnswerDecision.ABSTAINED_LOW_CONFIDENCE,
        grounded=grounded,
        abstention_reason="reason",
    )
    judge = FakeCorrectnessJudge([(1, "correct", "r")])

    report = evaluate_correctness(case=case, final_answer=final, judge=judge)

    assert report.applicable is False
    assert report.score is None
    assert report.expected_fact_score is None
    assert report.golden_contradictions == ()
    assert "abstained" in (report.not_applicable_reason or "")
    assert judge.calls == []


def test_unanswerable_case_is_not_applicable_for_correctness() -> None:
    case = make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
    )
    grounded = make_grounded_answer(answer_text="wrongly answered [1].", sources=["a.md"])
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)
    judge = FakeCorrectnessJudge([(1, "correct", "r")])

    report = evaluate_correctness(case=case, final_answer=final, judge=judge)

    assert report.applicable is False
    assert report.score is None
    assert report.expected_fact_score is None
    assert judge.calls == []


def test_judge_failure_is_wrapped_as_evaluation_judge_error() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(error=RuntimeError("boom"))

    with pytest.raises(EvaluationJudgeError, match="Correctness judge failed"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_prompt_states_golden_facts_are_authoritative_and_hides_evidence() -> None:
    case = make_golden_case(expected_facts=("Access tokens expire after 60 minutes",))
    judge = FakeCorrectnessJudge([(1, "correct", "r")])

    evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    system_prompt, user_prompt = judge.calls[0]
    normalised = " ".join(system_prompt.split())
    assert "AUTHORITATIVE" in normalised
    assert "not shown the evidence" in normalised
    assert "Do NOT use outside knowledge" in normalised
    assert "Golden expected facts" in user_prompt
    # contradiction-detection semantics are spelled out separately from faithfulness
    assert "DIRECTLY CONFLICT with the supplied golden truth" in normalised
    assert "report a contradiction merely because an answer statement is absent" in normalised
    assert "The golden facts are NOT exhaustive" in normalised
    assert "faithfulness" in normalised


# --- answer-level golden contradiction detection -------------------------------


def test_all_facts_correct_and_no_extra_claim_has_no_contradiction() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "correct", "r")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.expected_fact_score == 1.0
    assert report.has_golden_contradiction is False
    assert report.golden_contradictions == ()


def test_all_facts_correct_but_extra_contradicting_claim_flagged_while_score_stays_high() -> None:
    case = make_golden_case(expected_facts=("Access tokens expire after 60 minutes",))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "the answer states 60 minutes")],
        contradictions=[
            (
                1,
                "It also says tokens never expire.",
                "golden fact 1 says they expire in 60 min",
                [1],
            )
        ],
    )

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    # expected-fact coverage is untouched -- no numeric penalty for the contradiction
    assert report.expected_fact_score == 1.0
    assert report.score == 1.0
    assert report.correct_count == 1
    assert report.contradicted_count == 0
    # but the contradiction is surfaced as its own signal
    assert report.has_golden_contradiction is True
    assert report.golden_contradiction_count == 1
    assert report.golden_contradictions[0].claim_text == "It also says tokens never expire."
    assert report.golden_contradictions[0].conflicting_fact_ids == (1,)


def test_extra_claim_merely_absent_from_golden_truth_is_not_a_contradiction() -> None:
    case = make_golden_case(expected_facts=("Access tokens expire after 60 minutes",))
    # judge returns an EMPTY contradiction list for an extra-but-not-conflicting statement
    judge = FakeCorrectnessJudge([(1, "correct", "r")], contradictions=[])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.expected_fact_score == 1.0
    assert report.has_golden_contradiction is False
    assert report.golden_contradictions == ()


def test_expected_fact_contradiction_verdict_behaviour_is_unchanged() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge([(1, "correct", "r"), (2, "contradicted", "conflicts with f2")])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.expected_fact_score == pytest.approx(0.5)
    assert report.contradicted_count == 1
    assert report.fact_assessments[1].verdict.value == "contradicted"
    # OUTPUT B is independent and may legitimately be empty here
    assert report.has_golden_contradiction is False


def test_multiple_answer_level_contradictions_retained_independently() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r")],
        contradictions=[
            (1, "claim A", "conflicts with golden truth", [1]),
            (2, "claim B", "also conflicts", []),
        ],
    )

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.golden_contradiction_count == 2
    assert [c.contradiction_id for c in report.golden_contradictions] == [1, 2]
    assert [c.claim_text for c in report.golden_contradictions] == ["claim A", "claim B"]
    assert report.golden_contradictions[1].conflicting_fact_ids == ()


def test_empty_contradiction_list_is_accepted() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge([(1, "correct", "r")], raw_contradictions=[])

    report = evaluate_correctness(case=case, final_answer=_answered(), judge=judge)

    assert report.has_golden_contradiction is False
    assert report.golden_contradictions == ()


def test_duplicate_contradiction_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r")],
        contradictions=[(1, "claim A", "r"), (1, "claim B", "r")],
    )

    with pytest.raises(EvaluationJudgeOutputError, match="duplicate contradiction_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_bool_contradiction_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r")],
        raw_contradictions=[
            RawGoldenContradiction(contradiction_id=True, claim_text="c", rationale="r")
        ],
    )

    with pytest.raises(EvaluationJudgeOutputError, match="non-integer contradiction_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_non_contiguous_contradiction_ids_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r")],
        contradictions=[(1, "claim A", "r"), (3, "claim C", "r")],
    )

    with pytest.raises(EvaluationJudgeOutputError, match="contiguous 1"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_blank_contradiction_claim_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge([(1, "correct", "r")], contradictions=[(1, "   ", "r")])

    with pytest.raises(EvaluationJudgeOutputError, match="empty claim_text"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_blank_contradiction_rationale_rejected() -> None:
    case = make_golden_case(expected_facts=("f1",))
    judge = FakeCorrectnessJudge([(1, "correct", "r")], contradictions=[(1, "claim A", "  ")])

    with pytest.raises(EvaluationJudgeOutputError, match="empty rationale"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_invalid_conflicting_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r"), (2, "correct", "r")],
        contradictions=[(1, "claim A", "r", [5])],  # only facts 1..2 exist
    )

    with pytest.raises(EvaluationJudgeOutputError, match="invalid conflicting_fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_repeated_conflicting_fact_id_rejected() -> None:
    case = make_golden_case(expected_facts=("f1", "f2"))
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r"), (2, "correct", "r")],
        contradictions=[(1, "claim A", "r", [1, 1])],
    )

    with pytest.raises(EvaluationJudgeOutputError, match="repeated conflicting_fact_id"):
        evaluate_correctness(case=case, final_answer=_answered(), judge=judge)


def test_contradiction_list_not_consulted_on_the_not_applicable_path() -> None:
    case = make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
    )
    grounded = make_grounded_answer(answer_text="answered [1].", sources=["a.md"])
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)
    judge = FakeCorrectnessJudge(
        [(1, "correct", "r")], contradictions=[(1, "would-be contradiction", "r")]
    )

    report = evaluate_correctness(case=case, final_answer=final, judge=judge)

    assert report.applicable is False
    assert report.has_golden_contradiction is False
    assert report.golden_contradictions == ()
    assert judge.calls == []
