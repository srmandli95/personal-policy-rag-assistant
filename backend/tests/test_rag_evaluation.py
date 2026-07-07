import json

import pytest

from app.evaluation.eval_loader import load_eval_cases
from app.evaluation.eval_metrics import (
    citations_contain_expected_documents,
    citations_contain_expected_document_ids,
    contains_expected_terms,
    evaluate_answer_response,
    is_refusal_response,
)
from app.evaluation.eval_models import EvalCase
from app.evaluation.eval_runner import run_rag_evaluation


def make_case(**overrides) -> EvalCase:
    values = {
        "id": "health_emergency_001",
        "category": "health_insurance",
        "question": "Does my policy cover emergency care?",
        "expected_answer_contains": ["emergency care", "covered"],
        "expected_citation_document_contains": ["sample_health_policy"],
        "expected_refusal": False,
    }
    values.update(overrides)
    return EvalCase(**values)


def supported_response() -> dict:
    return {
        "answer": "Emergency care is covered based on plan rules.",
        "status": "answered",
        "validation_status": "supported",
        "citations": [
            {
                "chunk_id": "chunk-1",
                "document_name": "sample_health_policy.txt",
            }
        ],
        "evidence_chunks": [
            {
                "chunk_id": "chunk-1",
                "document_name": "sample_health_policy.txt",
                "chunk_text": "Emergency care is covered based on plan rules.",
            }
        ],
        "evidence_chunk_count": 1,
    }


def test_load_eval_cases_loads_valid_json(tmp_path):
    eval_file = tmp_path / "cases.json"
    eval_file.write_text(
        json.dumps([make_case().model_dump()]),
        encoding="utf-8",
    )

    cases = load_eval_cases(str(eval_file))

    assert len(cases) == 1
    assert cases[0].id == "health_emergency_001"


def test_load_eval_cases_raises_clear_error_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Evaluation file not found"):
        load_eval_cases(str(missing_file))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"id": "not-a-list"}, "JSON list"),
        ([{"id": "missing-required-fields"}], "Invalid evaluation case at index 0"),
    ],
)
def test_load_eval_cases_validates_structure_and_fields(tmp_path, payload, message):
    eval_file = tmp_path / "invalid.json"
    eval_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_eval_cases(str(eval_file))


def test_load_eval_cases_rejects_invalid_json(tmp_path):
    eval_file = tmp_path / "invalid.json"
    eval_file.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_cases(str(eval_file))


def test_contains_expected_terms_requires_all_terms():
    assert contains_expected_terms(
        "Emergency care is covered based on plan rules.",
        ["emergency care", "covered"],
    )
    assert not contains_expected_terms(
        "Emergency care is covered.",
        ["emergency care", "copay"],
    )


def test_contains_expected_terms_is_case_insensitive():
    assert contains_expected_terms("EMERGENCY CARE IS COVERED", ["Emergency Care"])


def test_citations_contain_expected_documents():
    citations = [
        {"document_name": "sample_health_policy.txt"},
        {"document_name": "sample_employee_benefits.md"},
    ]

    assert citations_contain_expected_documents(
        citations,
        ["health_policy", "employee_benefits"],
    )
    assert not citations_contain_expected_documents(citations, ["auto_policy"])


def test_citations_contain_expected_document_ids():
    citations = [{"document_id": "doc-1"}, {"document_id": "doc-2"}]

    assert citations_contain_expected_document_ids(citations, ["doc-1", "doc-2"])
    assert not citations_contain_expected_document_ids(citations, ["doc-3"])


def test_expected_refusal_passes_when_response_status_is_refused():
    case = make_case(
        id="unsupported_lasik_001",
        expected_answer_contains=[],
        expected_citation_document_contains=[],
        expected_refusal=True,
    )
    response = {
        "answer": "No supporting evidence was found.",
        "status": "refused",
        "validation_status": "unsupported",
        "citations": [],
        "evidence_chunk_count": 0,
    }

    result = evaluate_answer_response(case, response)

    assert result.passed is True
    assert result.actual_refusal is True
    assert result.citation_count == 0


def test_is_refusal_response_recognizes_refusal_language():
    assert is_refusal_response(
        {"answer": "I could not find enough evidence to answer this question."}
    )


def test_non_refusal_fails_when_citations_are_missing():
    response = supported_response()
    response["citations"] = []

    result = evaluate_answer_response(make_case(), response)

    assert result.passed is False
    assert result.checks["citations_present"] is False
    assert "Supported answer did not include citations." in result.failure_reasons


def test_non_refusal_fails_when_citation_ids_do_not_match_evidence():
    response = supported_response()
    response["citations"][0]["chunk_id"] = "missing-chunk"

    result = evaluate_answer_response(make_case(), response)

    assert result.passed is False
    assert result.checks["citation_chunk_ids_valid"] is False


def test_non_refusal_passes_with_supported_answer_citations_and_evidence():
    result = evaluate_answer_response(make_case(), supported_response())

    assert result.passed is True
    assert result.status == "passed"
    assert result.failure_reasons == []
    assert all(result.checks.values())


def test_evaluate_answer_response_returns_useful_failure_reasons():
    response = {
        "answer": "A vague answer.",
        "status": "answered",
        "validation_status": "unsupported",
        "citations": [],
        "evidence_chunks": [],
        "evidence_chunk_count": 0,
    }

    result = evaluate_answer_response(make_case(), response)

    assert result.passed is False
    assert len(result.failure_reasons) >= 4
    assert "Answer did not contain all expected terms." in result.failure_reasons
    assert "Answer validation status was not supported." in result.failure_reasons


def test_run_rag_evaluation_handles_successful_fake_answer(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation.eval_runner.generate_answer_from_evidence",
        lambda **_: supported_response(),
    )

    run = run_rag_evaluation(
        db=object(),
        user_id="local-user-123",
        cases=[make_case()],
    )

    assert run.total == 1
    assert run.passed == 1
    assert run.failed == 0
    assert run.pass_rate == 100.0


def test_run_rag_evaluation_passes_normalized_retrieval_settings(monkeypatch):
    captured = {}

    def fake_generate_answer(**kwargs):
        captured.update(kwargs)
        return supported_response()

    monkeypatch.setattr(
        "app.evaluation.eval_runner.generate_answer_from_evidence",
        fake_generate_answer,
    )

    run_rag_evaluation(
        db=object(),
        user_id="local-user-123",
        cases=[make_case()],
        top_k=4,
        rerank_top_k=9,
        vector_weight=7,
        bm25_weight=3,
        min_reranker_score=0.5,
    )

    assert captured["top_k"] == 4
    assert captured["rerank_top_k"] == 9
    assert captured["vector_weight"] == pytest.approx(0.7)
    assert captured["bm25_weight"] == pytest.approx(0.3)
    assert captured["min_reranker_score"] == 0.5


def test_run_rag_evaluation_catches_exceptions_per_case(monkeypatch):
    calls = 0

    def fake_generate_answer(**_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("retrieval unavailable")
        return supported_response()

    monkeypatch.setattr(
        "app.evaluation.eval_runner.generate_answer_from_evidence",
        fake_generate_answer,
    )

    run = run_rag_evaluation(
        db=object(),
        user_id="local-user-123",
        cases=[make_case(id="error-case"), make_case(id="passing-case")],
    )

    assert run.total == 2
    assert run.passed == 1
    assert run.failed == 1
    assert run.results[0].status == "error"
    assert run.results[0].failure_reasons == [
        "Evaluation error: retrieval unavailable"
    ]
    assert run.results[1].passed is True


def test_run_rag_evaluation_calculates_mixed_pass_rate(monkeypatch):
    responses = [
        supported_response(),
        {
            "answer": "No evidence.",
            "status": "refused",
            "validation_status": "unsupported",
            "citations": [],
            "evidence_chunk_count": 0,
        },
    ]

    monkeypatch.setattr(
        "app.evaluation.eval_runner.generate_answer_from_evidence",
        lambda **_: responses.pop(0),
    )

    run = run_rag_evaluation(
        db=object(),
        user_id="local-user-123",
        cases=[make_case(id="passing-case"), make_case(id="failing-case")],
    )

    assert run.total == 2
    assert run.passed == 1
    assert run.failed == 1
    assert run.pass_rate == 50.0


def test_run_rag_evaluation_empty_cases_has_zero_pass_rate():
    run = run_rag_evaluation(db=object(), user_id="local-user-123", cases=[])

    assert run.total == 0
    assert run.pass_rate == 0.0
