import json
from datetime import datetime
from types import SimpleNamespace

from app.api import evaluation_routes
from app.evaluation.eval_models import EvalCaseResult, EvalRunResult
from conftest import setup_auth_overrides


def _dataset(cases):
    return SimpleNamespace(
        id="dataset-1",
        user_id="user-1",
        name="Policy checks",
        original_file_name="golden.json",
        cases=cases,
        document_ids=["doc-1"],
        case_count=len(cases),
        created_at=datetime.utcnow(),
    )


def _supported_case():
    return {
        "id": "coverage-001",
        "category": "general",
        "question": "Is emergency care covered?",
        "expected_answer_contains": ["covered"],
        "expected_document_ids": ["doc-1"],
        "expected_refusal": False,
    }


def test_upload_evaluation_dataset_validates_owned_ready_documents(client, monkeypatch):
    setup_auth_overrides("user-1")
    monkeypatch.setattr(
        evaluation_routes,
        "get_document_by_id",
        lambda **_: SimpleNamespace(id="doc-1", status="embedded"),
    )
    monkeypatch.setattr(
        evaluation_routes,
        "create_evaluation_dataset",
        lambda _db, **values: _dataset(values["cases"]),
    )

    response = client.post(
        "/evaluations/datasets",
        data={"name": "Policy checks"},
        files={"file": ("golden.json", json.dumps([_supported_case()]), "application/json")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["dataset_id"] == "dataset-1"
    assert payload["document_ids"] == ["doc-1"]
    assert payload["cases"][0]["question"] == "Is emergency care covered?"


def test_upload_evaluation_dataset_rejects_unknown_document(client, monkeypatch):
    setup_auth_overrides("user-1")
    monkeypatch.setattr(evaluation_routes, "get_document_by_id", lambda **_: None)

    response = client.post(
        "/evaluations/datasets",
        files={"file": ("golden.json", json.dumps([_supported_case()]), "application/json")},
    )

    assert response.status_code == 422
    assert "unknown document ID" in response.json()["detail"]


def test_upload_evaluation_dataset_rejects_documents_on_refusal_case(client):
    setup_auth_overrides("user-1")
    case = _supported_case()
    case["expected_refusal"] = True

    response = client.post(
        "/evaluations/datasets",
        files={"file": ("golden.json", json.dumps([case]), "application/json")},
    )

    assert response.status_code == 422
    assert "cannot contain expected_document_ids" in response.json()["detail"]


def test_run_evaluation_dataset_returns_persisted_results(client, monkeypatch):
    setup_auth_overrides("user-1")
    dataset = _dataset([_supported_case()])
    running = SimpleNamespace(
        id="run-1",
        dataset_id="dataset-1",
        user_id="user-1",
        status="running",
        settings={},
        result=None,
        total=0,
        passed=0,
        failed=0,
        pass_rate=0.0,
        error_message=None,
        started_at=datetime.utcnow(),
        completed_at=None,
    )
    result = EvalRunResult(
        total=1,
        passed=1,
        failed=0,
        pass_rate=100.0,
        results=[
            EvalCaseResult(
                id="coverage-001",
                question="Is emergency care covered?",
                status="passed",
                passed=True,
                expected_refusal=False,
                actual_refusal=False,
            )
        ],
    )

    monkeypatch.setattr(evaluation_routes, "get_evaluation_dataset", lambda *_: dataset)
    monkeypatch.setattr(evaluation_routes, "create_evaluation_run", lambda *_args, **_kwargs: running)
    monkeypatch.setattr(evaluation_routes, "run_rag_evaluation", lambda **_: result)

    def complete(_db, run, result_payload):
        run.status = "completed"
        run.result = result_payload
        run.total = result_payload["total"]
        run.passed = result_payload["passed"]
        run.failed = result_payload["failed"]
        run.pass_rate = result_payload["pass_rate"]
        run.completed_at = datetime.utcnow()
        return run

    monkeypatch.setattr(evaluation_routes, "complete_evaluation_run", complete)

    response = client.post("/evaluations/datasets/dataset-1/runs", json={})

    assert response.status_code == 200
    assert response.json()["pass_rate"] == 100.0
    assert response.json()["results"][0]["passed"] is True

