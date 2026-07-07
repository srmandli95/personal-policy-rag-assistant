import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.evaluation.eval_models import EvalCase
from app.evaluation.eval_runner import run_rag_evaluation
from app.models.evaluation_dataset import EvaluationDataset
from app.models.evaluation_run import EvaluationRun
from app.models.user import User
from app.repositories.document_repository import get_document_by_id
from app.repositories.evaluation_repository import (
    complete_evaluation_run,
    create_evaluation_dataset,
    create_evaluation_run,
    delete_evaluation_dataset,
    fail_evaluation_run,
    get_evaluation_dataset,
    get_evaluation_run,
    list_evaluation_datasets,
    list_evaluation_runs,
)
from app.schemas.evaluation_schema import (
    EvaluationDatasetDetail,
    EvaluationDatasetListResponse,
    EvaluationDatasetSummary,
    EvaluationRunListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)


router = APIRouter(prefix="/evaluations", tags=["Evaluations"])
MAX_EVAL_FILE_BYTES = 2 * 1024 * 1024
MAX_EVAL_CASES = 250


def _dataset_summary(dataset: EvaluationDataset) -> EvaluationDatasetSummary:
    return EvaluationDatasetSummary(
        dataset_id=str(dataset.id),
        name=dataset.name,
        original_file_name=dataset.original_file_name,
        case_count=dataset.case_count,
        document_ids=dataset.document_ids or [],
        created_at=dataset.created_at,
    )


def _dataset_detail(dataset: EvaluationDataset) -> EvaluationDatasetDetail:
    return EvaluationDatasetDetail(
        **_dataset_summary(dataset).model_dump(),
        cases=[EvalCase.model_validate(case) for case in dataset.cases],
    )


def _run_response(run: EvaluationRun) -> EvaluationRunResponse:
    result = run.result or {}
    return EvaluationRunResponse(
        run_id=str(run.id),
        dataset_id=str(run.dataset_id),
        status=run.status,
        total=run.total,
        passed=run.passed,
        failed=run.failed,
        pass_rate=run.pass_rate,
        settings=run.settings or {},
        results=result.get("results") or [],
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


def _owned_dataset_or_404(db: Session, dataset_id: str, user_id: str) -> EvaluationDataset:
    dataset = get_evaluation_dataset(db, dataset_id, user_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation dataset not found")
    return dataset


def _parse_eval_payload(payload: Any, supplied_name: str | None) -> tuple[str, list[Any]]:
    if isinstance(payload, list):
        cases = payload
        payload_name = None
    elif isinstance(payload, dict):
        cases = payload.get("cases")
        payload_name = payload.get("name")
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Evaluation JSON must be a case array or an object containing a cases array",
        )

    if not isinstance(cases, list) or not cases:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Evaluation dataset must contain at least one case",
        )
    if len(cases) > MAX_EVAL_CASES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Evaluation dataset cannot contain more than {MAX_EVAL_CASES} cases",
        )

    clean_name = (supplied_name or payload_name or "Golden evaluation").strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="Evaluation dataset name is required")
    return clean_name[:160], cases


def _validate_cases(db: Session, user_id: str, raw_cases: list[Any]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        try:
            case = EvalCase.model_validate(raw_case)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid evaluation case at index {index}: {exc.errors()}",
            ) from exc

        case.id = case.id.strip()
        case.question = case.question.strip()
        if not case.id or not case.question:
            raise HTTPException(status_code=422, detail=f"Case at index {index} requires an id and question")
        if case.id in seen_ids:
            raise HTTPException(status_code=422, detail=f"Duplicate evaluation case id: {case.id}")
        seen_ids.add(case.id)

        case.expected_document_ids = list(dict.fromkeys(case.expected_document_ids))
        if case.expected_refusal and case.expected_document_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Refusal case '{case.id}' cannot contain expected_document_ids",
            )
        if not case.expected_refusal and not case.expected_document_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Supported case '{case.id}' requires at least one expected_document_id",
            )

        for document_id in case.expected_document_ids:
            document = get_document_by_id(db=db, document_id=document_id, user_id=user_id)
            if document is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Case '{case.id}' references an unknown document ID: {document_id}",
                )
            if document.status != "embedded":
                raise HTTPException(
                    status_code=422,
                    detail=f"Case '{case.id}' references a document that is not ready: {document_id}",
                )

        cases.append(case)

    return cases


@router.post("/datasets", response_model=EvaluationDatasetDetail, status_code=status.HTTP_201_CREATED)
async def upload_evaluation_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationDatasetDetail:
    """Validate and persist a user-owned golden evaluation JSON file."""
    if not file.filename or Path(file.filename).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Evaluation file must be JSON")

    contents = await file.read(MAX_EVAL_FILE_BYTES + 1)
    if len(contents) > MAX_EVAL_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Evaluation file exceeds the 2 MB limit")

    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Evaluation file contains invalid JSON") from exc

    clean_name, raw_cases = _parse_eval_payload(payload, name)
    user_id = str(current_user.id)
    cases = _validate_cases(db, user_id, raw_cases)
    document_ids = sorted({document_id for case in cases for document_id in case.expected_document_ids})
    dataset = create_evaluation_dataset(
        db,
        user_id=user_id,
        name=clean_name,
        original_file_name=Path(file.filename).name,
        cases=[case.model_dump() for case in cases],
        document_ids=document_ids,
    )
    return _dataset_detail(dataset)


@router.get("/datasets", response_model=EvaluationDatasetListResponse)
def get_evaluation_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationDatasetListResponse:
    datasets = list_evaluation_datasets(db, str(current_user.id))
    return EvaluationDatasetListResponse(datasets=[_dataset_summary(dataset) for dataset in datasets])


@router.get("/datasets/{dataset_id}", response_model=EvaluationDatasetDetail)
def get_evaluation_dataset_detail(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationDatasetDetail:
    return _dataset_detail(_owned_dataset_or_404(db, dataset_id, str(current_user.id)))


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_evaluation_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    dataset = _owned_dataset_or_404(db, dataset_id, str(current_user.id))
    delete_evaluation_dataset(db, dataset)


@router.post("/datasets/{dataset_id}/runs", response_model=EvaluationRunResponse)
def run_evaluation_dataset(
    dataset_id: str,
    request: EvaluationRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationRunResponse:
    user_id = str(current_user.id)
    dataset = _owned_dataset_or_404(db, dataset_id, user_id)
    settings = request.model_dump()
    run = create_evaluation_run(db, dataset_id=dataset_id, user_id=user_id, settings=settings)

    try:
        result = run_rag_evaluation(
            db=db,
            user_id=user_id,
            cases=[EvalCase.model_validate(case) for case in dataset.cases],
            **settings,
        )
        result.run_id = str(run.id)
        result.user_id = user_id
        result.eval_file = dataset.original_file_name
        run = complete_evaluation_run(db, run, result.model_dump(mode="json"))
    except Exception as exc:
        fail_evaluation_run(db, run, str(exc))
        raise HTTPException(status_code=500, detail=f"Evaluation run failed: {exc}") from exc

    return _run_response(run)


@router.get("/runs", response_model=EvaluationRunListResponse)
def get_evaluation_runs(
    dataset_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationRunListResponse:
    runs = list_evaluation_runs(db, str(current_user.id), dataset_id)
    return EvaluationRunListResponse(runs=[_run_response(run) for run in runs])


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run_detail(
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationRunResponse:
    run = get_evaluation_run(db, run_id, str(current_user.id))
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return _run_response(run)

