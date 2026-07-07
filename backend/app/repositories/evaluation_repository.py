from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.evaluation_dataset import EvaluationDataset
from app.models.evaluation_run import EvaluationRun


def create_evaluation_dataset(
    db: Session,
    *,
    user_id: str,
    name: str,
    original_file_name: str,
    cases: list[dict[str, Any]],
    document_ids: list[str],
) -> EvaluationDataset:
    dataset = EvaluationDataset(
        user_id=user_id,
        name=name,
        original_file_name=original_file_name,
        cases=cases,
        document_ids=document_ids,
        case_count=len(cases),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def list_evaluation_datasets(db: Session, user_id: str) -> list[EvaluationDataset]:
    return (
        db.query(EvaluationDataset)
        .filter(EvaluationDataset.user_id == user_id)
        .order_by(EvaluationDataset.created_at.desc())
        .all()
    )


def get_evaluation_dataset(
    db: Session,
    dataset_id: str,
    user_id: str,
) -> EvaluationDataset | None:
    return (
        db.query(EvaluationDataset)
        .filter(
            EvaluationDataset.id == dataset_id,
            EvaluationDataset.user_id == user_id,
        )
        .first()
    )


def delete_evaluation_dataset(db: Session, dataset: EvaluationDataset) -> None:
    db.query(EvaluationRun).filter(
        EvaluationRun.dataset_id == dataset.id,
        EvaluationRun.user_id == dataset.user_id,
    ).delete(synchronize_session=False)
    db.delete(dataset)
    db.commit()


def create_evaluation_run(
    db: Session,
    *,
    dataset_id: str,
    user_id: str,
    settings: dict[str, Any],
) -> EvaluationRun:
    run = EvaluationRun(
        dataset_id=dataset_id,
        user_id=user_id,
        status="running",
        settings=settings,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def complete_evaluation_run(
    db: Session,
    run: EvaluationRun,
    result: dict[str, Any],
) -> EvaluationRun:
    run.status = "completed"
    run.result = result
    run.total = int(result.get("total", 0))
    run.passed = int(result.get("passed", 0))
    run.failed = int(result.get("failed", 0))
    run.pass_rate = float(result.get("pass_rate", 0.0))
    run.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(run)
    return run


def fail_evaluation_run(db: Session, run: EvaluationRun, error: str) -> EvaluationRun:
    run.status = "failed"
    run.error_message = error
    run.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(run)
    return run


def list_evaluation_runs(
    db: Session,
    user_id: str,
    dataset_id: str | None = None,
) -> list[EvaluationRun]:
    query = db.query(EvaluationRun).filter(EvaluationRun.user_id == user_id)
    if dataset_id:
        query = query.filter(EvaluationRun.dataset_id == dataset_id)
    return query.order_by(EvaluationRun.created_at.desc()).all()


def get_evaluation_run(
    db: Session,
    run_id: str,
    user_id: str,
) -> EvaluationRun | None:
    return (
        db.query(EvaluationRun)
        .filter(EvaluationRun.id == run_id, EvaluationRun.user_id == user_id)
        .first()
    )

