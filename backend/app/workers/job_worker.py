import time
from datetime import datetime, timedelta

from sqlalchemy import or_

from app.config.settings import settings
from app.db.database import SessionLocal
from app.evaluation.eval_models import EvalCase
from app.evaluation.eval_runner import run_rag_evaluation
from app.ingestion.document_processing_service import process_document
from app.models.document_processing_job import DocumentProcessingJob
from app.models.evaluation_dataset import EvaluationDataset
from app.models.evaluation_run import EvaluationRun
from app.repositories.document_repository import get_document_by_id
from app.repositories.evaluation_repository import complete_evaluation_run, fail_evaluation_run
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _claim_document_job(db):
    stale_before = datetime.utcnow() - timedelta(minutes=settings.JOB_STALE_AFTER_MINUTES)
    job = (
        db.query(DocumentProcessingJob)
        .filter(
            or_(
                DocumentProcessingJob.status == "pending",
                (DocumentProcessingJob.status == "running")
                & (DocumentProcessingJob.started_at < stale_before),
            )
        )
        .order_by(DocumentProcessingJob.created_at)
        .with_for_update(skip_locked=True)
        .first()
    )
    if job:
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
    return job


def _claim_evaluation_run(db):
    run = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.status == "pending")
        .order_by(EvaluationRun.created_at)
        .with_for_update(skip_locked=True)
        .first()
    )
    if run:
        run.status = "running"
        run.started_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
    return run


def run_one_job() -> bool:
    """Claim and execute one job. Row locking permits multiple worker replicas."""
    db = SessionLocal()
    try:
        job = _claim_document_job(db)
        if job:
            document = get_document_by_id(db, job.document_id, job.user_id)
            if document is None:
                job.status = "failed"
                job.error_message = "Document no longer exists"
                job.completed_at = datetime.utcnow()
                db.commit()
            else:
                process_document(db, document, force=job.force, job=job)
            return True

        run = _claim_evaluation_run(db)
        if not run:
            return False
        dataset = (
            db.query(EvaluationDataset)
            .filter(EvaluationDataset.id == run.dataset_id, EvaluationDataset.user_id == run.user_id)
            .first()
        )
        if dataset is None:
            fail_evaluation_run(db, run, "Evaluation dataset no longer exists")
            return True
        try:
            result = run_rag_evaluation(
                db=db,
                user_id=run.user_id,
                cases=[EvalCase.model_validate(case) for case in dataset.cases],
                **(run.settings or {}),
            )
            result.run_id = str(run.id)
            result.user_id = run.user_id
            result.eval_file = dataset.original_file_name
            complete_evaluation_run(db, run, result.model_dump(mode="json"))
        except Exception as exc:
            logger.exception("Evaluation worker failed: run_id=%s", run.id)
            fail_evaluation_run(db, run, str(exc))
        return True
    finally:
        db.close()


def main() -> None:
    logger.info("Job worker started")
    while True:
        try:
            if not run_one_job():
                time.sleep(max(0.1, settings.JOB_POLL_INTERVAL_SECONDS))
        except Exception:
            logger.exception("Job worker iteration failed")
            time.sleep(max(0.1, settings.JOB_POLL_INTERVAL_SECONDS))


if __name__ == "__main__":
    main()
