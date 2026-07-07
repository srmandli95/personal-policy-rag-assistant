from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.eval_models import EvalCase, EvalCaseResult


class EvaluationDatasetSummary(BaseModel):
    dataset_id: str
    name: str
    original_file_name: str
    case_count: int
    document_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class EvaluationDatasetDetail(EvaluationDatasetSummary):
    cases: list[EvalCase]


class EvaluationDatasetListResponse(BaseModel):
    datasets: list[EvaluationDatasetSummary]


class EvaluationRunRequest(BaseModel):
    top_k: int = 5
    hybrid_top_k: int = 20
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    min_reranker_score: float | None = None


class EvaluationRunResponse(BaseModel):
    run_id: str
    dataset_id: str
    status: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    settings: dict[str, Any] = Field(default_factory=dict)
    results: list[EvalCaseResult] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class EvaluationRunListResponse(BaseModel):
    runs: list[EvaluationRunResponse]

