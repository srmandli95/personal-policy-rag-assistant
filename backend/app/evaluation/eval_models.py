from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """Input case used to evaluate RAG answer quality."""
    id: str
    category: str
    question: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_citation_document_contains: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_refusal: bool = False


class EvalCaseResult(BaseModel):
    """Evaluation result for a single case."""
    id: str
    question: str
    status: str
    passed: bool
    answer: str = ""
    validation_status: str | None = None
    expected_refusal: bool
    actual_refusal: bool
    citation_count: int = 0
    evidence_chunk_count: int = 0
    checks: dict[str, bool] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Aggregate result for a complete evaluation run."""
    run_id: str | None = None
    user_id: str | None = None
    eval_file: str | None = None
    created_at: str | None = None
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalCaseResult] = Field(default_factory=list)

