"""Request, attempt, and response schemas for V2.5 self-correction."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.multi_agent.schemas import MultiAgentResult
from app.planner.schemas import AnalysisPlan
from app.verification.schemas import VerificationResult

TerminalStatus = Literal[
    "not_corrected",
    "corrected",
    "unsupported",
    "uncorrectable",
    "correction_failure",
    "correction_validation_failed",
    "re_execution_failure",
    "budget_exhausted",
    "self_correction_failure",
]
CorrectionErrorCategory = Literal[
    "invalid_correction_request",
    "correction_not_supported",
    "correction_strategy_invalid",
    "correction_application_failed",
    "correction_validation_failed",
    "correction_budget_exhausted",
    "re_execution_failed",
    "verification_failed",
    "invalid_correction_state",
    "self_correction_failure",
]


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    dataset_id: UUID
    analysis_plan: AnalysisPlan
    execution_result: MultiAgentResult
    verification_result: VerificationResult
    correction_budget: int = Field(default=1, ge=0, le=2)


class CorrectionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: CorrectionErrorCategory
    code: str
    message: str


class CorrectionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1)
    trigger_verification: VerificationResult
    strategy_id: str
    eligibility_evidence: dict[str, Any] = Field(default_factory=dict)
    input_plan: AnalysisPlan
    corrected_plan: AnalysisPlan | None = None
    correction_status: Literal["applied", "failed", "validation_failed", "re_execution_failed", "verified", "failed_verification"]
    re_execution_result: MultiAgentResult | None = None
    verification_result: VerificationResult | None = None
    error: CorrectionError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CorrectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terminal_status: TerminalStatus
    terminal_reason: str | None = None
    question: str
    dataset_id: UUID
    original_plan: AnalysisPlan
    final_plan: AnalysisPlan
    original_execution_result: MultiAgentResult
    final_execution_result: MultiAgentResult
    original_verification_result: VerificationResult
    final_verification_result: VerificationResult
    attempts_used: int = Field(ge=0, le=2)
    correction_attempts: list[CorrectionAttempt] = Field(default_factory=list)
    errors: list[CorrectionError] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
