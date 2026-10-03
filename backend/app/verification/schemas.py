"""Pydantic contracts for V2.4 verification requests and outcomes."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.planner.schemas import AnalysisPlan

VerificationStatus = Literal["passed", "failed", "unsupported"]
VerificationCategory = Literal[
    "invalid_result_schema",
    "plan_execution_mismatch",
    "numerical_mismatch",
    "unsupported_claim",
    "consistency_violation",
    "verification_semantics_undefined",
    "verification_failure",
]


class VerificationRequest(BaseModel):
    """The V2.3 result is raw JSON here so the verifier can report schema errors."""

    model_config = ConfigDict(extra="forbid")

    analysis_plan: AnalysisPlan
    multi_agent_result: Any
    dataset_id: UUID | None = None


class VerificationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: VerificationCategory
    code: str
    message: str
    step_id: str | None = None


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: VerificationStatus
    message: str
    category: VerificationCategory | None = None
    step_ids: list[str] = Field(default_factory=list)
    expected: Any = None
    observed: Any = None
    evidence_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_version: str = "v2.4"
    request_id: str
    tolerance_absolute: float = 1e-9


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    checks: list[VerificationCheck] = Field(default_factory=list)
    failed_checks: list[VerificationCheck] = Field(default_factory=list)
    unsupported_checks: list[VerificationCheck] = Field(default_factory=list)
    verified_steps: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    errors: list[VerificationError] = Field(default_factory=list)
    metadata: VerificationMetadata

