"""Additive V2.4 endpoint for verifying structured V2.3 workflow results."""

from fastapi import APIRouter, Depends

from app.api.dataset_routes import get_dataset_repository
from app.services.dataset_repository import DatasetRepository
from app.verification.schemas import VerificationRequest, VerificationResult
from app.verification.service import verify_execution

router = APIRouter(prefix="/analysis", tags=["Verification v2.4"])


@router.post("/verify", response_model=VerificationResult)
def verify_multi_agent_result(
    request: VerificationRequest,
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> VerificationResult:
    return verify_execution(
        request.analysis_plan,
        request.multi_agent_result,
        repository=repository,
        dataset_id=request.dataset_id,
    )

