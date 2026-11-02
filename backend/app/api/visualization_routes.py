"""Additive V2.6 visualization API."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.visualization.schemas import VisualizationRequest, VisualizationResponse
from app.visualization.service import visualize

router = APIRouter(prefix="/analysis", tags=["Visualization v2.6"])


@router.post("/visualize", response_model=VisualizationResponse)
def visualize_analysis(
    request: VisualizationRequest,
    settings: Settings = Depends(get_settings),
) -> VisualizationResponse:
    if len(request.question) > settings.max_question_length:
        from app.visualization.schemas import VisualizationError

        return VisualizationResponse(
            status="failed",
            errors=[VisualizationError(
                category="invalid_visualization_input",
                code="QUESTION_TOO_LONG",
                message=f"The question exceeds the configured maximum of {settings.max_question_length} characters.",
            )],
        )
    if not request.question.strip():
        from app.visualization.schemas import VisualizationError

        return VisualizationResponse(
            status="failed",
            errors=[VisualizationError(
                category="invalid_visualization_input",
                code="EMPTY_QUESTION",
                message="A non-empty question is required.",
            )],
        )
    return visualize(request)
