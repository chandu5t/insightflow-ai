"""Response models for the health endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    app: str
    version: str
    environment: str