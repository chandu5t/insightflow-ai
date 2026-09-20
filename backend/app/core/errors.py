"""Standard error format for the whole API: {"error": ..., "code": ..., "details": {...}}."""

import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    # Upload and file problems
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_FILE = "EMPTY_FILE"
    INVALID_ENCODING = "INVALID_ENCODING"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    MISSING_HEADERS = "MISSING_HEADERS"
    DUPLICATE_COLUMNS = "DUPLICATE_COLUMNS"
    NO_DATA_ROWS = "NO_DATA_ROWS"
    TOO_MANY_ROWS = "TOO_MANY_ROWS"
    TOO_MANY_COLUMNS = "TOO_MANY_COLUMNS"
    MALFORMED_ROWS = "MALFORMED_ROWS"
    # Dataset lookup problems
    INVALID_DATASET_ID = "INVALID_DATASET_ID"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    INVALID_PREVIEW_ROWS = "INVALID_PREVIEW_ROWS"
    # Profiling and analysis tool problems (Module 3)
    DATASET_UNREADABLE = "DATASET_UNREADABLE"
    MISSING_COLUMN = "MISSING_COLUMN"
    AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_NUMERIC_VALUES = "INVALID_NUMERIC_VALUES"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    # General problems
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    HTTP_ERROR = "HTTP_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """An expected error that we want to show to the user in a clean way."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ToolError(AppError):
    """An analysis tool cannot give a result (missing column, bad numbers, not enough data).

    If it ever reaches a route, the normal handler returns HTTP 422 in the standard format.
    In Module 4 the question service will catch ToolError and answer with
    status "insufficient_data" instead (decision D-003).
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=422, details=details)

def _error_response(
    status_code: int,
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {"error": message, "code": code.value, "details": details or {}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every error uses the same JSON shape."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # We return only the field name and message, never the submitted values.
        errors = [
            {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return _error_response(
            422, ErrorCode.VALIDATION_ERROR, "The request is not valid.", {"errors": errors}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.METHOD_NOT_ALLOWED}
        code = codes.get(exc.status_code, ErrorCode.HTTP_ERROR)
        headers = dict(exc.headers) if exc.headers else None
        return _error_response(exc.status_code, code, str(exc.detail), headers=headers)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # The real reason goes to the log only. The user gets a safe, generic message.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error_response(
            500, ErrorCode.INTERNAL_ERROR, "Something went wrong on the server."
        )