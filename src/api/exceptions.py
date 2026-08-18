"""
Custom exception handlers for the API.
"""

from __future__ import annotations
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from src.api.schemas import ErrorResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _error_response(
    status_code: int,
    error: str,
    message: str
) -> JSONResponse:
    """
    Build a standardized error response.
    """

    return JSONResponse(
        status_code= status_code,
        content=ErrorResponse(
            error= error,
            message= message,
            status_code= status_code
        ).model_dump(),
    )

async def value_error_handler(
    request: Request,
    exc: ValueError,
) -> JSONResponse:

    logger.exception(exc)

    return _error_response(
        status_code= status.HTTP_400_BAD_REQUEST,
        error= "Validation Error",
        message= str(exc)
    )

async def file_not_found_handler(
    request: Request,
    exc: FileNotFoundError,
) -> JSONResponse:

    logger.exception(exc)

    return _error_response(
        status_code= status.HTTP_404_NOT_FOUND,
        error= "File Not Found",
        message= str(exc),
    )


async def internal_server_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:

    logger.exception(exc)

    return _error_response(
        status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
        error= "Internal Server Error",
        message= "An unexpected error occurred.",
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register application exception handlers.
    """

    app.add_exception_handler(
        ValueError,
        value_error_handler,
    )

    app.add_exception_handler(
        FileNotFoundError,
        file_not_found_handler,
    )

    app.add_exception_handler(
        Exception,
        internal_server_error_handler,
    )