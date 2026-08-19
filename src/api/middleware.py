"""
API middleware.
"""

from __future__ import annotations
import time
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Log incoming requests and outgoing responses.
    """

    async def dispatch(self, request, call_next):

        start_time = time.perf_counter()

        client = (
            request.client.host
            if request.client
            else "unknown"
        )

        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)

        logger.info(
            "Incoming request: %s %s from %s",
            request.method,
            request.url.path,
            client
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter()- start_time) * 1000

        logger.info(
            "Completed request: %s %s -> %d (%.2f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            client
        )

        return response