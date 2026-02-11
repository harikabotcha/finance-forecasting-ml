"""Custom middleware for request logging and timing."""

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.forecast.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every incoming request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "%s %s — %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response
