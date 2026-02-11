"""Centralized structured logging configuration."""

import logging
import sys
from datetime import datetime, timezone

from src.forecast.config import settings


class JSONFormatter(logging.Formatter):
    """Output log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = self.formatException(record.exc_info)
        return str(log_data)


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger
